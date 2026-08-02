"""
llm/client.py
-------------
Thin wrapper around the Ollama HTTP API for Gemma 4 (gemma4:4b).

Two public functions (system-design.md §1.5):
  get_next_question(history, role)  → str      (per-turn plain text call)
  score_session(history, session_id, role) → dict  (end-of-session strict JSON)

The Ollama endpoint is assumed to be at localhost:11434 (default).
Override via the OLLAMA_BASE_URL environment variable.

Retry-once-on-malformed-JSON (system-design.md §3):
  score_session() sends a stricter "return only JSON" reminder on the first
  parse failure, then falls back to the template scorecard if it fails again.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import requests

from llm.parser import (
    ParseError,
    build_fallback_scorecard,
    parse_question_and_tool_call,
    parse_score_json,
)
from llm.prompts import (
    SCORING_JSON_SCHEMA,
    build_coach_chat_prompt,
    build_model_answers_prompt,
    build_resume_improvement_prompt,
    build_scoring_prompt,
    build_system_prompt,
    format_history_for_prompt,
)

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_TAG = os.environ.get("OLLAMA_MODEL", "gemma4:4b")
_ACTIVE_MODEL_TAG = MODEL_TAG
_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))  # seconds


def get_active_model_tag() -> str:
    """Return the currently active Gemma model tag."""
    return _ACTIVE_MODEL_TAG


# ── Health check (called at startup by app.py) ────────────────────────────────

def check_ollama_ready() -> tuple[bool, str]:
    """
    Return ``(True, "")`` if Ollama is running and a Gemma 4 model is pulled.
    Return ``(False, error_message)`` otherwise.
    Automatically sets ``_ACTIVE_MODEL_TAG`` to the best available Gemma 4 model.
    """
    global _ACTIVE_MODEL_TAG
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        tags = resp.json()
        model_names = [m["name"] for m in tags.get("models", [])]

        if not model_names:
            return False, f"No models found in Ollama at {OLLAMA_BASE_URL}. Run: ollama pull gemma4:12b"

        # Check for exact match
        if MODEL_TAG in model_names:
            _ACTIVE_MODEL_TAG = MODEL_TAG
            return True, ""

        # Check for any gemma4 variant (e.g. gemma4:12b, gemma4:e4b, etc.)
        gemma_models = [m for m in model_names if "gemma4" in m or "gemma" in m]
        if gemma_models:
            _ACTIVE_MODEL_TAG = gemma_models[0]
            logger.info("Using available Gemma model tag: '%s'", _ACTIVE_MODEL_TAG)
            return True, ""

        # Fallback: use the first available model if any exists
        _ACTIVE_MODEL_TAG = model_names[0]
        logger.warning("No Gemma 4 model explicitly matched. Using first available model: '%s'", _ACTIVE_MODEL_TAG)
        return True, ""

    except requests.exceptions.ConnectionError:
        return False, (
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: run 'ollama serve' in a terminal."
        )
    except Exception as exc:
        return False, f"Ollama health check failed: {exc}"


# ── Core HTTP call ────────────────────────────────────────────────────────────

def _chat(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    num_predict: int | None = None,
) -> str:
    """
    POST to Ollama /api/chat and return the assistant message content.
    Includes 1 retry for first-call model warmup latency.
    """
    options: dict[str, Any] = {"temperature": temperature}
    options["num_predict"] = num_predict if num_predict is not None else 1024

    last_exception: Exception | None = None
    for attempt in range(2):
        if attempt == 1:
            # On retry, boost token budget and instruct model to skip lengthy thinking
            options["num_predict"] = max(options["num_predict"] * 2, 2048)
            options["think"] = False

        payload = {
            "model": _ACTIVE_MODEL_TAG,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message", {})
            content = msg.get("content", "").strip()

            if not content and msg.get("thinking"):
                thinking_text = msg.get("thinking", "").strip()
                # Remove <think> ... </think> blocks if present
                clean_text = re.sub(r"<think>.*?</think>", "", thinking_text, flags=re.DOTALL).strip()
                if clean_text:
                    # Take the last non-empty paragraph (where models write their final answer)
                    paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
                    if paragraphs:
                        logger.info("Extracted final question text from model thinking output.")
                        return paragraphs[-1]

                # Fallback: if thinking_text has lines, return the last non-empty line
                lines = [l.strip() for l in thinking_text.split("\n") if l.strip() and not l.strip().startswith("<")]
                if lines:
                    logger.info("Extracted last line from model thinking block.")
                    return lines[-1]

                logger.warning("Attempt %d/2: Gemma 4 output empty content during thinking.", attempt + 1)
                raise RuntimeError("Empty response content (token budget exhausted during thinking)")

            if content:
                return content

            raise RuntimeError("Empty response content from Ollama")

        except requests.exceptions.ConnectionError as exc:
            logger.exception("Ollama connection error (attempt %d/2): %s", attempt + 1, exc)
            last_exception = ConnectionError(
                f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Is it running?"
            )
        except requests.exceptions.HTTPError as exc:
            logger.exception("Ollama HTTP error (attempt %d/2): %s", attempt + 1, exc)
            last_exception = RuntimeError(f"Ollama API error: {exc}")
        except Exception as exc:
            logger.exception("Ollama request failed (attempt %d/2): %s", attempt + 1, exc)
            last_exception = exc

    if last_exception:
        raise last_exception
    raise RuntimeError("Ollama request failed after retry")


# ── Public API ────────────────────────────────────────────────────────────────

def assess_answer_quality_and_difficulty(
    history: list[dict],
    current_difficulty: str = "Medium",
) -> tuple[str, str, str]:
    """
    Assess candidate's latest response quality and return:
    (new_difficulty, quality_signal, reasoning)

    Quality signals: "strong", "adequate", "weak"
    Difficulty levels: "Easy", "Medium", "Hard"
    Constrained shift: max 1 step up or down per turn.
    """
    if not history:
        return "Medium", "adequate", "Opening turn baseline."

    last_candidate_turn = None
    for turn in reversed(history):
        if turn.get("speaker") == "candidate":
            last_candidate_turn = turn.get("content", "").strip()
            break

    if not last_candidate_turn or last_candidate_turn == "[skipped]":
        curr = (current_difficulty or "Medium").capitalize()
        new_diff = "Easy" if curr == "Medium" else "Easy" if curr == "Easy" else "Medium"
        return new_diff, "weak", "Candidate skipped or provided empty response."

    prompt = (
        "Evaluate the candidate's last answer quality in one word: 'strong', 'adequate', or 'weak'.\n"
        f"Last Candidate Answer: \"{last_candidate_turn[:500]}\"\n\n"
        "Return ONLY a single JSON object with keys:\n"
        '{"quality": "strong" | "adequate" | "weak", "reasoning": "one sentence justification"}'
    )

    try:
        messages = [
            {"role": "system", "content": "You are a senior interview evaluator. Output strict JSON only."},
            {"role": "user", "content": prompt},
        ]
        raw_res = _chat(messages, temperature=0.2, num_predict=150).strip()
        match = re.search(r"\{.*\}", raw_res, re.DOTALL)
        json_str = match.group(0) if match else raw_res
        parsed = json.loads(json_str)
        quality = str(parsed.get("quality", "adequate")).lower().strip()
        reasoning = str(parsed.get("reasoning", "Assessed response quality.")).strip()
        if quality not in ("strong", "adequate", "weak"):
            quality = "adequate"
    except Exception as exc:
        logger.warning("Answer quality assessment LLM call failed (%s); using default transition", exc)
        quality = "adequate"
        reasoning = "Default transition (LLM call unparseable)."

    curr = (current_difficulty or "Medium").capitalize()
    if quality == "strong":
        new_diff = "Hard" if curr == "Medium" else "Hard" if curr == "Hard" else "Medium"
    elif quality == "weak":
        new_diff = "Easy" if curr == "Medium" else "Easy" if curr == "Easy" else "Medium"
    else:
        new_diff = curr

    return new_diff, quality, reasoning


def get_next_question(
    history: list[dict],
    role: str,
    resume_context: str | None = None,
    jd_context: str | None = None,
    time_allotted_seconds: int = 90,
    current_turn: int = 1,
    total_turns: int = 5,
    difficulty: str = "Medium",
) -> tuple[str, str | None]:
    """
    Generate the next interview question given the conversation history, time budget, and difficulty.
    """
    transcript = format_history_for_prompt(history)
    system_prompt = build_system_prompt(
        role,
        resume_context=resume_context,
        jd_context=jd_context,
        time_allotted_seconds=time_allotted_seconds,
        current_turn=current_turn,
        total_turns=total_turns,
        conversation_history=transcript if history else "None yet — this is the opening question.",
        difficulty=difficulty,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Here is the interview conversation so far:\n\n{transcript}\n\n"
                f"Ask turn {current_turn} of {total_turns} question."
                if history
                else f"Start the interview. Ask turn 1 of {total_turns} question."
            ),
        },
    ]
    # Set generous token cap (1024 tokens) to allow reasoning + question output
    raw_res = _chat(messages, temperature=0.7, num_predict=1024).strip()
    return parse_question_and_tool_call(raw_res)


def score_session(
    history: list[dict],
    session_id: str,
    role: str,
) -> dict[str, Any]:
    """
    Score the full session transcript using the 5-dimension rubric.

    Makes one strict-JSON call; if parsing fails, retries once with a stronger
    instruction (system-design.md §3). If the retry also fails, returns the
    fallback template scorecard.

    Parameters
    ----------
    history : list[dict]
        Full turn list for the session.
    session_id : str
        Used to populate the ``session_id`` field in the scorecard.
    role : str
        Interview role.

    Returns
    -------
    dict
        Parsed scorecard dict matching system-design.md §1.6, or a fallback
        template if scoring failed.
    """
    import json  # noqa: PLC0415

    system_prompt = build_scoring_prompt(session_id, role)
    transcript = format_history_for_prompt(history)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Full interview transcript:\n\n{transcript}\n\n"
                "Return the evaluation JSON now."
            ),
        },
    ]

    raw = _chat(messages, temperature=0.2, num_predict=1536)  # low temp, generous token cap for JSON output

    try:
        return parse_score_json(raw)
    except ParseError:
        logger.warning("First scoring attempt returned invalid JSON — retrying with stricter instruction.")

    # Retry once with a stricter reminder
    messages.append({"role": "assistant", "content": raw})
    messages.append({
        "role": "user",
        "content": (
            "Your response was not valid JSON. "
            "Return ONLY the raw JSON object — no code fences, no text before or after, "
            "just the JSON object starting with { and ending with }."
        ),
    })

    raw2 = _chat(messages, temperature=0.1)

    fallback = build_fallback_scorecard(session_id)
    try:
        return parse_score_json(raw2)
    except ParseError:
        logger.error("Scoring failed after retry. Returning fallback scorecard.")
        return fallback


def ask_ai_coach(
    history: list[dict],
    scorecard: dict,
    coach_messages: list[dict],
    user_query: str,
) -> str:
    """
    Generate a grounded AI Coach response for post-interview feedback.
    """
    if not user_query or not user_query.strip():
        return "Please enter a question about your interview performance."

    system_prompt = build_coach_chat_prompt(history, scorecard)
    messages = [{"role": "system", "content": system_prompt}]

    for msg in coach_messages:
        speaker = msg.get("role") or msg.get("speaker")
        content = msg.get("content", "")
        if speaker and content:
            role_name = "user" if speaker in ("user", "candidate") else "assistant"
            messages.append({"role": role_name, "content": content})

    messages.append({"role": "user", "content": user_query.strip()})

    try:
        res = _chat(messages, temperature=0.7, num_predict=1024).strip()
        return res if res else "I am unable to analyze that question right now. Please try asking again."
    except Exception as exc:
        logger.warning("ask_ai_coach call failed: %s", exc)
        return f"⚠️ AI Coach is currently unavailable ({exc}). Please check if Ollama is running."


def generate_model_answers(
    history: list[dict],
    role: str,
    resume_context: str | None = None,
    jd_context: str | None = None,
) -> list[dict]:
    """
    Generate 3-4 bullet point model answers for each interviewer question in history.
    Returns list of dicts: [{'turn_index': 1, 'bullets': [...]}, ...]
    """
    if not history:
        return []

    interviewer_turns = [t for t in history if t.get("speaker") == "interviewer"]
    if not interviewer_turns:
        return []

    system_prompt = build_model_answers_prompt(history, role, resume_context, jd_context)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Generate 3-4 bullet point model answers for each turn now."},
    ]

    try:
        raw_res = _chat(messages, temperature=0.3, num_predict=1024).strip()
        match = re.search(r"\{.*\}", raw_res, re.DOTALL)
        json_str = match.group(0) if match else raw_res
        parsed = json.loads(json_str)
        answers = parsed.get("model_answers", [])
        if isinstance(answers, list) and len(answers) > 0:
            return answers
    except Exception as exc:
        logger.warning("generate_model_answers call failed (%s); building fallback outlines", exc)

    fallbacks = []
    for idx, turn in enumerate(interviewer_turns, start=1):
        q_text = turn.get("content", "Question")
        fallbacks.append({
            "turn_index": idx,
            "question": q_text,
            "bullets": [
                f"Define core terms and foundational concepts relevant to: {q_text[:60]}...",
                "Discuss technical implementation, system trade-offs, and scalability/edge cases.",
                "Provide a concrete real-world example from past experience demonstrating impact.",
            ],
        })
    return fallbacks


def generate_resume_improvements(
    resume_text: str | None,
    ats_info: dict | None,
    scorecard: dict | None,
) -> list[str]:
    """
    Synthesize ATS missing keywords + interview scorecard into 3-5 concrete resume rewrite suggestions.
    Omitted cleanly if no resume_text provided.
    """
    if not resume_text or not resume_text.strip():
        return []

    system_prompt = build_resume_improvement_prompt(resume_text, ats_info or {}, scorecard or {})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Synthesize resume improvements now."},
    ]

    try:
        raw_res = _chat(messages, temperature=0.3, num_predict=768).strip()
        match = re.search(r"\{.*\}", raw_res, re.DOTALL)
        json_str = match.group(0) if match else raw_res
        parsed = json.loads(json_str)
        suggestions = parsed.get("suggestions", [])
        if isinstance(suggestions, list) and len(suggestions) > 0:
            return [str(s).strip() for s in suggestions if s]
    except Exception as exc:
        logger.warning("generate_resume_improvements call failed (%s); using baseline synthesis", exc)

    missing = (ats_info or {}).get("missing", [])
    fallbacks = []
    if missing:
        missing_str = ", ".join(missing[:4])
        fallbacks.append(f"Incorporate missing target keywords ({missing_str}) into technical bullet points.")
    fallbacks.append("Add quantifiable impact metrics (e.g., latency reduced by X%, throughput increased) to project bullets.")
    fallbacks.append("Explicitly highlight system architecture trade-offs and error handling strategies in technical role sections.")
    return fallbacks
