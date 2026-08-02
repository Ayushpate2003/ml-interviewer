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

def get_next_question(
    history: list[dict],
    role: str,
    resume_context: str | None = None,
    time_allotted_seconds: int = 90,
    current_turn: int = 1,
    total_turns: int = 5,
) -> tuple[str, str | None]:
    """
    Generate the next interview question given the conversation history and time budget.

    Parameters
    ----------
    history : list[dict]
        Running turn list; each dict has ``speaker`` and ``content`` keys.
    role : str
        Interview role (e.g. "Backend Engineer").
    resume_context : str | None
        Optional candidate background/resume highlights to tailor questions.
    time_allotted_seconds : int
        Time budget for candidate's response in seconds (e.g., 60, 90, 120).
    current_turn : int
        Current turn number (1-indexed).
    total_turns : int
        Total number of turns in the session.

    Returns
    -------
    tuple[str, str | None]
        (question_text, followup_topic_or_None)
    """
    transcript = format_history_for_prompt(history)
    system_prompt = build_system_prompt(
        role,
        resume_context=resume_context,
        time_allotted_seconds=time_allotted_seconds,
        current_turn=current_turn,
        total_turns=total_turns,
        conversation_history=transcript if history else "None yet — this is the opening question.",
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
        logger.error("Both scoring attempts failed. Using fallback scorecard template.")
        return fallback
