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
from typing import Any

import requests

from llm.parser import ParseError, build_fallback_scorecard, parse_score_json
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
    """
    options: dict[str, Any] = {"temperature": temperature}
    if num_predict is not None:
        options["num_predict"] = num_predict

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
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Is it running?"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama API error: {exc}") from exc

    data = resp.json()
    return data["message"]["content"]


# ── Public API ────────────────────────────────────────────────────────────────

def get_next_question(history: list[dict], role: str) -> str:
    """
    Generate the next interview question given the conversation history.

    Parameters
    ----------
    history : list[dict]
        Running turn list; each dict has ``speaker`` and ``content`` keys.
    role : str
        Interview role (e.g. "Backend Engineer").

    Returns
    -------
    str
        The next question text (plain, no prose preamble).
    """
    system_prompt = build_system_prompt(role)
    transcript = format_history_for_prompt(history)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Here is the interview conversation so far:\n\n{transcript}\n\n"
                "Ask the next focused question."
                if history
                else "Start the interview. Ask the first question."
            ),
        },
    ]
    # Cap response length to 80 tokens (~1-2 sentences) for high generation speed
    return _chat(messages, temperature=0.7, num_predict=80).strip()


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

    raw = _chat(messages, temperature=0.2)  # low temp for structured output

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
