"""
llm/parser.py
-------------
Defensive JSON extraction from Gemma 4 / Ollama model output.

This is the P0 risk module (unittest.md §2): the model may return the JSON
wrapped in prose, code fences, or entirely fail to produce valid JSON.

Handles (in order of attempt):
  1. Clean JSON string.
  2. JSON wrapped in ```json … ``` or ``` … ``` code fences.
  3. JSON preceded by prose ("Here is the evaluation:\n{...}").
  4. Completely unparseable → returns ``fallback`` if provided, else raises.

Retry-once behaviour (system-design.md §3):
  The retry is orchestrated by ``llm/client.py``; ``parser.py`` is stateless
  and only responsible for extraction + fallback, not for re-calling the model.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Raised when JSON cannot be extracted and no fallback is given."""


def parse_score_json(
    raw: str,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Extract and parse the first valid JSON object from ``raw``.

    Parameters
    ----------
    raw : str
        Raw text output from the model.
    fallback : dict | None
        If provided and the raw text cannot be parsed, return this value
        instead of raising. Useful for graceful degradation in the report step.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ParseError
        If ``raw`` cannot be parsed and ``fallback`` is ``None``.
    """
    if not raw or not raw.strip():
        return _fallback_or_raise(raw, fallback, reason="empty model output")

    # Strategy 1 — try the raw string directly
    candidate = _try_parse(raw.strip())
    if candidate is not None:
        return candidate

    # Strategy 2 — strip code fences (```json ... ``` or ``` ... ```)
    stripped = _strip_code_fence(raw)
    if stripped:
        candidate = _try_parse(stripped)
        if candidate is not None:
            return candidate

    # Strategy 3 — find the first {...} block anywhere in the string
    candidate = _extract_first_json_object(raw)
    if candidate is not None:
        return candidate

    return _fallback_or_raise(raw, fallback, reason="no valid JSON found in model output")


# ── Private helpers ────────────────────────────────────────────────────────────

def _try_parse(text: str) -> dict[str, Any] | None:
    """Return parsed dict if ``text`` is valid JSON, else None."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _strip_code_fence(raw: str) -> str | None:
    """
    Remove ```json / ``` fences and return the inner text, or None.
    Handles both ```json and plain ``` delimiters.
    """
    # Match triple-backtick blocks, optional 'json' language hint
    pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_first_json_object(raw: str) -> dict[str, Any] | None:
    """
    Scan ``raw`` for the first ``{`` and attempt to parse from there,
    progressively shortening the candidate string if nested braces are involved.
    """
    start = raw.find("{")
    if start == -1:
        return None

    # Walk backwards from end to find matching closing brace
    for end in range(len(raw), start, -1):
        candidate = raw[start:end]
        result = _try_parse(candidate)
        if result is not None:
            return result
    return None


def _fallback_or_raise(
    raw: str,
    fallback: dict[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    if fallback is not None:
        logger.warning("parse_score_json: %s — using fallback. Raw (first 200 chars): %s", reason, raw[:200])
        return fallback
    raise ParseError(f"parse_score_json: {reason}. Raw output (first 200 chars): {raw[:200]}")


# ── Fallback template builder ─────────────────────────────────────────────────

def build_fallback_scorecard(session_id: str) -> dict[str, Any]:
    """
    Return a structurally-valid scorecard template used when Gemma 4's scoring
    call fails even after the retry (system-design.md §3).

    All scores are None to signal the report that scoring failed for these
    dimensions, rather than crashing report generation.
    """
    from llm.prompts import REQUIRED_DIMENSIONS  # noqa: PLC0415

    return {
        "session_id": session_id,
        "overall_score": None,
        "dimensions": [
            {
                "name": dim,
                "score": None,
                "justification": "Scoring unavailable — model did not return valid JSON.",
            }
            for dim in REQUIRED_DIMENSIONS
        ],
        "summary": (
            "Scoring could not be completed automatically. "
            "Please review the transcript manually."
        ),
    }
