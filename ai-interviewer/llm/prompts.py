"""
llm/prompts.py
--------------
System prompts, rubric definitions, and JSON schema for Gemma 4 calls.

Two distinct call modes (system-design.md §1.5):
  1. Per-turn call  → plain text out: next interview question.
  2. End-of-session → strict JSON out: scored evaluation.

Role-specific system prompts (mvp.md "Should Build"):
  - Backend Engineer
  - HR Round
  - System Design
"""

from __future__ import annotations

# ── Shared interviewer persona ─────────────────────────────────────────────────
_BASE_PERSONA = (
    "You are an expert technical interviewer conducting a mock job interview. "
    "Your tone is professional but encouraging. "
    "Ask ONE focused, non-repetitive follow-up question per turn. "
    "Never repeat a question that has already appeared in the conversation. "
    "Keep your question concise — ideally a single sentence, two at most."
)

# ── Role-specific context blocks ───────────────────────────────────────────────
_ROLE_CONTEXT: dict[str, str] = {
    "Backend Engineer": (
        "Focus areas: system design, APIs, databases, caching, concurrency, "
        "reliability engineering, and debugging war stories."
    ),
    "HR Round": (
        "Focus areas: behavioural questions (STAR format), motivation, "
        "conflict resolution, teamwork, career goals, and culture-fit."
    ),
    "System Design": (
        "Focus areas: large-scale distributed systems, capacity estimation, "
        "trade-offs between consistency and availability, data modelling, "
        "and architectural decision-making."
    ),
}

_RUBRIC_DESCRIPTION = (
    "Rubric dimensions (each scored 1–5 with a short justification):\n"
    "  • technical_depth      — correctness and depth of domain content.\n"
    "  • communication_clarity — structure, conciseness, and clarity.\n"
    "  • confidence_fluency   — coherence; penalise excessive filler words "
    "                            and heavy hedging (from transcript cues only).\n"
    "  • star_completeness    — for behavioural answers: Situation / Task / "
    "                            Action / Result all present.\n"
    "  • problem_solving      — reasoning process, not just the final answer.\n"
)

# ── Scoring JSON schema (exact contract from system-design.md §1.6) ────────────
SCORING_JSON_SCHEMA = {
    "session_id": "<string>",
    "overall_score": "<float, average of dimension scores>",
    "dimensions": [
        {
            "name": "technical_depth",
            "score": "<int 1-5>",
            "justification": "<1-2 sentences>",
        },
        {
            "name": "communication_clarity",
            "score": "<int 1-5>",
            "justification": "<1-2 sentences>",
        },
        {
            "name": "confidence_fluency",
            "score": "<int 1-5>",
            "justification": "<1-2 sentences>",
        },
        {
            "name": "star_completeness",
            "score": "<int 1-5>",
            "justification": "<1-2 sentences>",
        },
        {
            "name": "problem_solving",
            "score": "<int 1-5>",
            "justification": "<1-2 sentences>",
        },
    ],
    "summary": "<2-3 sentence overall summary with suggested next steps>",
}

# Required dimension names — used by parser.py for fallback template
REQUIRED_DIMENSIONS = [
    "technical_depth",
    "communication_clarity",
    "confidence_fluency",
    "star_completeness",
    "problem_solving",
]


# ── Public helpers ─────────────────────────────────────────────────────────────

def build_system_prompt(role: str) -> str:
    """
    Return the system prompt for the per-turn question-generation call.

    Parameters
    ----------
    role : str
        One of the keys in ``_ROLE_CONTEXT``, or any free-text role name.
        If the role is unrecognised, generic guidance is used.
    """
    role_ctx = _ROLE_CONTEXT.get(role, f"This is a {role} interview.")
    return (
        f"{_BASE_PERSONA}\n\n"
        f"Role context: {role_ctx}\n\n"
        "When generating a question, output ONLY the question text — "
        "no preamble, no 'Sure!', no meta-commentary."
    )


def build_scoring_prompt(session_id: str, role: str) -> str:
    """
    Return the system prompt for the end-of-session JSON scoring call.

    The user message for this call should be the full transcript.
    """
    import json  # noqa: PLC0415
    schema_str = json.dumps(SCORING_JSON_SCHEMA, indent=2)
    return (
        "You are an expert evaluator. "
        "Given the full mock-interview transcript below, score the candidate "
        f"on the rubric for a {role} interview.\n\n"
        f"{_RUBRIC_DESCRIPTION}\n"
        "Return ONLY valid JSON matching this schema exactly — no prose, "
        "no code fences, no explanation:\n\n"
        f"{schema_str}\n\n"
        f'Use session_id="{session_id}".'
    )


def format_history_for_prompt(history: list[dict]) -> str:
    """
    Convert the in-memory turn list to a readable transcript string for the prompt.

    Parameters
    ----------
    history : list[dict]
        Each dict has ``speaker`` (``'interviewer'``/``'candidate'``) and
        ``content`` keys.
    """
    lines = []
    for turn in history:
        speaker_label = "Interviewer" if turn["speaker"] == "interviewer" else "Candidate"
        lines.append(f"{speaker_label}: {turn['content']}")
    return "\n".join(lines)
