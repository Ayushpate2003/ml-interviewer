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
        "reliability engineering, microservices, and debugging war stories."
    ),
    "Frontend Engineer": (
        "Focus areas: web performance optimization, DOM manipulation, state management (Redux/Zustand/Context), "
        "JavaScript/TypeScript concepts, CSS architecture, browser storage, accessibility (a11y), responsive design, "
        "component lifecycle, and security (XSS/CORS/CSRF)."
    ),
    "DevOps / SRE": (
        "Focus areas: CI/CD automation pipelines, containerization & orchestration (Docker/Kubernetes), "
        "infrastructure as code (Terraform/Ansible), monitoring/observability (Prometheus/Grafana), "
        "incident management, high availability, zero-downtime deployments, and Linux systems tuning."
    ),
    "Cloud Computing": (
        "Focus areas: cloud architecture (AWS/GCP/Azure), serverless functions, cloud security & IAM policies, "
        "virtual networking (VPC, subnets, load balancing), cost optimization, multi-region failover, "
        "disaster recovery, and cloud migration strategies."
    ),
    "HR Round": (
        "Focus areas: behavioural questions (STAR format), motivation, "
        "conflict resolution, teamwork, career goals, and culture-fit."
    ),
    "System Design": (
        "Focus areas: large-scale distributed systems, capacity estimation, "
        "trade-offs between consistency and availability (CAP theorem), data modelling, "
        "message queues, and architectural decision-making."
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

def build_system_prompt(
    role: str,
    resume_context: str | None = None,
    jd_context: str | None = None,
    time_allotted_seconds: int = 90,
    current_turn: int = 1,
    total_turns: int = 5,
    conversation_history: str = "None yet — this is the opening question.",
) -> str:
    """
    Return the system prompt for the Resume/JD-Aware, Time-Calibrated Interview Question Generator.
    """
    has_resume = bool(resume_context and resume_context.strip())
    has_jd = bool(jd_context and jd_context.strip())

    if has_resume and has_jd:
        context_instructions = (
            f"RESUME CONTEXT:\n{resume_context}\n\n"
            f"JOB DESCRIPTION (JD) CONTEXT:\n{jd_context}\n\n"
            "INTERVIEW PERSONALIZATION INSTRUCTION (INTERSECTION MODE):\n"
            "You have access to BOTH the candidate's resume and the target Job Description (JD).\n"
            "Your primary objective is to ask questions at the INTERSECTION of the candidate's experience and the JD requirements.\n"
            "Target specific areas where the candidate's resume skills directly map to (or notably differ from) key JD responsibilities or tech stack requirements."
        )
    elif has_jd:
        context_instructions = (
            f"JOB DESCRIPTION (JD) CONTEXT:\n{jd_context}\n\n"
            "INTERVIEW PERSONALIZATION INSTRUCTION (JD-GROUNDED MODE):\n"
            "You have access to the target Job Description (JD).\n"
            "Ground your questions directly in the specific requirements, tools, architecture, and responsibilities stated in the JD."
        )
    elif has_resume:
        context_instructions = (
            f"RESUME CONTEXT:\n{resume_context}\n\n"
            "INTERVIEW PERSONALIZATION INSTRUCTION (RESUME-GROUNDED MODE):\n"
            "Ground your questions directly in the candidate's projects, technical skills, and past work experience from their resume."
        )
    else:
        context_instructions = (
            "CONTEXT: No resume or Job Description provided.\n\n"
            "INTERVIEW PERSONALIZATION INSTRUCTION (GENERIC ROLE MODE):\n"
            f"Ask high-quality, realistic mock interview questions tailored to the {role} role."
        )

    role_domain = role or "Software Engineer"
    role_context_str = _ROLE_CONTEXT.get(role_domain, "")
    role_focus_block = f"\nROLE FOCUS AREAS & DOMAIN SPECIFICITY:\n{role_context_str}\n" if role_context_str else ""
    t_sec = int(time_allotted_seconds) if time_allotted_seconds else 90
    cur_turn = int(current_turn) if current_turn else 1
    tot_turns = int(total_turns) if total_turns else 5
    conv_history = conversation_history.strip() if conversation_history else "None yet — this is the opening question."

    prompt = f"""SYSTEM PROMPT — Resume & JD-Aware, Time-Calibrated Interview Question Generator

ROLE
You are a senior {role_domain} interviewer with 10+ years of experience
conducting structured mock interviews.{role_focus_block}
You calibrate every question to fit realistically within the time the
candidate has to answer it. You never ask a question that cannot be
reasonably answered in the time given.

────────────────────────────────────────────────────────────
STAGE 1 — ANALYZE CONTEXT & PERSONALIZATION
────────────────────────────────────────────────────────────
{context_instructions}

────────────────────────────────────────────────────────────
STAGE 2 — GENERATE ONE TIME-CALIBRATED QUESTION
────────────────────────────────────────────────────────────
CONVERSATION SO FAR
{conv_history}
Turn {cur_turn} of {tot_turns}.

TIME BUDGET FOR THIS ANSWER
The candidate has {t_sec} seconds to answer whatever
question you ask next. This is a hard constraint on question design:

- Short budget (≤60s): ask something narrow and concrete.
- Medium budget (60–120s): one well-scoped question with room for a structured answer.
- Longer budget (>120s): ask a broader question inviting depth, but still ONE single question.

QUESTION RULES
1. Personalize using Stage 1 analysis (Resume ∩ JD intersection when both are available, JD requirements when JD-only, Resume content when Resume-only).
2. Prefer following up on the candidate's last answer over introducing a new topic when the last answer left something worth probing.
3. Do not repeat a question already asked in {conv_history}.
4. Match tone and content to {role_domain}.

────────────────────────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────────────────────────
Return exactly one question, written as it should be spoken aloud to
the candidate — no preamble, no commentary, no markdown code blocks.
End with a single short spoken cue on its own line:

  [TIME: You have {t_sec} seconds to answer.]

Nothing else follows that line."""

    return prompt


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
