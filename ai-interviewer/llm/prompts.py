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

# Per-question JSON schema for detailed, per-question feedback generation
PER_QUESTION_JSON_SCHEMA = {
    "questions": [
        {
            "question_number": "<int>",
            "question": "<exact interviewer question text>",
            "candidate_answer_summary": "<1-2 sentence summary of candidate's actual answer>",
            "overall_score": "<float 1-10, average of 5 dimension scores scaled to 10>",
            "difficulty_level": "Easy | Medium | Hard",
            "time_taken_hint": "<estimated e.g. ~30s, ~60s, ~90s>",
            "dimensions": {
                "technical_depth": {
                    "score": "<int 1-10>",
                    "did_well": "<what candidate did well technically>",
                    "missing_concepts": "<key technical concepts omitted>",
                    "knowledge_gaps": "<specific knowledge gaps identified>",
                    "recommendation": "<specific actionable recommendation>"
                },
                "communication_clarity": {
                    "score": "<int 1-10>",
                    "clarity": "<assessment of clarity>",
                    "structure": "<assessment of structure>",
                    "grammar_vocabulary": "<assessment of language use>",
                    "suggestion": "<how to communicate this answer better>"
                },
                "confidence_fluency": {
                    "score": "<int 1-10>",
                    "confidence": "<assessment of speaking confidence>",
                    "fluency_pacing": "<assessment of fluency>",
                    "filler_words": "<observed filler words or hesitations>",
                    "recommendation": "<how to improve confidence>"
                },
                "star_completeness": {
                    "score": "<int 1-10>",
                    "situation": "<present/absent/partial + brief note>",
                    "task": "<present/absent/partial + brief note>",
                    "action": "<present/absent/partial + brief note>",
                    "result": "<present/absent/partial + brief note>",
                    "missing_components": "<list of missing STAR parts>",
                    "suggestion": "<how to restructure using STAR>"
                },
                "problem_solving": {
                    "score": "<int 1-10>",
                    "logical_thinking": "<assessment of logic>",
                    "decision_making": "<assessment of decision quality>",
                    "solution_quality": "<assessment of solution>",
                    "alternative_approaches": "<approaches candidate missed>"
                }
            },
            "strengths": ["<strength 1>", "<strength 2>"],
            "weaknesses": ["<weakness 1>", "<weakness 2>"],
            "interviewer_expected": "<what a top candidate would have covered>",
            "how_to_improve": "<practical, actionable advice for next time>",
            "strong_answer_example": "<concise example of a strong answer to this specific question>",
            "practice_tips": ["<tip 1>", "<tip 2>"],
            "readiness_level": "Beginner | Intermediate | Advanced",
            "priority_for_improvement": "Low | Medium | High",
            "estimated_impact": "<e.g. 'Improving this area could add ~1.5 points to overall score'>"
        }
    ]
}


# ── Public helpers ─────────────────────────────────────────────────────────────

def build_system_prompt(
    role: str,
    resume_context: str | None = None,
    jd_context: str | None = None,
    time_allotted_seconds: int = 90,
    current_turn: int = 1,
    total_turns: int = 5,
    conversation_history: str = "None yet — this is the opening question.",
    difficulty: str = "Medium",
) -> str:
    """
    Return the system prompt for the Resume/JD-Aware, Adaptive Difficulty, Time-Calibrated Interview Question Generator.
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

    diff_upper = (difficulty or "Medium").capitalize()
    if diff_upper == "Hard":
        difficulty_instructions = (
            "ADAPTIVE QUESTION DIFFICULTY: HARD (Escalated Depth & Tradeoff Probing)\n"
            "The candidate has demonstrated strong performance in previous turns. Escalate question difficulty:\n"
            "- Ask about complex failure modes, high-scale bottlenecks, architectural trade-offs, or advanced edge cases.\n"
            "- Challenge the candidate to justify design choices with technical rigor."
        )
    elif diff_upper == "Easy":
        difficulty_instructions = (
            "ADAPTIVE QUESTION DIFFICULTY: EASY (Core Fundamentals & Calibrated Focus)\n"
            "The candidate gave a vague or hesitant response on previous turns. Calibrate difficulty to core fundamentals:\n"
            "- Ask a clear, direct, foundational question focusing on core concepts rather than advanced edge cases.\n"
            "- Help the candidate establish a solid baseline without overloading them."
        )
    else:
        difficulty_instructions = (
            "ADAPTIVE QUESTION DIFFICULTY: MEDIUM (Standard Mock Interview Depth)\n"
            "Maintain standard mock interview depth: ask a well-scoped technical or behavioral question matching expected industry standards."
        )

    role_domain = role or "Software Engineer"
    role_context_str = _ROLE_CONTEXT.get(role_domain, "")
    role_focus_block = f"\nROLE FOCUS AREAS & DOMAIN SPECIFICITY:\n{role_context_str}\n" if role_context_str else ""
    t_sec = int(time_allotted_seconds) if time_allotted_seconds else 90
    cur_turn = int(current_turn) if current_turn else 1
    tot_turns = int(total_turns) if total_turns else 5
    conv_history = conversation_history.strip() if conversation_history else "None yet — this is the opening question."

    prompt = f"""SYSTEM PROMPT — Resume & JD-Aware, Adaptive Difficulty, Time-Calibrated Interview Question Generator

ROLE
You are a senior {role_domain} interviewer with 10+ years of experience
conducting structured mock interviews.{role_focus_block}
You calibrate every question to fit realistically within the time the
candidate has to answer it. You never ask a question that cannot be
reasonably answered in the time given.

────────────────────────────────────────────────────────────
STAGE 1A — ANALYZE CONTEXT & PERSONALIZATION
────────────────────────────────────────────────────────────
{context_instructions}

────────────────────────────────────────────────────────────
STAGE 1B — ADAPTIVE DIFFICULTY CALIBRATION
────────────────────────────────────────────────────────────
{difficulty_instructions}

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
1. Personalize using Stage 1A analysis and calibrate difficulty using Stage 1B instructions.
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


def build_coach_chat_prompt(history: list[dict], scorecard: dict) -> str:
    """
    Return system prompt for post-interview AI Coach chat grounded in transcript & scorecard.
    """
    import json  # noqa: PLC0415
    transcript = format_history_for_prompt(history)
    score_summary = json.dumps(scorecard, indent=2) if scorecard else "Scorecard unavailable."

    return f"""SYSTEM PROMPT — AI Interview Feedback Coach

ROLE
You are an empathetic, expert AI Interview Coach. The candidate has just completed a mock interview session. Your job is to answer the candidate's follow-up questions about their performance, scores, and how to improve.

SESSION TRANSCRIPT
{transcript}

RUBRIC SCORECARD & JUSTIFICATIONS
{score_summary}

RULES
1. Ground your answers STRICTLY in the transcript and scorecard provided above. Reference specific quotes, turns, and dimension scores.
2. If the user asks for advice on how to improve an answer, give actionable, concrete examples tailored to what they actually said.
3. SCOPE CONSTRAINT: If the user asks a question completely unrelated to this interview session (e.g. general trivia, coding tasks unrelated to the interview, 'what is the capital of France?'), POLITELY REDIRECT them back to discussing their interview feedback. Do not answer off-topic queries."""


def build_model_answers_prompt(
    history: list[dict],
    role: str,
    resume_context: str | None = None,
    jd_context: str | None = None,
) -> str:
    """
    Return system prompt for generating 3-4 bullet point model answers for each interviewer question.
    """
    transcript = format_history_for_prompt(history)
    ctx = ""
    if resume_context and resume_context.strip():
        ctx += f"\nResume Context: {resume_context[:400]}"
    if jd_context and jd_context.strip():
        ctx += f"\nJD Context: {jd_context[:400]}"

    return f"""SYSTEM PROMPT — Model Answer Outline Generator

ROLE
You are a principal technical interviewer for {role} roles.{ctx}

TRANSCRIPT
{transcript}

TASK
For each question asked by the interviewer in the transcript above, generate 3-4 bullet points summarizing what a comprehensive, high-scoring model answer should include.

OUTPUT FORMAT
Return ONLY valid JSON matching this schema:
{{
  "model_answers": [
    {{
      "turn_index": 1,
      "bullets": [
        "Core technical foundation / direct definition",
        "Key trade-offs, architecture, or edge cases",
        "Practical implementation detail or real-world example"
      ]
    }}
  ]
}}"""


def build_resume_improvement_prompt(
    resume_text: str,
    ats_info: dict,
    scorecard: dict,
) -> str:
    """
    Return system prompt for synthesizing ATS missing keywords + interview scorecard into resume rewrite suggestions.
    """
    import json  # noqa: PLC0415
    missing = ", ".join((ats_info or {}).get("missing", [])) or "None identified."
    score_summary = json.dumps(scorecard, indent=2) if scorecard else "None"

    return f"""SYSTEM PROMPT — Resume Rewrite & Improvement Synthesizer

ROLE
You are an executive resume coach and senior technical interviewer.

RESUME HIGHLIGHTS
{resume_text[:1000]}

ATS MISSING KEYWORDS
{missing}

INTERVIEW SCORECARD & FEEDBACK
{score_summary}

TASK
Synthesize the candidate's ATS keyword gaps with their actual mock interview performance into 3-5 concrete, high-impact resume bullet point rewrite suggestions.
For each suggestion, provide a revised bullet point or section addition that explicitly incorporates target skills they demonstrated or missed during the interview.

OUTPUT FORMAT
Return ONLY a valid JSON object matching this schema:
{{
  "suggestions": [
    "Specific resume rewrite suggestion 1...",
    "Specific resume rewrite suggestion 2...",
    "Specific resume rewrite suggestion 3..."
  ]
}}"""


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


def build_per_question_feedback_prompt(
    history: list[dict],
    role: str,
    resume_context: str | None = None,
    jd_context: str | None = None,
) -> str:
    """
    Return the system prompt for generating per-question detailed feedback.

    The user message for this call should be a simple trigger like
    "Generate per-question feedback now."

    The LLM is instructed to score each interviewer question individually
    across all 5 rubric dimensions (scored 1-10) and provide rich coaching
    metadata: STAR breakdown, strengths, weaknesses, example strong answer,
    practice tips, readiness level, and improvement priority.
    """
    import json  # noqa: PLC0415

    transcript = format_history_for_prompt(history)
    schema_str = json.dumps(PER_QUESTION_JSON_SCHEMA, indent=2)

    ctx = ""
    if resume_context and resume_context.strip():
        ctx += f"\nResume Context (first 400 chars): {resume_context[:400]}"
    if jd_context and jd_context.strip():
        ctx += f"\nJob Description Context (first 400 chars): {jd_context[:400]}"

    return f"""SYSTEM PROMPT — Per-Question Detailed Interview Feedback Generator

ROLE
You are an experienced Technical Interviewer, Hiring Manager, and Career Coach for {role} roles.{ctx}

Your task is to analyze each interview question individually from the transcript below and provide
detailed, actionable, coaching-focused feedback.

EVALUATION RUBRIC (score each dimension 1-10 per question):
  • technical_depth       — correctness, depth, domain accuracy, key concepts covered
  • communication_clarity — structure, conciseness, clarity of explanation
  • confidence_fluency    — delivery, decisiveness, fluency, filler word avoidance
  • star_completeness     — presence of Situation / Task / Action / Result structure
  • problem_solving       — logical reasoning, edge case handling, solution quality

TRANSCRIPT
{transcript}

INSTRUCTIONS
1. Analyze EVERY question asked by the Interviewer in the transcript above.
2. For each question, evaluate the immediately following Candidate response.
3. If the candidate skipped a question (answer is '[skipped]' or empty), assign score 1 to all dimensions and note the skip in candidate_answer_summary.
4. overall_score = average of the 5 dimension scores (already on 1-10 scale).
5. difficulty_level: infer from the question complexity (Easy/Medium/Hard).
6. time_taken_hint: estimate from answer length (e.g., '~30s' for very short, '~90s' for detailed).
7. strong_answer_example: write a concise 3-5 sentence example of a high-scoring answer to this specific question.
8. Tailor all recommendations specifically to what the candidate actually said — avoid generic advice.

OUTPUT FORMAT
Return ONLY valid JSON matching this schema exactly — no prose, no code fences, no explanation:

{schema_str}"""

