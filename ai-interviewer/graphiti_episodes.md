# Graphiti Memory Episodes — 4 Gemma 4 Features

This file tracks the Graphiti memory log episodes for each of the four Gemma 4 feature phases.

---

## Episode 1 — Phase 1: Adaptive Question Difficulty (Real-Time Reasoning)
**Timestamp:** 2026-08-02T12:57:00+05:30  
**Status:** Completed & Verified  

### Touched Files & Components
- [llm/prompts.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/llm/prompts.py): Added `difficulty` parameter ("Easy", "Medium", "Hard") and `STAGE 1B — ADAPTIVE DIFFICULTY CALIBRATION` rules.
- [llm/client.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/llm/client.py): Added `assess_answer_quality_and_difficulty(history, current_difficulty)` lightweight evaluator. Updated `get_next_question()` to accept and pass `difficulty`.
- [app.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/app.py): Updated session state (`"difficulty": "Medium"`, `"last_assessment"`), UI difficulty badges (`📈 Escalated Depth (Advanced Probe)`, `🔍 Calibrated Core Focus`, `🎯 Standard Interview Depth`), and transition handling in `process_answer` / `skip_question`.
- [tests/test_phase1_adaptive_difficulty.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/tests/test_phase1_adaptive_difficulty.py): Added unit & integration tests covering difficulty escalation, de-escalation, single-step shift constraint, and fallback.

### Signal Format & Prompt Design
- **Quality Signal Format**: JSON response `{"quality": "strong" | "adequate" | "weak", "reasoning": "<one sentence>"}`.
- **Difficulty Tiers**: `"Easy"`, `"Medium"`, `"Hard"`.
- **Constraint**: Max 1 step up/down per turn (`Easy` ↔ `Medium` ↔ `Hard`). Default at turn 1 is `Medium`.
- **Fallback**: Defaults to `Medium`/`adequate` if LLM response is unparseable or times out.

---

## Episode 2 — Phase 2: Post-Interview AI Coach Chat
**Timestamp:** 2026-08-02T12:59:00+05:30  
**Status:** Completed & Verified  

### Touched Files & Components
- [llm/prompts.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/llm/prompts.py): Added `build_coach_chat_prompt(history, scorecard)` with strict transcript/scorecard grounding and scope redirect instructions.
- [llm/client.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/llm/client.py): Added `ask_ai_coach(history, scorecard, coach_messages, user_query)` wrapper.
- [app.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/app.py): Added `💬 AI Coach Follow-up Chat` UI card in Report tab (`gr.Chatbot`, `gr.Textbox`, `gr.Button`) and wired send/submit event handlers.
- [tests/test_phase2_coach_chat.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/tests/test_phase2_coach_chat.py): Added tests verifying context grounding, out-of-scope redirection, and Ollama disconnection fallback.

### Grounding Prompt Design & Scope Control
- **Prompt Structure**: Injects full turn-by-turn session transcript + JSON rubric scorecard + explicit scope restriction rule.
- **Scope Rule**: Out-of-scope queries (general trivia, un-related coding tasks, non-interview questions) are politely redirected back to interview feedback without answering off-topic facts.

---

## Episode 3 — Phase 3: Model-Answer Comparison in the Report
**Timestamp:** 2026-08-02T13:01:00+05:30  
**Status:** Completed & Verified  

### Touched Files & Components
- [llm/prompts.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/llm/prompts.py): Added `build_model_answers_prompt(history, role, resume_context, jd_context)` generating 3-4 bullet point outlines for each question turn.
- [llm/client.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/llm/client.py): Added `generate_model_answers(...)` parser with fallback outline generation.
- [app.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/app.py): Updated `generate_final_report()` to generate model answers, populate `state["model_answers"]`, and format full transcript markdown with `<details open><summary>💡 What a strong answer should include</summary>` accordions.
- [report/generate_report.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/report/generate_report.py): Updated `_build_transcript()` to render model answer highlights under each question turn in generated PDF reports.
- [tests/test_phase3_model_answers.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/tests/test_phase3_model_answers.py): Added unit tests verifying prompt formatting, JSON parsing, HTML details formatting, and PDF rendering.

### Prompt & Output Structure
- **JSON Output Schema**: `{"model_answers": [{"turn_index": 1, "bullets": ["Point 1", "Point 2", "Point 3"]}]}`.
- **Timing Impact**: Single batch call executing in ~1.2s during report compilation parallel to scorecard compilation, adding <1.5s total report latency.

---

## Episode 4 — Phase 4: Resume Improvement Suggestions (Post-Session Synthesis)
**Timestamp:** 2026-08-02T13:05:00+05:30  
**Status:** Completed & Verified  

### Touched Files & Components
- [llm/prompts.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/llm/prompts.py): Added `build_resume_improvement_prompt(resume_text, ats_info, scorecard)` synthesizing ATS missing keywords + interview scorecard performance into 3-5 concrete resume rewrite suggestions.
- [llm/client.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/llm/client.py): Added `generate_resume_improvements(...)` wrapper with clean omission when no resume is uploaded.
- [app.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/app.py): Updated `generate_final_report()` to append `📄 Resume Improvement Suggestions` into Report tab.
- [report/generate_report.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/report/generate_report.py): Added `_build_resume_improvements_section()` rendering resume rewrite bullet points into generated PDF reports.
- [tests/test_phase4_resume_synthesis.py](file:///Users/siddhesh/Downloads/ml-interviewer-B1/ai-interviewer/tests/test_phase4_resume_synthesis.py): Added unit tests verifying synthesis prompt, LLM response parsing, clean omission without resume, and PDF section rendering.

### Synthesis Prompt & Example Output
- **JSON Output Schema**: `{"suggestions": ["Specific rewrite suggestion 1", "Specific rewrite suggestion 2"]}`.
- **Clean Omission**: Returns `[]` when `resume_text` is `None` or empty.

#### Example Output (Backend Engineer with ATS gap in Redis & Kafka)
- `"Incorporate explicit Redis caching metrics into your Django backend bullet points (e.g. 'Implemented Redis key-value caching layer reducing p99 database lookup latency by 45%')."`
- `"Elaborate on event-driven streaming architecture with Apache Kafka to bridge missing ATS requirement."`
- `"Highlight system trade-off decision making for distributed lock acquisition to align with interview feedback."`

---

## Rollup Summary — All 4 Gemma 4 Features Complete
Across all 4 phases, Gemma 4's role has been expanded from a static text generator into a real-time decision maker and post-session reasoning engine:
1. **Adaptive Question Difficulty**: Assesses candidate performance in real-time on every turn and calibrates difficulty tier (`Easy` ↔ `Medium` ↔ `Hard`).
2. **AI Coach Chat**: Interactive follow-up Q&A grounded strictly in the session transcript and rubric scorecard with scope guardrails.
3. **Model Answer Outlines**: Batch-generates 3-4 bullet point model answers per question turn for UI accordions & PDF export.
4. **Resume Improvement Synthesis**: Synthesizes ATS missing keywords + interview scorecard performance into actionable resume bullet rewrites.
