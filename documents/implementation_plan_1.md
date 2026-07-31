# Implementation Plan — Privacy-First AI Interviewer (Gemma 4)

A fully offline, single-machine spoken mock-interview app. Gemma 4 (via Ollama) does question generation and structured scoring; faster-whisper handles STT; Piper handles TTS; SQLite persists sessions; ReportLab generates the PDF report.

---

## User Review Required

> [!IMPORTANT]
> **Graphiti / MCP memory** — The prompt references `add_memory`, `search_memory_nodes`, `search_memory_facts`, and `get_episodes` as Graphiti MCP tools. These are **not** currently listed in your active MCP server configuration. If you want per-phase memory snapshots written to a Graphiti server, you need to tell me the server name and confirm the MCP connection. Otherwise I will skip those calls and write equivalent phase-summary logs to `documents/build_log.md` instead — this keeps all build facts inside the repo. **Please confirm which you prefer before I start.**

> [!IMPORTANT]
> **Ollama + Gemma 4 pull** — Phase 1 requires `ollama pull gemma4:4b` which will download several GB. If Ollama is not installed yet, I will also install it. Make sure you have ~5 GB free disk space and are OK running the pull command.

> [!WARNING]
> **`piper-tts` pip package availability** — Piper's official Python wheel (`piper-tts`) may not be available for all macOS/Python combos. The fallback is to call the Piper CLI binary directly. I will detect this during Phase 1 and record the workaround.

---

## Open Questions

> [!IMPORTANT]
> 1. **Graphiti MCP** — use live Graphiti calls or repo-based `build_log.md`?
> 2. **Python version / virtual env** — should I create a fresh `venv` inside the project root, or use an existing environment?
> 3. **Piper voice** — do you have a preferred voice model (`en_US-lessac-medium` is the standard default), or should I auto-download the first available English voice?
> 4. **Demo hardware** — are you on Apple Silicon (M-series) or Intel Mac? This affects which faster-whisper backend (`int8` vs `float32`) and which Gemma 4 variant to default to.

---

## Proposed Changes

### Phase 0 — Bootstrap & Memory Init

#### [NEW] `ai-interviewer/` (whole folder tree)

Creates the canonical folder structure from `architecture.md §5` with empty stub files:

```
ai-interviewer/
├── app.py
├── requirements.txt
├── stt/
│   ├── __init__.py
│   └── transcribe.py
├── llm/
│   ├── __init__.py
│   ├── client.py
│   ├── prompts.py
│   └── parser.py
├── memory/
│   ├── __init__.py
│   ├── db.py
│   └── schema.sql
├── tts/
│   ├── __init__.py
│   └── speak.py
├── report/
│   ├── __init__.py
│   └── generate_report.py
├── data/
│   └── .gitkeep
└── tests/
    ├── conftest.py
    ├── test_parser.py
    ├── test_transcribe.py
    ├── test_db.py
    └── test_report.py
```

---

### Phase 1 — Environment Setup

#### [NEW] `requirements.txt`
Pins: `faster-whisper`, `gradio`, `reportlab`, `piper-tts` (or CLI fallback), `pytest`, `requests` (Ollama HTTP wrapper).

#### [MODIFY] `app.py`
Startup check: verify Ollama is reachable at `localhost:11434` and `gemma4:4b` is listed in `/api/tags`. Fail fast with a clear on-screen error box before the interview begins.

---

### Phase 2 — Audio Capture + STT

#### [MODIFY] `stt/transcribe.py`
- `TranscriptionError` exception class.
- `load_stt_model()` — loads `faster-whisper` `small` (or `base.en`) **once** at import/startup.
- `transcribe(audio_bytes) -> str` — writes to temp buffer, calls `model.transcribe(...)`, returns joined segment text; empty/silent → `""`; bad bytes → raises `TranscriptionError`.

#### [MODIFY] `tests/test_transcribe.py`
Three tests from `unittest.md §3.2`. Fixture audio: silent WAV + a short known sample WAV baked into `tests/fixtures/`.

---

### Phase 3 — Memory / DB

#### [MODIFY] `memory/schema.sql`
Exact DDL from `architecture.md §6`: `sessions`, `turns`, `scores` tables.

#### [MODIFY] `memory/db.py`
- `create_session(db, role) -> session_id`
- `add_turn(db, session_id, speaker, content)`
- `get_turns(db, session_id) -> list`
- `save_scores(db, session_id, scores_list)`
- `get_scores(db, session_id) -> list`

In-process: `get_turns` also reflects the in-memory list kept in `app.py` state.

#### [MODIFY] `tests/test_db.py`
Two tests from `unittest.md §3.3` using `pytest` `tmp_path` fixture.

---

### Phase 4 — Gemma 4 Reasoning Loop

#### [MODIFY] `llm/client.py`
Thin wrapper around `POST http://localhost:11434/api/chat`. Target model tag: `gemma4:4b`. Raises `ConnectionError` if Ollama unreachable.

#### [MODIFY] `llm/prompts.py`
- `SYSTEM_PROMPT_TEMPLATE` — interviewer persona + role context + rubric description (substitutes `{role}` at call time).
- `SCORING_JSON_SCHEMA` — the exact five-dimension schema from `system-design.md §1.6`.

#### [MODIFY] `llm/parser.py`
- `parse_score_json(raw, fallback=None) -> dict` — handles clean JSON, code-fenced JSON, prose-prefixed JSON, and unparseable output (returns `fallback` if provided, else raises).

#### [MODIFY] `tests/test_parser.py` and `tests/test_transcribe.py`
Four parser tests (`unittest.md §3.1`) + two Ollama-mocked client tests (`unittest.md §3.5`). Ollama is **never** called live in automated tests.

---

### Phase 5 — Scoring / Evaluation

#### [MODIFY] `llm/client.py`
Add `score_session(history, session_id) -> dict` — strict JSON output call.

#### [MODIFY] `llm/parser.py`
Add retry-once-on-malformed-JSON logic per `system-design.md §3`: sends a stricter `"return only JSON"` reminder prompt on first failure; if second attempt also fails, returns a template noting which dimension failed.

---

### Phase 6 — TTS (`tts/speak.py`)

#### [MODIFY] `tts/speak.py`
- `load_tts_model()` — downloads/caches one English Piper voice (default: `en_US-lessac-medium`).
- `speak(text)` — generates audio bytes, plays via Gradio's audio output (or OS playback fallback).
- Manual listening verification only; no automated test.

---

### Phase 7 — Report (`report/generate_report.py`)

#### [MODIFY] `report/generate_report.py`
ReportLab PDF with 4 sections: cover, scorecard table, full transcript, summary/next-steps. Gracefully handles `None` scores (per `unittest.md §3.4`).

#### [MODIFY] `tests/test_report.py`
Three tests from `unittest.md §3.4` using `tmp_path`. Includes a PDF text-extraction helper using PyPDF2 or pdfplumber (added to `requirements.txt`).

---

### Phase 8 — UI Wiring (`app.py`)

#### [MODIFY] `app.py`
Assembles the 3-screen Gradio UI from `userflow.md §2`:
1. **Setup** — role dropdown, mic test button, "🔒 100% Offline" badge.
2. **Live Interview** — question text + TTS, record/stop, live transcript, turn counter (`Q N of ~5`), Skip button.
3. **Report** — scorecard, expandable transcript, PDF download, new-session button.

Edge cases wired: silence recovery, skip → `[skipped]` placeholder, mic permission hint (`C1` branch).

---

### Phase 9 — Integration Test Pass

- Run full automated suite: `pytest tests/ -v`.
- Manual 5-turn offline checklist from `unittest.md §4`.
- Bug fixes recorded in phase episode / build log.

---

### Phase 10 — Polish & Demo Prep

#### [MODIFY] `llm/prompts.py`
Role-specific system prompt variants: Backend Engineer, HR Round, System Design.

#### [NEW] `README.md`
Setup instructions, demo script, Definition of Done confirmation.

Optional stretch (only if schedule allows):
- Gemma 4 native audio path for one demo turn (30 s cap caveat documented).
- Resume/JD keyword seeding (no RAG — just keyword extraction).

---

## Verification Plan

### Automated Tests
```bash
cd ai-interviewer
pytest tests/ -v --tb=short
```
Expected: all tests in `test_parser.py`, `test_transcribe.py`, `test_db.py`, `test_report.py` pass. Zero live LLM or Ollama calls.

### Manual Verification
- **Startup check**: kill Ollama → app shows error on screen, does not crash.
- **Full offline loop**: Wi-Fi off → 5-turn interview → PDF downloads and opens correctly.
- **Silence recovery**: stay silent for one turn → "didn't catch that" prompt appears, no crash.
- **Skip**: click Skip → `[skipped]` in transcript, report notes it.
- **PDF content**: all turns present, scorecard with 5 dimensions, summary paragraph.
- **Timing**: full 5-turn session under ~5 minutes on demo machine.
