# Task Tracker — Privacy-First AI Interviewer

## Phase 0 — Bootstrap & Memory Init
- [ ] Create folder structure (architecture.md §5)
- [ ] Create all stub files
- [ ] Write Phase 0 build log entry

## Phase 1 — Environment Setup
- [ ] Create requirements.txt
- [ ] Create venv and install dependencies
- [ ] Verify Ollama startup check logic in app.py
- [ ] Write Phase 1 build log entry

## Phase 2 — Audio Capture + STT
- [ ] Implement stt/transcribe.py (TranscriptionError, load_stt_model, transcribe)
- [ ] Create test fixtures (silent WAV, sample WAV)
- [ ] Write and run tests/test_transcribe.py (3 tests from unittest.md §3.2)
- [ ] Write Phase 2 build log entry

## Phase 3 — Memory / DB
- [ ] Implement memory/schema.sql (exact DDL from architecture.md §6)
- [ ] Implement memory/db.py (create_session, add_turn, get_turns, save_scores, get_scores)
- [ ] Write and run tests/test_db.py (2 tests from unittest.md §3.3)
- [ ] Write Phase 3 build log entry

## Phase 4 — Gemma 4 Reasoning Loop
- [ ] Implement llm/client.py (Ollama /api/chat wrapper)
- [ ] Implement llm/prompts.py (system prompt template + scoring JSON schema)
- [ ] Implement llm/parser.py (defensive JSON extraction)
- [ ] Write and run tests/test_parser.py (4 tests from unittest.md §3.1)
- [ ] Write and run tests/test_client.py (2 mocked tests from unittest.md §3.5)
- [ ] Write Phase 4 build log entry

## Phase 5 — Scoring / Evaluation
- [ ] Add score_session() to llm/client.py
- [ ] Add retry-once-on-malformed-JSON to llm/parser.py
- [ ] Verify 5-dimension JSON contract matches system-design.md §1.6 exactly
- [ ] Write Phase 5 build log entry

## Phase 6 — TTS Module
- [ ] Implement tts/speak.py (Piper loader + speak function)
- [ ] Manual listening check
- [ ] Write Phase 6 build log entry

## Phase 7 — Report Module
- [ ] Implement report/generate_report.py (ReportLab PDF: cover + scorecard + transcript + summary)
- [ ] Write and run tests/test_report.py (3 tests from unittest.md §3.4)
- [ ] Write Phase 7 build log entry

## Phase 8 — UI Wiring
- [ ] Implement app.py Screen 1: Setup (role dropdown, mic test, offline badge)
- [ ] Implement app.py Screen 2: Live Interview (question+TTS, record/stop, transcript, turn counter, skip)
- [ ] Implement app.py Screen 3: Report (scorecard, expandable transcript, PDF download, new session)
- [ ] Wire all edge cases (silence, skip, mic permission hint)
- [ ] Write Phase 8 build log entry

## Phase 9 — Integration Test Pass
- [ ] Run full automated suite: pytest tests/ -v
- [ ] Manual 5-turn offline checklist (unittest.md §4)
- [ ] Fix any bugs found
- [ ] Write Phase 9 build log entry

## Phase 10 — Polish & Demo Prep
- [ ] Role-specific prompts (Backend Engineer, HR Round, System Design)
- [ ] Final "🔒 100% Offline" badge polish
- [ ] Write README.md
- [ ] Write Phase 10 build log entry
- [ ] Write final "ai-interviewer – Build Complete" episode
