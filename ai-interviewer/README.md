# Privacy-First AI Interviewer 🎙️🔒

> A fully offline, single-machine spoken mock-interview app powered by **Gemma 4** (via Ollama).
> Nothing you say ever leaves your device.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## What It Does

1. **Listens** to you speak via your browser microphone.
2. **Transcribes** your answer locally using `faster-whisper` (no cloud STT).
3. **Asks intelligent follow-ups** using Gemma 4 (via Ollama) — same model does both reasoning and evaluation.
4. **Speaks** each question aloud using Piper TTS.
5. **Scores** your full session across 5 rubric dimensions (structured JSON from Gemma 4).
6. **Generates** a recruiter-ready PDF report with your scorecard and full transcript.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| UI | Gradio | Built-in mic component, offline-capable local server |
| STT | faster-whisper (`small`) | pip-installable, CPU-friendly, no GPU required |
| LLM Runtime | Ollama | One-command install, serves Gemma 4 locally |
| Core Model | Gemma 4 E4B (`gemma4:4b`) | Best reasoning/speed for laptop; Apache-2.0 licensed |
| Memory | Python list + SQLite | Zero infra, survives restarts |
| TTS | Piper | Fast on CPU, simple CLI/Python API |
| Report | ReportLab | Direct PDF, no browser dependency |

---

## Hardware Requirements

- **Minimum:** 8 GB RAM, quad-core CPU, working mic/speakers → runs E4B comfortably.
- **Recommended:** 16 GB RAM or RTX 3060+ GPU → enables the 12B model for higher-quality reasoning.
- **Network:** Only needed for one-time model downloads. No internet required during the interview.

---

## Setup

### 1. Install Ollama & pull Gemma 4

```bash
# Install Ollama from https://ollama.ai
ollama pull gemma4:4b
ollama serve   # keep this running in a separate terminal
```

### 2. Create a Python virtual environment

```bash
cd ai-interviewer
python3 -m venv .venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Piper TTS on macOS:** If `piper-tts` fails to install, download the binary from
> [rhasspy/piper releases](https://github.com/rhasspy/piper/releases) and place it on your `PATH`.
> The app will detect and use the CLI binary automatically.

### 4. Run the app

```bash
python app.py
# Open http://localhost:7860 in your browser
```

If Ollama isn't running or the model isn't pulled, the app will show a clear error banner before the interview starts — it never fails mid-session.

---

## Running Tests

All tests are deterministic and require **no live Ollama/model calls**:

```bash
cd ai-interviewer
pytest tests/ -v --tb=short
```

Expected output: all tests in `test_parser.py`, `test_transcribe.py`, `test_db.py`, `test_report.py`, `test_client.py` pass.

---

## Project Structure

```
ai-interviewer/
├── app.py                  # Gradio entrypoint (3-screen UI)
├── requirements.txt
├── stt/
│   └── transcribe.py       # faster-whisper wrapper
├── llm/
│   ├── client.py           # Ollama/Gemma 4 HTTP wrapper
│   ├── prompts.py          # System prompts, rubric, JSON schema
│   └── parser.py           # Defensive JSON extraction
├── memory/
│   ├── db.py               # SQLite CRUD
│   └── schema.sql          # sessions / turns / scores schema
├── tts/
│   └── speak.py            # Piper TTS wrapper
├── report/
│   └── generate_report.py  # ReportLab PDF builder
├── data/                   # SQLite DB + Piper voice models
└── tests/
    ├── conftest.py
    ├── test_parser.py
    ├── test_transcribe.py
    ├── test_db.py
    ├── test_report.py
    └── test_client.py
```

---

## Definition of Done ✅

> A judge can start the app **with no internet connection**, speak 3–4 answers to spoken/on-screen questions, and receive a **downloadable PDF report** with per-dimension scores and justifications, entirely on one machine.

---

## Privacy Guarantee

- Audio bytes are transcribed locally and immediately discarded (never written to disk permanently).
- Transcripts and scores are stored only in the local SQLite DB (`data/interview_sessions.db`).
- No network calls are made during an interview session — only the one-time model downloads require internet.

---

## Interview Roles

| Role | Focus Areas |
|---|---|
| Backend Engineer | APIs, databases, caching, concurrency, debugging |
| HR Round | STAR behavioural questions, motivation, teamwork |
| System Design | Distributed systems, trade-offs, capacity estimation |

---

## Stretch Goals (post-MVP)

- Gemma 4 native audio for one demo turn (bypasses faster-whisper entirely, 30s clip cap applies).
- Resume/JD keyword upload → question seeding.
- ChromaDB/FAISS RAG over past interview sessions.
- Real-time coaching hints during the answer.
- Multi-language support (Gemma 4 supports 140+ languages).
