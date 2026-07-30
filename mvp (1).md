# MVP — 1-Day Build Plan

## Guiding Principle
Judges score **Gemma Integration, Impact, Functionality, Presentation** — not code elegance or feature count. Everything below is chosen to maximize a *working, demoable* loop where Gemma 4 is visibly doing the thinking.

## MVP Feature Set

### ✅ Must Build (core loop)
1. Gradio web UI with a mic-record component and a "start interview" screen (role/domain picker).
2. Local STT: **faster-whisper (`small` or `base.en`)** transcribes each recorded answer.
3. Conversation memory: a simple Python list of `{role, content}` turns, mirrored into SQLite so a report can be generated after the session (and so state survives a Gradio refresh).
4. Gemma 4 (via **Ollama**, `E4B` variant) as the single reasoning core:
   - Given the transcript + history, generates the next interview question.
   - After the session, scores the whole transcript against a rubric and returns structured JSON.
5. Local TTS: **Piper** speaks each generated question aloud.
6. Report generation: render the JSON score + transcript into a clean PDF (ReportLab) the user can download.

### 🟡 Should Build (if time allows, ~last 2 hours)
7. Role-specific question seeding (pick "Backend Engineer" vs "HR Round" vs "System Design" → different system prompt).
8. A visible "🔒 100% Offline — no audio leaves this device" badge/toggle in the UI that judges can point to.

### 🟢 Stretch (only if core loop is done early)
9. Feed the raw audio clip directly into **Gemma 4 E4B's native audio understanding** for one "wow" demo turn (bypassing whisper entirely) — great for the Gemma Integration score, but keep faster-whisper as the default reliable path since Gemma 4's audio input is capped at 30 seconds per clip.
10. Resume/JD upload → lightweight keyword-based question seeding (skip full RAG/embeddings — not worth the setup time in a 1-day sprint).

### ❌ Explicitly Cut
- Speaker diarization, VAD tuning (use Gradio's built-in stop-recording button instead of Silero VAD — one less dependency to debug on demo day).
- Multi-language UI, emotion/eye-contact detection, dashboards, vector DB, Redis, LangGraph — all deferred to "Stretch Goals" in `architecture.md`.

## Hour-by-Hour Plan (assumes ~10 working hours)
| Time | Task |
|---|---|
| H0–H1 | Environment setup: `ollama pull gemma4:4b` (or the E4B tag Ollama publishes), `pip install faster-whisper gradio reportlab piper-tts`. Verify mic works in Gradio. |
| H1–H2.5 | Build audio capture + faster-whisper transcription; print transcript to console/UI. |
| H2.5–H4.5 | Build the Gemma 4 prompt loop: system prompt with interviewer persona + rubric, feed transcript history, parse next-question output. Get one full manual Q→A→Q loop working via CLI before wiring to UI. |
| H4.5–H6 | Wire into Gradio UI: mic in → transcript → next question out → Piper speaks it. |
| H6–H7.5 | Build end-of-session scoring call (Gemma 4 returns strict JSON) + ReportLab report generation. |
| H7.5–H9 | Polish: role picker, offline badge, error handling for silence/empty transcript, basic styling. |
| H9–H10 | Record demo video, write Kaggle/README write-up, rehearse live run-through twice. |

## Definition of Done
- A judge can start the app with no internet connection, speak 3–4 answers to spoken/on-screen questions, and receive a downloadable PDF report with per-dimension scores and justifications, entirely on one machine.

## Minimal Tech Stack (see `architecture.md` for the full comparison tables)
| Layer | Choice | Why this one for a 1-day build |
|---|---|---|
| UI | Gradio | Built-in `gr.Audio` mic component — no separate audio-capture library needed |
| STT | faster-whisper (`small`) | `pip install`-able, CPU-friendly, no GPU required |
| Reasoning | Gemma 4 E4B via Ollama | One-command install (`ollama run gemma4`), fits 8GB RAM, native audio + function calling if you want the stretch goal |
| Memory | Python list + SQLite | Zero infra, survives restarts, trivial schema |
| TTS | Piper | Fast, small, natural-enough voices, simple CLI |
| Report | ReportLab | Generates PDF directly, no browser/HTML-to-PDF dependency to install |
