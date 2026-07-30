# System Design — Privacy-First AI Interviewer

> Companion to `architecture.md` (diagrams, tool comparisons) and `mvp.md` (build plan). This document goes one level deeper into how each module actually behaves.

## 1. Module Breakdown

### 1.1 UI Module (`app.py`)
- Screens: (1) Setup — pick role/domain, mic check; (2) Live Interview — question text/audio + record button + live transcript; (3) Report — scorecard + PDF download link.
- State held in Gradio `gr.State`: `session_id`, running `history` list, current turn index.
- Explicit **"🔒 Offline Mode" indicator** always visible — this is a presentation asset as much as a feature.

### 1.2 Audio Capture
- Uses Gradio's built-in `gr.Audio(source="microphone")` — avoids adding `sounddevice`/`PyAudio` as separate dependencies for the MVP.
- Manual stop-recording (button click) substitutes for real-time VAD in the MVP; Silero VAD is a stretch-goal upgrade to auto-detect end-of-speech and feel more natural.

### 1.3 STT Module (`stt/transcribe.py`)
- `faster-whisper` `small` (or `base.en` for English-only demos, faster) model, loaded once at startup (not per-turn) to avoid repeated load latency.
- Input: raw audio bytes from Gradio → written to a temp buffer/file → `model.transcribe(...)` → return concatenated segment text.
- Fallback/stretch: pass the raw clip directly to Gemma 4's native audio ASR prompt (`"Transcribe the following speech segment..."`) for a subset of turns, to explicitly demo Gemma 4 doing more than text reasoning.

### 1.4 Memory Module (`memory/db.py`)
- SQLite chosen over Redis/LangGraph/TinyDB for the MVP: zero infrastructure, single file, trivial schema (see `architecture.md §6`), and it's what actually needs to survive to build the final report.
- In-process, memory also lives as a plain Python list of `{speaker, content}` dicts passed directly into the LLM prompt each turn — SQLite is the durability layer, not the hot path.

### 1.5 LLM Integration Module (`llm/`)
- `client.py`: thin wrapper around the Ollama HTTP API (`POST /api/chat`) targeting the local `gemma4:4b` (E4B) tag.
- `prompts.py`: holds the **system prompt** (interviewer persona + role context + rubric definition) and the **JSON schema** the scoring call must return.
- `parser.py`: defensive JSON extraction (model output may wrap JSON in prose or code fences) — this is the single highest-risk integration point and gets dedicated unit tests (see `unittest.md`).

**Two distinct Gemma 4 calls per session:**
1. *Per-turn call* (plain text out): "Given this conversation so far, ask one focused, non-repetitive follow-up question appropriate for a {role} interview."
2. *End-of-session call* (strict JSON out): "Given the full transcript, score the candidate on the rubric below and justify each score in 1–2 sentences."

### 1.6 Scoring / Evaluation Module
Rubric dimensions (all 1–5 scale, each with a short justification string):
- **Technical depth** — correctness and depth of domain content.
- **Communication clarity** — structure, conciseness.
- **Confidence/fluency** — from transcript cues (filler words, hedging, coherence) — explicitly *not* from audio tone/prosody in the MVP, since that would require extra models.
- **Completeness (STAR)** — for behavioral questions, whether Situation/Task/Action/Result are all present.
- **Problem-solving approach** — reasoning process, not just the final answer.

Example JSON contract:
```json
{
  "session_id": "abc123",
  "overall_score": 3.8,
  "dimensions": [
    {
      "name": "technical_depth",
      "score": 4,
      "justification": "Correctly explained index usage and its tradeoffs, though didn't mention write-amplification costs."
    },
    {
      "name": "communication_clarity",
      "score": 4,
      "justification": "Answers were structured with a clear beginning-middle-end, minimal rambling."
    },
    {
      "name": "confidence_fluency",
      "score": 3,
      "justification": "Frequent use of 'I think maybe' and restarts suggest some hesitation."
    },
    {
      "name": "star_completeness",
      "score": 3,
      "justification": "Situation and Action were described; Result/impact was left vague."
    },
    {
      "name": "problem_solving",
      "score": 4,
      "justification": "Walked through a logical elimination of alternatives before settling on an approach."
    }
  ],
  "summary": "Solid technical fundamentals with room to tighten up quantifying outcomes and speaking with more certainty."
}
```

### 1.7 TTS Module (`tts/speak.py`)
- Piper CLI/Python binding, one pre-downloaded voice model, called synchronously right after a question is generated; audio played back through the browser via Gradio's audio output component.

### 1.8 Report Module (`report/generate_report.py`)
- Pulls `sessions`, `turns`, `scores` for a `session_id` from SQLite.
- ReportLab builds a PDF with: header (candidate/session info), full Q&A transcript, per-dimension score table, overall summary paragraph.
- Sample structure:
  1. Cover section — role, date, overall score.
  2. Scorecard table — dimension / score / one-line justification.
  3. Full transcript appendix.
  4. Summary & suggested next steps (from the `summary` field above).

## 2. Component Comparison Detail (condensed — full tables in `architecture.md`)
- **VAD**: cut from MVP (manual stop button); Silero VAD is the stretch-goal choice for real-time cutoff since it's lightweight and CPU-friendly.
- **Diarization**: out of scope entirely — single candidate, single mic, no multi-speaker need.
- **Knowledge base/RAG**: out of scope for MVP; if added, ChromaDB is the lightest-weight local vector store to index a resume/JD for question seeding.

## 3. Error Handling & Edge Cases
| Case | Handling |
|---|---|
| Silence / empty transcript | STT returns empty string → UI prompts "I didn't catch that, could you try again?" without calling the LLM |
| Ollama not running / model not pulled | Startup check with a clear on-screen error before the interview begins, not mid-session |
| Malformed JSON from scoring call | `parser.py` retries once with a stricter "return only JSON" reminder prompt; if it still fails, falls back to a template report noting scoring failed for that dimension |
| Mic permission denied | Gradio surfaces the browser permission prompt; app shows a one-line troubleshooting hint |

## 4. Non-Goals (repeated from PRD for engineering clarity)
No multi-user auth, no cloud fallback path, no persistent server deployment — this is a single-session, single-machine tool by design, which is itself the pitch.
