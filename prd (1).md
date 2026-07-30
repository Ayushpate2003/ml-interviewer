# PRD — Privacy-First AI Interviewer (powered by Gemma 4)

## 1. Problem Statement
Candidates preparing for interviews rarely get realistic, judgment-free mock interview practice. Existing "AI interviewer" tools are cloud-based — they upload raw voice recordings to third-party servers, which is a non-starter for sensitive use cases (internal promotion practice, HR screening rehearsal, students discussing personal projects). There is no widely-available tool that runs a **full conversational interview loop entirely on-device**, with no audio ever leaving the machine.

## 2. Goal
Build an offline, privacy-first AI interviewer that listens to a candidate's spoken answers, asks intelligent dynamic follow-ups, and produces a recruiter-ready scored report — with **Gemma 4 as the core reasoning engine** for both understanding and evaluation, not a bolt-on chatbot wrapper.

## 3. Target Users
- Candidates rehearsing technical/behavioral interviews before a real one.
- Bootcamps/campus placement cells running mock-interview drives at scale (no per-seat cloud API cost, no data leaving the lab network).
- Privacy-sensitive orgs (defense, healthcare, finance) that can't send interview audio to a third party.

## 4. Core Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | Capture candidate speech via microphone in-browser | Must |
| FR2 | Convert speech to text locally (no cloud STT) | Must |
| FR3 | Maintain full conversation history/context across turns | Must |
| FR4 | Generate a contextual, non-repetitive follow-up question using Gemma 4 | Must |
| FR5 | Score each answer across defined rubric dimensions using Gemma 4 (structured JSON output) | Must |
| FR6 | Speak the next question aloud via local TTS | Should |
| FR7 | Generate a downloadable, recruiter-ready report (PDF/DOCX) at the end of the session | Must |
| FR8 | Let the user pick an interview domain/role (e.g., "Backend Engineer", "HR round") | Should |
| FR9 | Resume-aware question generation (parse an uploaded resume/JD to seed the first questions) | Could (stretch) |
| FR10 | Multi-language interview support (leveraging Gemma 4's 140+ language training) | Could (stretch) |

## 5. Non-Functional Requirements
- **Privacy**: No network calls carrying audio, transcript, or resume content. The only permitted network activity is the initial one-time model download.
- **Latency**: Perceived turn latency (candidate stops talking → next question appears) should be under ~6–8 seconds on a mid-range laptop CPU/GPU — acceptable for a live demo, not production-grade.
- **Portability**: Must run on a single laptop with no internet during the actual interview, per the offline/on-device requirement.
- **Explainability**: Every score Gemma 4 emits must come with a short natural-language justification, not a bare number — this is what makes the report "recruiter-ready" instead of a black box.

## 6. Success Metrics (mapped to judging rubric)
| Rubric criterion | How this PRD addresses it |
|---|---|
| Gemma Integration (30%) | Gemma 4 is the *only* model doing reasoning: follow-up generation, scoring, and (optionally) audio understanding itself — not a rules engine with an LLM sprinkled on top. |
| Innovation & Impact (30%) | Solves a real, common pain point (interview anxiety/practice access) with a genuinely novel constraint (zero cloud, zero data exposure) that most competitors ignore. |
| Functionality (20%) | End-to-end working loop: speak → transcribe → reason → respond → score → report, demoable live without internet. |
| Presentation (20%) | Clear narrative: "your interview never leaves your laptop" is an easy, demo-friendly hook; report output is a tangible, screenshot-able artifact. |

## 7. Out of Scope (for the hackathon build)
- Speaker diarization (single candidate, single mic — not needed).
- Video/webcam analysis, eye-contact or emotion detection (listed as stretch goals only).
- Multi-user concurrent sessions / cloud deployment / auth.
- Production-grade error recovery, packaging, or installers.

## 8. Assumptions
- Demo machine has at least 8GB RAM (16GB preferred) and a working microphone/speakers.
- Judges will run or watch a local demo; no hosted URL is required.
- "Gemma 4" refers to Google DeepMind's Gemma 4 family (E2B/E4B/12B/26B-MoE/31B, Apache 2.0, released April 2026), run locally via Ollama or Hugging Face Transformers.

## 9. Risks
| Risk | Mitigation |
|---|---|
| Local LLM inference too slow on judges' hardware | Default to Gemma 4 **E2B/E4B** (edge sizes, 2–5GB RAM) rather than 26B/31B |
| STT accuracy drops on accented/noisy speech | Use faster-whisper `small`/`base.en` as a robust fallback even if using Gemma 4's native audio path for the "wow" demo |
| Demo-day Wi-Fi/model-pull failures | Pre-download all models the night before; never rely on live network during judging |
