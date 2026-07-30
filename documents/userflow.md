# User Flow — Privacy-First AI Interviewer

## 1. Flow Diagram
```mermaid
flowchart TD
    A[Launch app locally] --> B[Setup screen:\npick role/domain, mic check]
    B --> C{Mic permission granted?}
    C -- No --> C1[Show troubleshooting hint,\nretry permission] --> C
    C -- Yes --> D[Show "🔒 100% Offline" badge\n+ first question, text + spoken]
    D --> E[Candidate clicks Record,\nspeaks answer, clicks Stop]
    E --> F[Transcribing... spinner]
    F --> G{Transcript empty/silent?}
    G -- Yes --> G1["I didn't catch that" prompt] --> E
    G -- No --> H[Transcript shown on screen]
    H --> I[Gemma 4 generates next\nfollow-up question]
    I --> J[Question shown + spoken via TTS]
    J --> K{More questions?\ne.g. 4-6 turn budget}
    K -- Yes --> E
    K -- No --> L[Session ends:\n"Generating your report..."]
    L --> M[Gemma 4 scores full transcript]
    M --> N[Report screen:\nscorecard + summary + PDF download]
    N --> O[Candidate downloads PDF\nor starts a new session]
```

## 2. Screen-by-Screen Detail

### Screen 1 — Setup
- Role/domain dropdown (e.g., Backend Engineer, HR Round, System Design).
- "Test microphone" button that records 2 seconds and plays it back, so candidates fix audio issues before the real interview starts.
- Visible, persistent badge: **"🔒 100% Offline — nothing you say leaves this device."**

### Screen 2 — Live Interview
- Current question shown as text (and spoken aloud via Piper).
- Record button (press to start, press again to stop — no silent auto-cutoff in MVP to keep behavior predictable for a live demo).
- Live transcript of the candidate's last answer appears once STT finishes, so the candidate can visually confirm they were understood correctly.
- Turn counter (e.g., "Question 3 of ~5") so the candidate knows roughly how long is left.

### Screen 3 — Report
- Overall score prominently displayed.
- Per-dimension table (score + one-line justification) — scannable in under 30 seconds, which matters for a live judging demo.
- Full transcript available as an expandable section (not the first thing shown — the scorecard is the headline).
- "Download PDF" button and "Start new session" button.

## 3. Edge Cases in the Flow
- **Candidate rambles very long** → STT still transcribes fully; Gemma 4's context window (128K+ for edge models) comfortably holds a handful of long answers, so no truncation logic is needed for a hackathon-length session.
- **Candidate wants to skip a question** → "Skip" button feeds a `[skipped]` placeholder into history so Gemma 4 doesn't ask about it again and the report notes it as unanswered.
- **Session interrupted (browser refresh)** → SQLite persistence means a `session_id` can, in principle, be resumed; for the MVP demo, simplest behavior is "refresh = new session," with resume listed as a stretch improvement.

## 4. Presentation Note
Because the judging rubric weights **Presentation & Writeup (20%)** and **Functionality (20%)** heavily, this flow is deliberately shallow — 3 screens, 4-6 conversational turns — so a live demo fits inside a 2–3 minute video with no dead air waiting on model inference.
