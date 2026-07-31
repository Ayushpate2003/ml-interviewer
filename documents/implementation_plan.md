# Implementation Plan - Phase 1: Real VAD (Silero) Speech Auto-Stop

Implement Silero VAD to detect end-of-speech silence (~1.5s–2.0s) on incoming candidate audio, auto-triggering answer submission while preserving the manual Record/Stop override.

## User Review Required

> [!IMPORTANT]
> **Key Architecture Decisions:**
> 1. **Silero VAD Model**: We use `silero-vad` (version 6.2.1) with PyTorch, operating at a 16kHz sampling rate.
> 2. **Silence Threshold**: We set a **1.5s silence threshold** after speech has started. A brief pause (< 1.5s) will NOT trigger auto-stop, avoiding false cutoffs during natural pauses.
> 3. **Manual Fallback**: The manual Record/Stop button remains fully operational so users can override or manually stop at any point.
> 4. **Visual Indicator**: A dynamic `vad_status` Markdown badge ("🎙️ Listening..." -> "✓ Got it (end-of-speech detected)") provides live feedback.

## Proposed Changes

### VAD Module (`ai-interviewer/stt/vad.py`)

#### [NEW] [vad.py](file:///Users/astra/projects/github%20clone/ml-interviewer/ai-interviewer/stt/vad.py)
- Create `load_vad_model()` to initialize Silero VAD model safely once.
- Create `check_end_of_speech(audio_filepath_or_bytes, silence_threshold_sec=1.5)`:
  - Reads audio file/bytes and extracts speech timestamps using `get_speech_timestamps`.
  - Determines if speech was detected AND if the end of speech is followed by >= 1.5 seconds of trailing silence.
  - Returns `(is_end_of_speech: bool, status_msg: str)`.

### UI & Handlers (`ai-interviewer/app.py`)

#### [MODIFY] [app.py](file:///Users/astra/projects/github%20clone/ml-interviewer/ai-interviewer/app.py)
- Import VAD module and initialize model at startup.
- Add `vad_status` Markdown badge on the **Live Interview** tab above `answer_audio`.
- Update `process_answer` to check VAD state and update `vad_status` badge ("✓ Got it" on end-of-speech, "🎙️ Listening..." during active recording).

### Automated Tests (`ai-interviewer/tests/`)

#### [NEW] [test_vad.py](file:///Users/astra/projects/github%20clone/ml-interviewer/ai-interviewer/tests/test_vad.py)
- Test `check_end_of_speech` with mock/sample audio clips (silence vs active speech).
- Test graceful degradation if audio file is corrupt or unreadable.

## Verification Plan

### Automated Tests
- Run full pytest test suite:
  ```bash
  cd ai-interviewer && .venv/bin/python -m pytest tests/ -v
  ```

### Manual Integration Checklist (Phase 1)
1. **Auto-Stop Verification**: Speak an answer, pause for ~1.5s–2.0s -> verify VAD badge updates to "✓ Got it" and process proceeds.
2. **Short Pause Verification**: Pause for < 1.0s mid-answer -> verify recording does NOT cut off prematurely.
3. **Manual Override**: Press manual Stop button -> verify manual submission functions as expected.
4. **Full 5-Turn Offline Run**: Execute full 5-turn interview with Wi-Fi off to verify end-to-end flow per `unittest.md §4`.

### Memory Update
- Write Phase 1 Graphiti memory episode documenting Silero VAD version, threshold tuning, and test results.
