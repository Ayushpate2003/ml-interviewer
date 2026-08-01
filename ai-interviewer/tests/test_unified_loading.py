"""
tests/test_unified_loading.py
------------------------------
Unit tests for the unified loading state and atomic grouped outputs on Screen 2.
"""

from unittest.mock import patch
import pytest

from app import process_recording_stop


@patch("app.check_end_of_speech")
@patch("app.analyze_transcript_fluency")
@patch("app._transcribe_candidate_audio")
def test_unified_loading_yields_step1_and_step2(mock_transcribe, mock_fluency, mock_vad):
    mock_transcribe.return_value = ("I optimized a database query.", "⚡ STT Engine: faster-whisper")
    mock_vad.return_value = (False, "Silence detected (end of speech)")
    mock_fluency.return_value = (0.95, 0.05, "📊 **Fluency Signal:** 🟢 Confident")

    st = {
        "session_id": "test-session-unified",
        "role": "Backend Engineer",
        "history": [{"speaker": "interviewer", "content": "Question 1"}],
        "turn_index": 0,
        "finished": False,
        "resume_mode": "generic",
    }

    steps = list(process_recording_stop("audio.wav", st))
    assert len(steps) == 2

    # Step 1: Immediate unified loading banner shown
    step1 = steps[0]
    state_1, transcript_1, stt_1, fluency_1, vad_1, err_1, status_1 = step1
    assert status_1.get("visible") is True
    assert "Analyzing your answer" in status_1.get("value", "")
    assert transcript_1 == ""

    # Step 2: Atomic grouped outputs revealed together and banner hidden
    step2 = steps[1]
    state_2, transcript_2, stt_2, fluency_2, vad_2, err_2, status_2 = step2
    assert status_2.get("visible") is False
    assert transcript_2 == "I optimized a database query."
    assert "faster-whisper" in stt_2
    assert "Confident" in fluency_2
    assert "Silence detected" in vad_2
    assert err_2.get("visible") is False


@patch("app.check_end_of_speech")
@patch("app.analyze_transcript_fluency")
@patch("app._transcribe_candidate_audio")
def test_unified_loading_handles_subsignal_failure_decoupled(mock_transcribe, mock_fluency, mock_vad):
    mock_transcribe.return_value = ("I used Kafka for event streaming.", "⚡ STT Engine: faster-whisper")
    mock_vad.side_effect = Exception("VAD runtime error")
    mock_fluency.side_effect = Exception("Fluency runtime error")

    st = {
        "session_id": "test-session-decoupled",
        "role": "System Design",
        "history": [{"speaker": "interviewer", "content": "Question 1"}],
        "turn_index": 0,
        "finished": False,
    }

    steps = list(process_recording_stop("audio.wav", st))
    assert len(steps) == 2

    step2 = steps[1]
    _, transcript_2, stt_2, fluency_2, vad_2, _, status_2 = step2
    assert status_2.get("visible") is False
    assert transcript_2 == "I used Kafka for event streaming."
    assert "faster-whisper" in stt_2
    assert "Unavailable" in fluency_2
    assert "unavailable" in vad_2.lower()
