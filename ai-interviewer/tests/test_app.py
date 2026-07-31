"""
tests/test_app.py
------------------
Unit tests for app.py question formatting, turn progression, and error handling.
"""

from unittest.mock import patch
import pytest

from app import _format_question_md, start_interview, process_answer, skip_question


def test_format_question_md():
    # Normal question
    formatted = _format_question_md("What is indexed access in databases?")
    assert "Interviewer's Question" in formatted
    assert "What is indexed access in databases?" in formatted

    # Error message
    err_formatted = _format_question_md("⚠️ Ollama Error: connection refused")
    assert "warning-box" in err_formatted
    assert "Interviewer Error" in err_formatted

    # Empty text
    empty_formatted = _format_question_md("")
    assert "warning-box" in empty_formatted


@patch("app.check_ollama_ready")
@patch("app.get_next_question")
@patch("app.speak")
def test_start_interview(mock_speak, mock_get_next, mock_check):
    mock_check.return_value = (True, "")
    mock_get_next.return_value = ("Tell me about your past experience.", None)
    mock_speak.return_value = b"fake-audio-bytes"

    state, q_text, audio_update, turn_lbl, setup_err, resume_status, tab_update = start_interview("Backend Engineer")

    assert state is not None
    assert state["role"] == "Backend Engineer"
    assert state["turn_index"] == 0
    assert "Tell me about your past experience." in q_text
    assert turn_lbl == "Question 1 of ~5"
    assert tab_update.selected == "interview"
    mock_get_next.assert_called_once_with([], "Backend Engineer", resume_context="")


@patch("app.check_ollama_ready")
def test_start_interview_failure_stays_on_setup(mock_check):
    mock_check.return_value = (False, "Ollama down")

    state, q_text, audio_update, turn_lbl, setup_err, resume_status, tab_update = start_interview("Backend Engineer")

    assert state is None
    assert "Cannot start" in q_text
    assert tab_update.selected == "setup"


@patch("app.transcribe_native_gemma")
@patch("app.get_next_question")
@patch("app.speak")
def test_process_answer_multi_turn(mock_speak, mock_get_next, mock_gemma_transcribe):
    mock_gemma_transcribe.return_value = ("I worked on caching.", "🎙️ Gemma 4 Native Audio Perception")
    mock_get_next.return_value = ("How did you invalidate cache keys?", "Cache Invalidation")
    mock_speak.return_value = b"audio2"

    initial_state = {
        "session_id": "test-session-123",
        "role": "Backend Engineer",
        "history": [{"speaker": "interviewer", "content": "Tell me about your past experience."}],
        "turn_index": 0,
        "finished": False,
    }

    state, transcript, stt_badge, fluency_badge, q_text, audio_update, turn_lbl, finished = process_answer("fake_audio.wav", initial_state)

    assert state["turn_index"] == 1
    assert transcript == "I worked on caching."
    assert "Gemma 4 Native Audio Perception" in stt_badge
    assert "Fluency Signal" in fluency_badge
    assert "How did you invalidate cache keys?" in q_text
    assert "Probing deeper on" in q_text
    assert turn_lbl == "Question 2 of ~5"
    assert not finished


@patch("app.transcribe_native_gemma")
@patch("app.get_next_question")
def test_process_answer_handles_llm_error(mock_get_next, mock_gemma_transcribe):
    mock_gemma_transcribe.return_value = ("Candidate response text.", "🎙️ Gemma 4 Native Audio Perception")
    mock_get_next.side_effect = RuntimeError("Ollama server down")

    initial_state = {
        "session_id": "test-session-456",
        "role": "Backend Engineer",
        "history": [{"speaker": "interviewer", "content": "Question 1"}],
        "turn_index": 0,
        "finished": False,
    }

    state, transcript, stt_badge, fluency_badge, q_text, audio_update, turn_lbl, finished = process_answer("fake_audio.wav", initial_state)

    assert "warning-box" in q_text
    assert "Interviewer Error" in q_text
    assert "Ollama server down" in q_text


@patch("app.get_next_question")
@patch("app.speak")
def test_skip_question_advances(mock_speak, mock_get_next):
    mock_get_next.return_value = ("What is ACID?", None)
    mock_speak.return_value = b"audio_acid"

    initial_state = {
        "session_id": "test-session-789",
        "role": "System Design",
        "history": [{"speaker": "interviewer", "content": "Question 1"}],
        "turn_index": 0,
        "finished": False,
    }

    state, transcript, q_text, audio_update, turn_lbl, finished = skip_question(initial_state)

    assert state["turn_index"] == 1
    assert transcript == "[skipped]"
    assert "What is ACID?" in q_text
    assert turn_lbl == "Question 2 of ~5"
