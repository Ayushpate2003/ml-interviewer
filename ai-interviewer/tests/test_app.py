"""
tests/test_app.py
------------------
Unit tests for app.py question formatting, turn progression, and error handling.
"""

from unittest.mock import patch
import pytest

from app import _format_question_md, process_answer, process_recording_stop, skip_question, start_interview


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

    state, q_text, audio_update, turn_lbl, setup_err, resume_status, timer_html, tab_update = start_interview("Backend Engineer")

    assert state is not None
    assert state["role"] == "Backend Engineer"
    assert state["turn_index"] == 0
    assert "Tell me about your past experience." in q_text
    assert turn_lbl == "Question 1 of 5"
    assert tab_update.selected == "interview"
    assert mock_get_next.called
    assert "Generic mode" in resume_status.get("value", "")


@patch("app.check_ollama_ready")
@patch("app.get_next_question")
@patch("app.speak")
@patch("app.extract_resume_highlights")
def test_start_interview_resume_mode(mock_extract, mock_speak, mock_get_next, mock_check):
    mock_check.return_value = (True, "")
    mock_get_next.return_value = ("Tell me about your project.", None)
    mock_speak.return_value = b"fake-audio-bytes"
    mock_extract.return_value = ("Python, Redis, Kafka", "ok")

    state, q_text, audio_update, turn_lbl, setup_err, resume_status, timer_html, tab_update = start_interview("Backend Engineer", resume_file="resume.txt")
    assert state["resume_mode"] == "resume"
    assert "Resume mode enabled" in resume_status.get("value", "")


@patch("app.check_ollama_ready")
def test_start_interview_failure_stays_on_setup(mock_check):
    mock_check.return_value = (False, "Ollama down")

    state, q_text, audio_update, turn_lbl, setup_err, resume_status, timer_html, tab_update = start_interview("Backend Engineer")

    assert state is None
    assert "Cannot start" in q_text
    assert tab_update.selected == "setup"


@patch("app.transcribe_native_gemma")
@patch("app.get_next_question")
@patch("app.speak")
@patch("app.increment_turns_completed")
def test_process_answer_multi_turn(mock_inc_turns, mock_speak, mock_get_next, mock_gemma_transcribe):
    mock_inc_turns.return_value = 1
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

    state, transcript, stt_badge, fluency_badge, q_text, audio_update, turn_lbl, finished = process_answer("I worked on caching.", initial_state)

    assert state["turn_index"] == 1
    assert transcript == "I worked on caching."
    assert "STT Engine" in stt_badge
    assert "Fluency Signal" in fluency_badge
    assert "How did you invalidate cache keys?" in q_text
    assert "Probing deeper on" in q_text
    assert turn_lbl == "Question 2 of 5"
    assert not finished


@patch("app.increment_turns_completed")
def test_process_answer_stops_at_five(mock_inc_turns):
    mock_inc_turns.return_value = 5
    st = {
        "session_id": "test-session-final",
        "role": "Backend Engineer",
        "history": [{"speaker": "interviewer", "content": "Q5"}],
        "turn_index": 4,
        "turns_completed": 4,
        "finished": False,
        "resume_mode": "generic",
        "pending_transcript": "final answer",
    }
    state, transcript, stt_badge, fluency_badge, q_text, audio_update, turn_lbl, finished = process_answer("", st)
    assert finished is True
    assert state["finished"] is True
    assert turn_lbl == "Question 5 of 5"


@patch("app.transcribe_native_gemma")
@patch("app.get_next_question")
@patch("app.increment_turns_completed")
def test_process_answer_handles_llm_error(mock_inc_turns, mock_get_next, mock_gemma_transcribe):
    mock_inc_turns.return_value = 1
    mock_gemma_transcribe.return_value = ("Candidate response text.", "🎙️ Gemma 4 Native Audio Perception")
    mock_get_next.side_effect = RuntimeError("Ollama server down")

    initial_state = {
        "session_id": "test-session-456",
        "role": "Backend Engineer",
        "history": [{"speaker": "interviewer", "content": "Question 1"}],
        "turn_index": 0,
        "finished": False,
    }

    state, transcript, stt_badge, fluency_badge, q_text, audio_update, turn_lbl, finished = process_answer("Candidate response text.", initial_state)

    assert "warning-box" in q_text
    assert "Interviewer Error" in q_text
    assert "Ollama server down" in q_text


@patch("app.get_next_question")
@patch("app.speak")
@patch("app.increment_turns_completed")
def test_skip_question_advances(mock_inc_turns, mock_speak, mock_get_next):
    mock_inc_turns.return_value = 1
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
    assert turn_lbl == "Question 2 of 5"


@patch("app.check_end_of_speech")
@patch("app.analyze_transcript_fluency")
@patch("app._transcribe_candidate_audio")
def test_process_recording_stop_decouples_optional_failures(mock_transcribe, mock_fluency, mock_vad):
    mock_transcribe.return_value = ("Transcript text", "⚡ STT Engine: faster-whisper")
    mock_vad.side_effect = RuntimeError("vad down")
    mock_fluency.side_effect = RuntimeError("fluency down")

    st = {
        "session_id": "test-session-stop",
        "role": "Backend Engineer",
        "history": [{"speaker": "interviewer", "content": "Q1"}],
        "turn_index": 0,
        "turns_completed": 0,
        "finished": False,
        "resume_mode": "generic",
    }

    steps = list(process_recording_stop("fake.wav", st))
    assert len(steps) == 2
    # Step 1: Immediate unified loading state
    loading_step = steps[0]
    assert loading_step[6].get("visible") is True
    assert "Analyzing your answer" in loading_step[6].get("value", "")

    # Step 2: Completion state with error isolation
    next_state, transcript, stt_badge, fluency_badge, vad_status, submit_error, status_update = steps[1]
    assert next_state["pending_transcript"] == "Transcript text"
    assert transcript == "Transcript text"
    assert "STT Engine" in stt_badge
    assert "Unavailable" in fluency_badge
    assert "unavailable" in vad_status.lower()
    assert submit_error.get("visible") is False
    assert status_update.get("visible") is False
