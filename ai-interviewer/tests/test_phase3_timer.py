"""
tests/test_phase3_timer.py
--------------------------
Unit tests for Phase 3: Per-question timer with non-blocking expiration and alerts.
"""

from unittest.mock import patch
import pytest

from app import _build_timer_html, process_answer, start_interview


def test_build_timer_html_format():
    html = _build_timer_html(90)
    assert 'id="timer-counter">90<' in html
    assert 'timer-display' in html
    assert 'playTone' in html
    assert "Time's up" in html


@patch("app.check_ollama_ready", return_value=(True, ""))
@patch("app.get_next_question", return_value=("Explain database indexing.", None))
@patch("app.speak", return_value=b"audio")
def test_timer_initialization_in_start_interview(mock_speak, mock_get_next, mock_check):
    state, q_text, q_audio, turn_lbl, setup_err, r_status, timer_html, tab = start_interview("Backend Engineer", timer_seconds=60)
    assert state["timer_seconds"] == 60
    assert 'id="timer-counter">60<' in timer_html


@patch("app.increment_turns_completed", return_value=1)
@patch("app.get_next_question", return_value=("Explain B-trees.", None))
@patch("app.speak", return_value=b"audio2")
def test_non_blocking_timer_expiration_allows_normal_submission(mock_speak, mock_get_next, mock_inc_turns):
    st = {
        "session_id": "test-timer-expired",
        "role": "Backend Engineer",
        "history": [{"speaker": "interviewer", "content": "Q1"}],
        "turn_index": 0,
        "max_turns": 5,
        "timer_seconds": 60,
        "finished": False,
    }

    # Candidate submits answer after timer expiration
    state, transcript, stt_badge, fluency_badge, q_text, audio_out, turn_lbl, finished = process_answer(
        "Candidate answered late after timer expired.", st
    )

    assert finished is False
    assert transcript == "Candidate answered late after timer expired."
    assert "Explain B-trees." in q_text
