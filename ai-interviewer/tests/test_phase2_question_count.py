"""
tests/test_phase2_question_count.py
------------------------------------
Unit tests for Phase 2: User-selectable question quantity (3 / 5 / 7 / 10).
"""

from unittest.mock import patch
import pytest

from app import process_answer, start_interview


@patch("app.check_ollama_ready", return_value=(True, ""))
@patch("app.get_next_question", return_value=("What is an API?", None))
@patch("app.speak", return_value=b"fake-audio")
@patch("app.increment_turns_completed")
def test_3_question_session_completes_at_turn_3(mock_inc_turns, mock_speak, mock_get_next, mock_check):
    state, _, _, turn_lbl, _, _, _, _ = start_interview("Backend Engineer", False, None, num_questions=3)
    assert state["max_turns"] == 3
    assert turn_lbl == "Question 1 of 3"

    mock_inc_turns.return_value = 1
    state, _, _, _, _, _, turn_lbl, finished = process_answer("Answer 1", state)
    assert not finished
    assert turn_lbl == "Question 2 of 3"

    mock_inc_turns.return_value = 2
    state, _, _, _, _, _, turn_lbl, finished = process_answer("Answer 2", state)
    assert not finished
    assert turn_lbl == "Question 3 of 3"

    # Turn 3 completes the session
    mock_inc_turns.return_value = 3
    state, _, _, _, _, _, turn_lbl, finished = process_answer("Answer 3", state)
    assert finished is True
    assert state["finished"] is True
    assert turn_lbl == "Question 3 of 3"


@patch("app.check_ollama_ready", return_value=(True, ""))
@patch("app.get_next_question", return_value=("What is microservices?", None))
@patch("app.speak", return_value=b"fake-audio")
@patch("app.increment_turns_completed")
def test_7_question_session_completes_at_turn_7(mock_inc_turns, mock_speak, mock_get_next, mock_check):
    state, _, _, turn_lbl, _, _, _, _ = start_interview("System Design", False, None, num_questions=7)
    assert state["max_turns"] == 7
    assert turn_lbl == "Question 1 of 7"

    for t in range(1, 7):
        mock_inc_turns.return_value = t
        state, _, _, _, _, _, turn_lbl, finished = process_answer(f"Answer {t}", state)
        assert not finished

    # Turn 7 completes session
    mock_inc_turns.return_value = 7
    state, _, _, _, _, _, turn_lbl, finished = process_answer("Answer 7", state)
    assert finished is True
    assert state["finished"] is True
    assert turn_lbl == "Question 7 of 7"
