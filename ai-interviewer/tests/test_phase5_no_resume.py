"""
tests/test_phase5_no_resume.py
-------------------------------
Unit and integration tests for Phase 5: No-resume passthrough verification.
"""

from unittest.mock import patch
import pytest

from app import generate_final_report, process_answer, start_interview


@patch("app.check_ollama_ready", return_value=(True, ""))
@patch("app.get_next_question", return_value=("Explain object-oriented programming.", None))
@patch("app.speak", return_value=b"fake-audio")
def test_start_interview_no_resume_generic_mode(mock_speak, mock_get_next, mock_check):
    state, q_text, audio_update, turn_lbl, setup_err, resume_status, timer_html, tab_update = start_interview(
        "Backend Engineer", False, None
    )

    assert state["resume_mode"] == "generic"
    assert state["resume_context"] == ""
    assert "Generic mode (no resume detected)" in resume_status.get("value", "")
    assert "Explain object-oriented programming." in q_text


@patch("app.check_ollama_ready", return_value=(True, ""))
@patch("app.get_next_question", return_value=("What is REST?", None))
@patch("app.speak", return_value=b"fake-audio")
@patch("app.increment_turns_completed")
@patch("app.score_session")
@patch("app.generate_report", return_value="/tmp/test_report.pdf")
def test_full_no_resume_session_e2e(mock_pdf, mock_score, mock_inc_turns, mock_speak, mock_get_next, mock_check):
    mock_score.return_value = {
        "overall_score": 85.0,
        "summary": "Solid technical knowledge.",
        "dimensions": [
            {"dimension": "Technical Accuracy", "score": 8.5, "justification": "Good"},
            {"dimension": "Communication Clarity", "score": 8.0, "justification": "Clear"},
            {"dimension": "Problem Solving", "score": 8.5, "justification": "Structured"},
            {"dimension": "System Design", "score": 8.0, "justification": "Good layout"},
            {"dimension": "Cultural Fit", "score": 9.0, "justification": "Great fit"},
        ],
    }

    state, _, _, _, _, _, _, _ = start_interview("Backend Engineer", False, None, num_questions=3)
    assert state["resume_mode"] == "generic"

    for t in range(1, 3):
        mock_inc_turns.return_value = t
        state, _, _, _, _, _, _, finished = process_answer(f"Generic answer {t}", state)
        assert not finished

    mock_inc_turns.return_value = 3
    state, _, _, _, _, _, _, finished = process_answer("Generic answer 3", state)
    assert finished is True

    # Report generation
    st, scorecard, fig_radar, fig_bar, pdf_path, *rest = generate_final_report(state)
    assert scorecard is not None
    assert "85" in scorecard
    assert pdf_path == "/tmp/test_report.pdf"
