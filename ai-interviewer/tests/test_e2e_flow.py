"""
tests/test_e2e_flow.py
-----------------------
End-to-end flow integration tests for Privacy-First AI Interviewer.
Verifies:
  1. Full 5-question interview submission loop.
  2. Plotly chart creation (_create_report_charts) returns populated figures.
  3. Report generation runs smoothly without exceptions.
  4. 3x back-to-back full session executions complete without errors.
"""

from __future__ import annotations

from unittest.mock import patch
import pytest

from app import _create_report_charts, generate_final_report, process_answer, start_interview


def _mock_llm_response(history, role, resume_context=None, **kwargs):
    turn_count = len(history) // 2 + 1
    return f"Question {turn_count}: Can you elaborate on your experience with this topic?", "system_design"


def _mock_score_session(history, session_id, role):
    return {
        "session_id": session_id,
        "overall_score": 4.2,
        "dimensions": [
            {"name": "technical_depth", "score": 4, "justification": "Detailed understanding of database indices and caching."},
            {"name": "communication_clarity", "score": 5, "justification": "Concise and well-structured responses."},
            {"name": "confidence_fluency", "score": 4, "justification": "High fluency with minimal hesitation."},
            {"name": "star_completeness", "score": 4, "justification": "Good coverage of Situation, Task, Action, and Result."},
            {"name": "problem_solving", "score": 4, "justification": "Structured approach to trade-off analysis."},
        ],
        "summary": "Demonstrated strong domain technical knowledge and clear communication.",
    }


def test_plotly_chart_creation_success(sample_session):
    """Verify _create_report_charts returns valid, non-None Plotly figures."""
    scorecard = sample_session["scorecard"]
    fig_radar, fig_bar = _create_report_charts(scorecard)

    assert fig_radar is not None
    assert fig_bar is not None
    assert fig_radar.layout.title.text == "5-Dimension Rubric Radar Chart"
    assert fig_bar.layout.title.text == "Rubric Dimension Breakdown"


def test_plotly_chart_creation_graceful_fallback_on_invalid_data():
    """Verify _create_report_charts degrades gracefully to (None, None) on empty/None input."""
    assert _create_report_charts(None) == (None, None)
    assert _create_report_charts({}) == (None, None)
    assert _create_report_charts({"dimensions": []}) == (None, None)


@patch("app.get_next_question", side_effect=_mock_llm_response)
@patch("app.score_session", side_effect=_mock_score_session)
def test_full_5_question_session_e2e(mock_score, mock_q):
    """
    Simulate a full 5-question interview flow from setup to Q5 report generation.
    """
    # 1. Start interview
    state, q1, _, turn_lbl, setup_err, _, _, tab = start_interview("Backend Engineer", False, None)
    assert state is not None
    assert turn_lbl == "Question 1 of 5"
    assert state["turn_index"] == 0

    # 2. Answers for Questions 1 through 4
    answers = [
        "I designed REST APIs and optimized PostgreSQL queries.",
        "We used Redis for session caching and rate limiting.",
        "I resolved a deadlock by enforcing strict lock ordering.",
        "I monitored p99 latency using Prometheus and Grafana.",
    ]

    for ans in answers:
        state, _, _, _, next_q, _, turn_lbl, finished = process_answer(ans, state)
        assert not finished
        assert "Question" in turn_lbl

    # 3. Answer Question 5 (final turn)
    ans_q5 = "I conducted post-mortems and added automated stress testing."
    state, _, _, _, end_q, _, turn_lbl, finished = process_answer(ans_q5, state)
    assert finished
    assert state["finished"] is True

    # 4. Generate final report
    state, summary_md, fig_radar, fig_bar, pdf_path, *rest = generate_final_report(state)
    assert summary_md is not None
    assert "overall_score" in summary_md.lower() or "overall score" in summary_md.lower()
    assert fig_radar is not None
    assert fig_bar is not None
    assert pdf_path is not None and pdf_path.endswith(".pdf")


@patch("app.get_next_question", side_effect=_mock_llm_response)
@patch("app.score_session", side_effect=_mock_score_session)
def test_3x_back_to_back_full_sessions(mock_score, mock_q):
    """
    Acceptance Criteria 4: Run full flow 3 times back-to-back to confirm stability.
    """
    for run in range(1, 4):
        state, _, _, _, _, _, _, _ = start_interview("System Design", False, None)
        for t in range(4):
            state, _, _, _, _, _, _, finished = process_answer(f"Answer {t+1} for run {run}", state)
            assert not finished

        state, _, _, _, _, _, _, finished = process_answer(f"Final answer for run {run}", state)
        assert finished

        state, summary_md, fig_radar, fig_bar, pdf_path, *rest = generate_final_report(state)
        assert fig_radar is not None
        assert fig_bar is not None
        assert pdf_path.endswith(".pdf")
