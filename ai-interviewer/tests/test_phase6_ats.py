"""
tests/test_phase6_ats.py
-------------------------
Unit tests for Phase 6: Local/offline ATS resume scoring and report integration.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from report.generate_report import generate_report
from utils.ats import calculate_ats_score


def test_calculate_ats_score_backend_engineer():
    resume_text = "Senior Python Engineer experienced with SQL, PostgreSQL, REST APIs, Microservices, Docker, Git, CI/CD, and AWS."
    res = calculate_ats_score(resume_text, role="Backend Engineer")

    assert res["score"] is not None
    assert res["score"] > 40
    assert "python" in res["matched"]
    assert "sql" in res["matched"]
    assert "docker" in res["matched"]
    assert len(res["suggestions"]) > 0
    assert "Resume ATS Score" in res["formatted_md"]


def test_calculate_ats_score_empty_text():
    res = calculate_ats_score("", role="Backend Engineer")
    assert res["score"] is None
    assert res["matched"] == []
    assert res["missing"] == []
    assert res["suggestions"] == []
    assert res["formatted_md"] == ""


def test_pdf_report_includes_ats_section(tmp_path):
    session = {
        "session_id": "test-ats-pdf-1234",
        "role": "Backend Engineer",
        "turns": [{"speaker": "interviewer", "content": "Q1"}, {"speaker": "candidate", "content": "A1"}],
        "scorecard": {"overall_score": 4.5, "dimensions": [{"name": "Technical Accuracy", "score": 4.5, "justification": "Great"}]},
        "ats_info": {
            "score": 82,
            "matched": ["python", "sql", "docker"],
            "missing": ["kubernetes"],
            "suggestions": ["Add Kubernetes to increase score."],
        },
    }

    pdf_path = generate_report(session, out_dir=tmp_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
