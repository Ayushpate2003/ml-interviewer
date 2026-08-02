"""
tests/test_phase4_resume_synthesis.py
--------------------------------------
Unit & integration tests for Phase 4: Resume Improvement Suggestions.
Verifies prompt construction, LLM synthesis, clean omission when no resume is uploaded,
and PDF rendering.
"""

from unittest.mock import MagicMock, patch
import pytest

from llm.client import generate_resume_improvements
from llm.prompts import build_resume_improvement_prompt
from report.generate_report import _build_resume_improvements_section, generate_report


def test_build_resume_improvement_prompt():
    resume_text = "Senior Python Developer with 5 years experience in Django and PostgreSQL."
    ats_info = {"missing": ["Docker", "Kubernetes", "Redis"]}
    scorecard = {"overall_score": 80}

    prompt = build_resume_improvement_prompt(resume_text, ats_info, scorecard)
    assert "Docker, Kubernetes, Redis" in prompt
    assert "suggestions" in prompt


@patch("llm.client._chat", return_value='{"suggestions": ["Highlight Redis caching experience in Django projects.", "Quantify Kubernetes cluster deployment scale."]}')
def test_generate_resume_improvements_success(mock_chat):
    resume_text = "Experienced Backend Engineer."
    ats_info = {"missing": ["Redis", "Kubernetes"]}
    scorecard = {"overall_score": 75}

    sug = generate_resume_improvements(resume_text, ats_info, scorecard)
    assert len(sug) == 2
    assert "Redis" in sug[0]


def test_generate_resume_improvements_omitted_when_no_resume():
    sug_none = generate_resume_improvements(None, {"missing": ["Docker"]}, {"overall_score": 80})
    assert sug_none == []

    sug_empty = generate_resume_improvements("   ", {"missing": ["Docker"]}, {"overall_score": 80})
    assert sug_empty == []


@patch("llm.client._chat", side_effect=Exception("LLM offline"))
def test_generate_resume_improvements_fallback(mock_chat):
    resume_text = "Software Engineer."
    ats_info = {"missing": ["Kafka", "GraphQL"]}
    scorecard = {"overall_score": 70}

    sug = generate_resume_improvements(resume_text, ats_info, scorecard)
    assert len(sug) > 0
    assert any("Kafka" in s for s in sug)


def test_build_resume_improvements_section_pdf():
    from reportlab.lib.styles import getSampleStyleSheet
    styles = getSampleStyleSheet()

    improvements = ["Add Docker containerization metrics.", "Elaborate on database indexing trade-offs."]
    elements = _build_resume_improvements_section(styles, improvements)
    assert len(elements) >= 3
    texts = [e.text for e in elements if hasattr(e, "text")]
    assert any("Resume Improvement" in t for t in texts)
    assert any("Docker" in t for t in texts)
