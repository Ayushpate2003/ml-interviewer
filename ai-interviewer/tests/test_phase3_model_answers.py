"""
tests/test_phase3_model_answers.py
-----------------------------------
Unit & integration tests for Phase 3: Model-Answer Comparison.
Verifies model-answer prompt generation, LLM parsing, UI details formatting, and PDF rendering.
"""

from unittest.mock import MagicMock, patch
import pytest

from app import _format_full_transcript_with_model_answers
from llm.client import generate_model_answers
from llm.prompts import build_model_answers_prompt
from report.generate_report import _build_transcript


def test_build_model_answers_prompt():
    history = [
        {"speaker": "interviewer", "content": "Explain B-tree indexes."},
        {"speaker": "candidate", "content": "Indexes speed up queries."},
    ]
    prompt = build_model_answers_prompt(history, role="Backend Engineer")
    assert "Explain B-tree indexes" in prompt
    assert "model_answers" in prompt


@patch("llm.client._chat", return_value='{"model_answers": [{"turn_index": 1, "bullets": ["B-tree balance", "O(log N)", "WAL interaction"]}]}')
def test_generate_model_answers_success(mock_chat):
    history = [
        {"speaker": "interviewer", "content": "Explain B-tree indexes."},
        {"speaker": "candidate", "content": "Indexes speed up queries."},
    ]
    answers = generate_model_answers(history, role="Backend Engineer")
    assert len(answers) == 1
    assert answers[0]["turn_index"] == 1
    assert "B-tree balance" in answers[0]["bullets"][0]


@patch("llm.client._chat", side_effect=Exception("LLM timeout"))
def test_generate_model_answers_fallback(mock_chat):
    history = [
        {"speaker": "interviewer", "content": "What is Python GIL?"},
        {"speaker": "candidate", "content": "Global interpreter lock."},
    ]
    answers = generate_model_answers(history, role="Backend Engineer")
    assert len(answers) == 1
    assert answers[0]["turn_index"] == 1
    assert len(answers[0]["bullets"]) == 3


def test_format_full_transcript_with_model_answers_details_tag():
    history = [
        {"speaker": "interviewer", "content": "What is REST?"},
        {"speaker": "candidate", "content": "REST is an API architectural style."},
    ]
    model_answers = [
        {"turn_index": 1, "bullets": ["Statelessness", "Standard HTTP verbs", "Cacheability"]}
    ]

    md = _format_full_transcript_with_model_answers(history, model_answers)
    assert "<details open><summary>" in md
    assert "What a strong answer should include" in md
    assert "- Statelessness" in md


def test_build_transcript_pdf_includes_model_answers():
    from reportlab.lib.styles import getSampleStyleSheet
    styles = getSampleStyleSheet()

    turns = [
        {"speaker": "interviewer", "content": "What is Docker?"},
        {"speaker": "candidate", "content": "Containerization platform."},
    ]
    model_answers = [
        {"turn_index": 1, "bullets": ["OS-level virtualization", "Dockerfile & layers", "Resource isolation"]}
    ]

    elements = _build_transcript(styles, turns, model_answers=model_answers)
    assert len(elements) > 2
    # Verify Paragraph elements created for model answers
    texts = [e.text for e in elements if hasattr(e, "text")]
    assert any("What a strong answer should include" in t or "Model Answer Highlights" in t for t in texts)
