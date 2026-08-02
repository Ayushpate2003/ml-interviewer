"""
tests/test_phase1_adaptive_difficulty.py
-----------------------------------------
Unit & integration tests for Phase 1: Adaptive Question Difficulty.
Verifies quality signals, single-step difficulty transitions, system prompt rules,
and UI badge rendering.
"""

from unittest.mock import MagicMock, patch
import pytest

from app import _format_question_md, process_answer, start_interview
from llm.client import assess_answer_quality_and_difficulty, get_next_question
from llm.prompts import build_system_prompt


def test_assess_answer_quality_empty_history():
    new_diff, quality, reasoning = assess_answer_quality_and_difficulty([], current_difficulty="Medium")
    assert new_diff == "Medium"
    assert quality == "adequate"


def test_assess_answer_quality_skipped_turn():
    history = [
        {"speaker": "interviewer", "content": "What is Python?"},
        {"speaker": "candidate", "content": "[skipped]"},
    ]
    new_diff, quality, reasoning = assess_answer_quality_and_difficulty(history, current_difficulty="Medium")
    assert new_diff == "Easy"
    assert quality == "weak"


@patch("llm.client._chat", return_value='{"quality": "strong", "reasoning": "Excellent explanation of database indexing."}')
def test_assess_answer_quality_strong_escalation(mock_chat):
    history = [
        {"speaker": "interviewer", "content": "Explain B-tree indexes."},
        {"speaker": "candidate", "content": "B-tree indexes maintain a balanced tree structure with O(log N) lookup complexity..."},
    ]
    new_diff, quality, reasoning = assess_answer_quality_and_difficulty(history, current_difficulty="Medium")
    assert new_diff == "Hard"
    assert quality == "strong"


@patch("llm.client._chat", return_value='{"quality": "strong", "reasoning": "Great answer."}')
def test_assess_answer_quality_capped_at_hard(mock_chat):
    history = [
        {"speaker": "interviewer", "content": "How do you handle split brain in Raft?"},
        {"speaker": "candidate", "content": "Raft uses term numbers and quorum voting..."},
    ]
    new_diff, quality, reasoning = assess_answer_quality_and_difficulty(history, current_difficulty="Hard")
    assert new_diff == "Hard"
    assert quality == "strong"


@patch("llm.client._chat", return_value='{"quality": "weak", "reasoning": "Candidate gave a vague one-word response."}')
def test_assess_answer_quality_weak_deescalation(mock_chat):
    history = [
        {"speaker": "interviewer", "content": "Explain distributed transaction handling."},
        {"speaker": "candidate", "content": "Not sure really."},
    ]
    new_diff, quality, reasoning = assess_answer_quality_and_difficulty(history, current_difficulty="Hard")
    assert new_diff == "Medium"
    assert quality == "weak"


@patch("llm.client._chat", side_effect=Exception("Ollama timeout"))
def test_assess_answer_quality_fallback_on_llm_error(mock_chat):
    history = [
        {"speaker": "interviewer", "content": "Explain REST APIs."},
        {"speaker": "candidate", "content": "REST uses HTTP verbs like GET, POST..."},
    ]
    new_diff, quality, reasoning = assess_answer_quality_and_difficulty(history, current_difficulty="Medium")
    assert new_diff == "Medium"
    assert quality == "adequate"


def test_build_system_prompt_difficulty_instructions():
    prompt_hard = build_system_prompt("Backend Engineer", difficulty="Hard")
    assert "ADAPTIVE QUESTION DIFFICULTY: HARD" in prompt_hard
    assert "Escalate question difficulty" in prompt_hard

    prompt_easy = build_system_prompt("Backend Engineer", difficulty="Easy")
    assert "ADAPTIVE QUESTION DIFFICULTY: EASY" in prompt_easy
    assert "Calibrate difficulty to core fundamentals" in prompt_easy


def test_format_question_md_difficulty_badges():
    md_hard = _format_question_md("What is Paxos?", difficulty="Hard")
    assert "Escalated Depth (Advanced Probe)" in md_hard

    md_easy = _format_question_md("What is an array?", difficulty="Easy")
    assert "Calibrated Core Focus" in md_easy

    md_med = _format_question_md("Explain REST.", difficulty="Medium")
    assert "Standard Interview Depth" in md_med
