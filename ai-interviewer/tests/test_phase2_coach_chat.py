"""
tests/test_phase2_coach_chat.py
--------------------------------
Unit & integration tests for Phase 2: Post-interview AI Coach Chat.
Verifies grounding in session transcript & scorecard, scope restriction, and error fallback.
"""

from unittest.mock import MagicMock, patch
import pytest

from llm.client import ask_ai_coach
from llm.prompts import build_coach_chat_prompt


def test_build_coach_chat_prompt_embeds_context():
    history = [
        {"speaker": "interviewer", "content": "What is Python GIL?"},
        {"speaker": "candidate", "content": "The Global Interpreter Lock limits execution to one thread at a time in CPython."},
    ]
    scorecard = {
        "overall_score": 85,
        "dimensions": [
            {"name": "Technical Depth", "score": 9, "justification": "Clear explanation of GIL mechanics."},
            {"name": "Communication Clarity", "score": 8, "justification": "Concise and well structured."},
        ],
    }

    prompt = build_coach_chat_prompt(history, scorecard)
    assert "Global Interpreter Lock" in prompt
    assert "Technical Depth" in prompt
    assert "POLITELY REDIRECT" in prompt


@patch("llm.client._chat", return_value="You scored 8/10 on Communication Clarity because your explanation of GIL was concise, but you could have elaborated on multiprocessing as an alternative.")
def test_ask_ai_coach_valid_query(mock_chat):
    history = [
        {"speaker": "interviewer", "content": "What is Python GIL?"},
        {"speaker": "candidate", "content": "It limits threads in CPython."},
    ]
    scorecard = {"overall_score": 80}
    coach_history = []
    user_query = "Why did I score 8 on Communication Clarity?"

    res = ask_ai_coach(history, scorecard, coach_history, user_query)
    assert "Communication Clarity" in res
    assert mock_chat.called


@patch("llm.client._chat", return_value="I am here to help you review your mock interview session! Let's stay focused on your performance and scores.")
def test_ask_ai_coach_off_topic_redirection(mock_chat):
    history = [{"speaker": "interviewer", "content": "Hi"}, {"speaker": "candidate", "content": "Hello"}]
    scorecard = {"overall_score": 75}
    user_query = "What is the capital of France?"

    res = ask_ai_coach(history, scorecard, [], user_query)
    assert "mock interview" in res or "focused" in res or "interview" in res


@patch("llm.client._chat", side_effect=Exception("Ollama disconnected"))
def test_ask_ai_coach_error_fallback(mock_chat):
    history = []
    scorecard = {}
    res = ask_ai_coach(history, scorecard, [], "How did I do?")
    assert "unavailable" in res or "Error" in res
