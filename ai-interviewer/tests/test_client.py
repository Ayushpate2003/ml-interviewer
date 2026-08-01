"""
tests/test_client.py
--------------------
Unit tests for llm/client.py — mocked Ollama calls (unittest.md §3.5).

Ollama is NEVER called live in these tests. The requests.post call is mocked
using pytest-mock to verify the payload shape and error propagation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from llm.client import get_next_question, score_session


class MockResponse:
    """Minimal requests.Response mock."""

    def __init__(self, json_data: dict, status_code: int = 200):
        self._json = json_data
        self.status_code = status_code

    def json(self) -> dict:
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _make_chat_response(content: str) -> MockResponse:
    return MockResponse({"message": {"content": content}})


class TestGetNextQuestion:

    def test_client_sends_expected_payload(self):
        """unittest.md §3.5 test 1: Correct model tag and role in payload."""
        history = [{"speaker": "candidate", "content": "I debugged a race condition."}]

        with patch("llm.client.requests.post") as mock_post:
            mock_post.return_value = _make_chat_response("What was the impact of that bug?")
            result, topic = get_next_question(history, role="Backend Engineer")

        assert mock_post.called
        sent_payload = mock_post.call_args[1]["json"]
        assert sent_payload["model"] in ("gemma4:4b", "gemma4:e4b", "gemma4:12b")
        # System message should contain interviewer persona and role-specific context.
        # The prompt embeds the context description text, not the role name literally.
        system_content = sent_payload["messages"][0]["content"]
        assert "expert technical interviewer" in system_content  # persona
        assert "APIs, databases, caching" in system_content      # Backend Engineer context block
        assert isinstance(result, str)
        assert len(result) > 0

    def test_client_raises_clear_error_if_ollama_unreachable(self):
        """unittest.md §3.5 test 2: ConnectionError raised when Ollama is down."""
        with patch("llm.client.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("refused")
            with pytest.raises(ConnectionError, match="Cannot reach Ollama"):
                get_next_question([], role="Backend Engineer")

    def test_client_retries_on_initial_failure_and_succeeds(self):
        """First request fails due to warmup latency, second attempt succeeds."""
        history = [{"speaker": "candidate", "content": "Hello"}]
        with patch("llm.client.requests.post") as mock_post:
            mock_post.side_effect = [
                requests.exceptions.ConnectionError("warmup timeout"),
                _make_chat_response("What is your experience with databases?"),
            ]
            result, topic = get_next_question(history, role="Backend Engineer")

        assert mock_post.call_count == 2
        assert "What is your experience with databases?" in result


class TestScoreSession:

    def test_score_session_returns_parsed_scorecard(self):
        """score_session should parse and return the Gemma 4 scorecard dict."""
        valid_json = """{
          "session_id": "test-123",
          "overall_score": 4.0,
          "dimensions": [
            {"name": "technical_depth", "score": 4, "justification": "Good."},
            {"name": "communication_clarity", "score": 4, "justification": "Clear."},
            {"name": "confidence_fluency", "score": 4, "justification": "Fluent."},
            {"name": "star_completeness", "score": 4, "justification": "Complete."},
            {"name": "problem_solving", "score": 4, "justification": "Logical."}
          ],
          "summary": "Great performance."
        }"""
        history = [
            {"speaker": "interviewer", "content": "Tell me about a bug."},
            {"speaker": "candidate", "content": "I fixed a race condition."},
        ]
        with patch("llm.client.requests.post") as mock_post:
            mock_post.return_value = _make_chat_response(valid_json)
            result = score_session(history, session_id="test-123", role="Backend Engineer")

        assert result["overall_score"] == 4.0
        assert len(result["dimensions"]) == 5

    def test_score_session_retries_on_malformed_json_and_falls_back(self):
        """
        When both scoring attempts return malformed JSON, score_session should
        return the fallback template (not raise) per system-design.md §3.
        """
        history = [{"speaker": "candidate", "content": "Something..."}]
        garbage = "Sorry, I cannot evaluate this right now."

        with patch("llm.client.requests.post") as mock_post:
            # Both the first and the retry return garbage
            mock_post.return_value = _make_chat_response(garbage)
            result = score_session(history, session_id="fallback-test", role="HR Round")

        # Should NOT raise; should return the fallback template
        assert result["overall_score"] is None
        assert len(result["dimensions"]) == 5
        # All dimensions should have None scores
        for dim in result["dimensions"]:
            assert dim["score"] is None
