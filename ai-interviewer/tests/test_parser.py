"""
tests/test_parser.py
---------------------
Unit tests for llm/parser.py — the P0 risk module (unittest.md §3.1).

All four test cases from unittest.md §3.1 are implemented here.
No live Ollama or model calls — parser is purely deterministic string logic.
"""

from __future__ import annotations

import pytest

from llm.parser import ParseError, parse_question_and_tool_call, parse_score_json


class TestParseQuestionAndToolCall:

    def test_plain_question_without_tool_call(self):
        raw = "What is the difference between TCP and UDP?"
        q, topic = parse_question_and_tool_call(raw)
        assert q == "What is the difference between TCP and UDP?"
        assert topic is None

    def test_question_with_tool_call(self):
        raw = (
            '{"name": "flag_followup_topic", "arguments": {"topic": "Redis Cluster", "reason": "Mentions high availability"}}\n'
            'How do you handle split-brain scenarios in your Redis Cluster?'
        )
        q, topic = parse_question_and_tool_call(raw)
        assert topic == "Redis Cluster"
        assert "How do you handle split-brain scenarios" in q

    def test_malformed_tool_call_graceful_fallback(self):
        raw = "flag_followup_topic invalid json block\nCan you explain your database schema?"
        q, topic = parse_question_and_tool_call(raw)
        assert "Can you explain your database schema?" in q


class TestParseScoreJson:

    def test_parses_clean_json(self):
        """Strategy 1: raw string is already valid JSON."""
        raw = '{"overall_score": 4.0, "dimensions": []}'
        result = parse_score_json(raw)
        assert result["overall_score"] == 4.0
        assert result["dimensions"] == []

    def test_parses_json_wrapped_in_code_fence(self):
        """Strategy 2: JSON is inside ```json … ``` fences."""
        raw = '```json\n{"overall_score": 3.5, "dimensions": []}\n```'
        result = parse_score_json(raw)
        assert result["overall_score"] == 3.5

    def test_parses_json_with_prose_preamble(self):
        """Strategy 3: JSON is preceded by prose text."""
        raw = 'Here is the evaluation:\n{"overall_score": 2.0, "dimensions": []}'
        result = parse_score_json(raw)
        assert result["overall_score"] == 2.0

    def test_raises_or_falls_back_on_unparseable_output(self):
        """
        Strategy 4: completely unparseable.
        With fallback provided → returns fallback cleanly.
        Without fallback → raises ParseError.
        """
        raw = "Sorry, I cannot provide a score right now."

        # With fallback
        fallback = {"overall_score": None, "dimensions": []}
        result = parse_score_json(raw, fallback=fallback)
        assert result["overall_score"] is None  # falls back cleanly, doesn't crash

        # Without fallback → should raise
        with pytest.raises(ParseError):
            parse_score_json(raw)

    def test_parses_json_in_plain_code_fence(self):
        """Code fence without 'json' hint should also be stripped."""
        raw = '```\n{"overall_score": 5.0, "dimensions": []}\n```'
        result = parse_score_json(raw)
        assert result["overall_score"] == 5.0

    def test_parses_full_5_dimension_scorecard(self):
        """Ensure the full scorecard structure from system-design.md §1.6 parses correctly."""
        raw = """{
          "session_id": "abc123",
          "overall_score": 3.8,
          "dimensions": [
            {"name": "technical_depth", "score": 4, "justification": "Good."},
            {"name": "communication_clarity", "score": 4, "justification": "Clear."},
            {"name": "confidence_fluency", "score": 3, "justification": "Some hedging."},
            {"name": "star_completeness", "score": 3, "justification": "Vague result."},
            {"name": "problem_solving", "score": 4, "justification": "Logical."}
          ],
          "summary": "Solid candidate."
        }"""
        result = parse_score_json(raw)
        assert result["session_id"] == "abc123"
        assert result["overall_score"] == 3.8
        assert len(result["dimensions"]) == 5
        assert result["dimensions"][0]["name"] == "technical_depth"

    def test_empty_raw_string_falls_back(self):
        """Empty model output with fallback → returns fallback."""
        result = parse_score_json("", fallback={"overall_score": None, "dimensions": []})
        assert result["overall_score"] is None
