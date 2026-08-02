"""
tests/test_per_question_feedback.py
-------------------------------------
Unit tests for the per-question detailed feedback pipeline.

Covers:
  - generate_per_question_feedback() with mocked LLM (happy path)
  - generate_per_question_feedback() with malformed JSON (graceful fallback)
  - generate_per_question_feedback() with empty history
  - _render_per_question_html() with valid data
  - _render_per_question_html() with empty list
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_HISTORY = [
    {"speaker": "interviewer", "content": "Explain caching strategies for a high-traffic API."},
    {"speaker": "candidate",   "content": "We can use Redis with TTL-based eviction."},
    {"speaker": "interviewer", "content": "How would you handle cache invalidation?"},
    {"speaker": "candidate",   "content": "[skipped]"},
]

SAMPLE_PER_QUESTION_JSON = {
    "questions": [
        {
            "question_number": 1,
            "question": "Explain caching strategies for a high-traffic API.",
            "candidate_answer_summary": "Candidate mentioned Redis and TTL-based eviction.",
            "overall_score": 5.0,
            "difficulty_level": "Medium",
            "time_taken_hint": "~30s",
            "dimensions": {
                "technical_depth": {
                    "score": 5,
                    "did_well": "Mentioned Redis correctly.",
                    "missing_concepts": "Cache-aside vs write-through not discussed.",
                    "knowledge_gaps": "Eviction policies beyond TTL not covered.",
                    "recommendation": "Study LRU, LFU eviction and compare strategies.",
                },
                "communication_clarity": {
                    "score": 5,
                    "clarity": "Answer was brief but clear.",
                    "structure": "Lacked structured breakdown.",
                    "grammar_vocabulary": "Good vocabulary.",
                    "suggestion": "Use STAR or enumerate options before diving in.",
                },
                "confidence_fluency": {
                    "score": 5,
                    "confidence": "Moderate confidence.",
                    "fluency_pacing": "Pacing was acceptable.",
                    "filler_words": "None notable.",
                    "recommendation": "Expand on reasoning to show deeper confidence.",
                },
                "star_completeness": {
                    "score": 3,
                    "situation": "absent",
                    "task": "absent",
                    "action": "partial – mentioned Redis",
                    "result": "absent",
                    "missing_components": "Situation, Task, Result",
                    "suggestion": "Ground answer in a real-world scenario with measurable outcome.",
                },
                "problem_solving": {
                    "score": 5,
                    "logical_thinking": "Logical but shallow.",
                    "decision_making": "One option considered.",
                    "solution_quality": "Adequate for simple scenarios.",
                    "alternative_approaches": "CDN-level caching, distributed cache, circuit breaker.",
                },
            },
            "strengths": ["Correctly identified Redis as a caching tool."],
            "weaknesses": ["Did not cover cache invalidation strategies."],
            "interviewer_expected": "Cache-aside pattern, TTL, LRU, write-through, invalidation.",
            "how_to_improve": "Practice explaining caching layers from browser to DB.",
            "strong_answer_example": "For a high-traffic API, I would implement a multi-layer caching strategy...",
            "practice_tips": ["Study the Caching chapter in System Design Primer.", "Mock answer with a timer."],
            "readiness_level": "Intermediate",
            "priority_for_improvement": "Medium",
            "estimated_impact": "Could add ~1.5 points to overall score.",
        },
        {
            "question_number": 2,
            "question": "How would you handle cache invalidation?",
            "candidate_answer_summary": "Candidate skipped this question.",
            "overall_score": 1.0,
            "difficulty_level": "Hard",
            "time_taken_hint": "~0s",
            "dimensions": {
                "technical_depth": {"score": 1, "did_well": "N/A", "missing_concepts": "All", "knowledge_gaps": "All", "recommendation": "Study cache invalidation patterns."},
                "communication_clarity": {"score": 1, "clarity": "N/A", "structure": "N/A", "grammar_vocabulary": "N/A", "suggestion": "Answer the question."},
                "confidence_fluency": {"score": 1, "confidence": "N/A", "fluency_pacing": "N/A", "filler_words": "N/A", "recommendation": "Practice under time pressure."},
                "star_completeness": {"score": 1, "situation": "absent", "task": "absent", "action": "absent", "result": "absent", "missing_components": "All", "suggestion": "Use STAR."},
                "problem_solving": {"score": 1, "logical_thinking": "N/A", "decision_making": "N/A", "solution_quality": "N/A", "alternative_approaches": "Study invalidation strategies."},
            },
            "strengths": [],
            "weaknesses": ["Question was skipped."],
            "interviewer_expected": "Event-driven invalidation, TTL, versioned cache keys.",
            "how_to_improve": "Study cache invalidation strategies.",
            "strong_answer_example": "Cache invalidation can be handled via event-driven pub/sub...",
            "practice_tips": ["Read about cache invalidation on Martin Fowler's blog."],
            "readiness_level": "Beginner",
            "priority_for_improvement": "High",
            "estimated_impact": "High impact — this was skipped entirely.",
        },
    ]
}


# ── Tests: generate_per_question_feedback ─────────────────────────────────────

class TestGeneratePerQuestionFeedback:

    def test_happy_path_returns_questions(self):
        """Mocked LLM returns valid JSON → function returns parsed question list."""
        mock_response = json.dumps(SAMPLE_PER_QUESTION_JSON)
        with patch("llm.client._chat", return_value=mock_response):
            from llm.client import generate_per_question_feedback
            result = generate_per_question_feedback(SAMPLE_HISTORY, role="Backend Engineer")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["question_number"] == 1
        assert result[1]["question_number"] == 2
        assert result[0]["overall_score"] == 5.0
        assert "technical_depth" in result[0]["dimensions"]

    def test_malformed_json_returns_fallback(self):
        """Mocked LLM returns invalid JSON → graceful fallback list is returned, no crash."""
        with patch("llm.client._chat", return_value="NOT VALID JSON {{{"):
            from llm.client import generate_per_question_feedback
            result = generate_per_question_feedback(SAMPLE_HISTORY, role="Backend Engineer")

        assert isinstance(result, list)
        # Two interviewer turns → two fallback entries
        assert len(result) == 2
        for item in result:
            assert "question_number" in item
            assert "dimensions" in item
            assert item["overall_score"] == 1.0
            assert item["readiness_level"] == "Beginner"

    def test_empty_history_returns_empty_list(self):
        """Empty history → returns [] without calling LLM."""
        with patch("llm.client._chat") as mock_chat:
            from llm.client import generate_per_question_feedback
            result = generate_per_question_feedback([], role="Backend Engineer")

        assert result == []
        mock_chat.assert_not_called()

    def test_no_interviewer_turns_returns_empty(self):
        """History with only candidate turns → returns [] without calling LLM."""
        history = [
            {"speaker": "candidate", "content": "Hello."},
            {"speaker": "candidate", "content": "I have 3 years of experience."},
        ]
        with patch("llm.client._chat") as mock_chat:
            from llm.client import generate_per_question_feedback
            result = generate_per_question_feedback(history, role="HR Round")

        assert result == []
        mock_chat.assert_not_called()

    def test_skipped_answer_in_fallback(self):
        """Fallback correctly uses interviewer question text for each entry."""
        with patch("llm.client._chat", return_value="{}"):
            from llm.client import generate_per_question_feedback
            result = generate_per_question_feedback(SAMPLE_HISTORY, role="Backend Engineer")

        assert result[0]["question"] == "Explain caching strategies for a high-traffic API."
        assert result[1]["question"] == "How would you handle cache invalidation?"


# ── Tests: _render_per_question_html ─────────────────────────────────────────

class TestRenderPerQuestionHtml:

    def _get_renderer(self):
        """Import the renderer (avoids Gradio import at module level in tests)."""
        import importlib, sys
        # Patch out gradio so app.py can be imported without a running Gradio instance
        import unittest.mock as um
        with um.patch.dict(sys.modules, {"gradio": um.MagicMock()}):
            spec = importlib.util.spec_from_file_location(
                "app", "app.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod._render_per_question_html

    def test_empty_list_returns_fallback_html(self):
        """Empty questions list → fallback HTML string (non-empty, no crash)."""
        renderer = self._get_renderer()
        result = renderer([])
        assert isinstance(result, str)
        assert len(result) > 0
        assert "No per-question feedback" in result

    def test_valid_data_contains_details_tag(self):
        """Valid questions → HTML contains <details> accordion elements."""
        renderer = self._get_renderer()
        result = renderer(SAMPLE_PER_QUESTION_JSON["questions"])
        assert "<details" in result
        assert "Question 1" in result
        assert "Question 2" in result

    def test_score_pills_present(self):
        """Score pills are included in summary line for each question."""
        renderer = self._get_renderer()
        result = renderer(SAMPLE_PER_QUESTION_JSON["questions"])
        assert "Score: 5.0/10" in result
        assert "Score: 1.0/10" in result

    def test_star_breakdown_present(self):
        """STAR section is rendered (look for STAR label in HTML)."""
        renderer = self._get_renderer()
        result = renderer(SAMPLE_PER_QUESTION_JSON["questions"])
        assert "STAR Framework Breakdown" in result
        assert "Situation" in result
        assert "Result" in result

    def test_strong_example_collapsed_by_default(self):
        """Example answer is inside a collapsed <details> tag by default."""
        renderer = self._get_renderer()
        result = renderer(SAMPLE_PER_QUESTION_JSON["questions"])
        assert "Click to expand example answer" in result

    def test_colour_coding_green_for_high_score(self):
        """High-score questions get green colour (#2ECC71) in pills."""
        renderer = self._get_renderer()
        high_score_q = [{**SAMPLE_PER_QUESTION_JSON["questions"][0], "overall_score": 9.0}]
        result = renderer(high_score_q)
        assert "#2ECC71" in result

    def test_colour_coding_red_for_low_score(self):
        """Low-score questions get red colour (#E74C3C) in pills."""
        renderer = self._get_renderer()
        low_score_q = [{**SAMPLE_PER_QUESTION_JSON["questions"][1], "overall_score": 1.0}]
        result = renderer(low_score_q)
        assert "#E74C3C" in result
