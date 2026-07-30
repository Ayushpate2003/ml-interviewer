"""
tests/test_db.py
-----------------
Unit tests for memory/db.py (unittest.md §3.3).

Uses pytest's tmp_path fixture via the tmp_db conftest fixture to ensure
each test runs against a fresh, isolated SQLite DB — no shared state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memory.db import (
    add_turn,
    create_session,
    get_scores,
    get_session,
    get_turns,
    save_scores,
)


class TestSessionAndTurns:

    def test_create_session_and_append_turns(self, tmp_db: Path):
        """unittest.md §3.3 test 1: Create session, append turns, verify retrieval."""
        session_id = create_session(tmp_db, role="Backend Engineer")
        assert session_id  # non-empty UUID string

        add_turn(tmp_db, session_id, speaker="interviewer", content="Tell me about a challenging bug.")
        add_turn(tmp_db, session_id, speaker="candidate", content="I once debugged a race condition...")

        turns = get_turns(session_id, tmp_db)
        assert len(turns) == 2
        assert turns[0]["speaker"] == "interviewer"
        assert turns[0]["content"] == "Tell me about a challenging bug."
        assert turns[1]["speaker"] == "candidate"

    def test_turns_ordered_by_insertion(self, tmp_db: Path):
        """Turns must come back in insertion order (turn_id ASC)."""
        session_id = create_session(tmp_db, role="HR Round")
        for i in range(5):
            speaker = "interviewer" if i % 2 == 0 else "candidate"
            add_turn(tmp_db, session_id, speaker=speaker, content=f"Message {i}")

        turns = get_turns(session_id, tmp_db)
        contents = [t["content"] for t in turns]
        assert contents == [f"Message {i}" for i in range(5)]

    def test_invalid_speaker_raises(self, tmp_db: Path):
        """Speaker must be 'candidate' or 'interviewer'."""
        session_id = create_session(tmp_db, role="Test")
        with pytest.raises(ValueError, match="speaker must be"):
            add_turn(tmp_db, session_id, speaker="robot", content="Beep boop")

    def test_get_turns_empty_for_new_session(self, tmp_db: Path):
        """Fresh session with no turns → empty list, no crash."""
        session_id = create_session(tmp_db, role="System Design")
        turns = get_turns(session_id, tmp_db)
        assert turns == []


class TestScores:

    def test_scores_persist_and_retrieve(self, tmp_db: Path):
        """unittest.md §3.3 test 2: Scores persisted and correctly retrieved."""
        session_id = create_session(tmp_db, role="HR Round")
        save_scores(
            tmp_db,
            session_id,
            [{"dimension": "communication_clarity", "score": 4, "justification": "Clear"}],
        )
        scores = get_scores(session_id, tmp_db)
        assert len(scores) == 1
        assert scores[0]["score"] == 4
        assert scores[0]["dimension"] == "communication_clarity"
        assert scores[0]["justification"] == "Clear"

    def test_save_all_five_dimensions(self, tmp_db: Path):
        """All 5 rubric dimensions should be storable and retrievable."""
        session_id = create_session(tmp_db, role="Backend Engineer")
        dimensions = [
            {"dimension": "technical_depth", "score": 4, "justification": "Deep."},
            {"dimension": "communication_clarity", "score": 3, "justification": "OK."},
            {"dimension": "confidence_fluency", "score": 3, "justification": "Some hedging."},
            {"dimension": "star_completeness", "score": 2, "justification": "Vague."},
            {"dimension": "problem_solving", "score": 5, "justification": "Excellent."},
        ]
        save_scores(tmp_db, session_id, dimensions)
        scores = get_scores(session_id, tmp_db)
        assert len(scores) == 5

    def test_get_scores_empty_when_none_saved(self, tmp_db: Path):
        """No scores saved → returns empty list, no crash."""
        session_id = create_session(tmp_db, role="System Design")
        scores = get_scores(session_id, tmp_db)
        assert scores == []

    def test_save_scores_with_none_values_graceful(self, tmp_db: Path):
        """save_scores with None list → should not crash."""
        session_id = create_session(tmp_db, role="Test")
        save_scores(tmp_db, session_id, None)  # should be a no-op
        assert get_scores(session_id, tmp_db) == []

    def test_session_metadata_retrievable(self, tmp_db: Path):
        """get_session should return the session row with the role."""
        session_id = create_session(tmp_db, role="HR Round")
        session = get_session(session_id, tmp_db)
        assert session is not None
        assert session["role"] == "HR Round"
        assert session["session_id"] == session_id
