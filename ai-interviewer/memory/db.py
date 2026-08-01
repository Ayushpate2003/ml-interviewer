"""
memory/db.py
------------
SQLite persistence layer for interview sessions, turns, and scores.

Design decisions (system-design.md §1.4 / architecture.md §6):
- SQLite is the *durability* layer, not the hot path.
- The in-process turn list (a plain Python list in app.py's gr.State) is the
  source of truth for the running prompt; SQLite is written to mirror it so
  the report can be generated after the session.
- DB file defaults to data/interview_sessions.db; overrideable via the `db`
  parameter on each function (enables tmp_path fixtures in tests).
- No connection pooling needed — single-user, single-machine app.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Default DB path ────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_DEFAULT_DB = _HERE.parent / "data" / "interview_sessions.db"

# ── Schema ────────────────────────────────────────────────────────────────────
_SCHEMA_FILE = _HERE / "schema.sql"


def _get_conn(db: str | Path) -> sqlite3.Connection:
    """Open (and create if needed) the SQLite database, applying the schema."""
    db = Path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    # Apply schema (CREATE TABLE IF NOT EXISTS — idempotent)
    schema_sql = _SCHEMA_FILE.read_text()
    conn.executescript(schema_sql)
    _ensure_schema_upgrades(conn)
    conn.commit()
    return conn


def _ensure_schema_upgrades(conn: sqlite3.Connection) -> None:
    """
    Apply additive schema upgrades for existing DB files created before
    newer columns were introduced.
    """
    cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
    }
    if "turns_completed" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN turns_completed INTEGER DEFAULT 0")
    if "max_turns" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN max_turns INTEGER DEFAULT 5")
    if "resume_mode" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN resume_mode TEXT DEFAULT 'generic'")
    if "resume_context" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN resume_context TEXT DEFAULT ''")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ats_scores (
            session_id       TEXT PRIMARY KEY REFERENCES sessions(session_id),
            score            INTEGER,
            matched_keywords TEXT,
            missing_keywords TEXT,
            suggestions      TEXT
        )
        """
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Public CRUD ───────────────────────────────────────────────────────────────

def create_session(
    db: str | Path = _DEFAULT_DB,
    role: str = "",
    *,
    max_turns: int = 5,
    resume_mode: str = "generic",
    resume_context: str = "",
) -> str:
    """
    Create a new interview session row.

    Returns
    -------
    str
        The new ``session_id`` (UUID4 string).
    """
    session_id = str(uuid.uuid4())
    conn = _get_conn(db)
    with conn:
        conn.execute(
            (
                "INSERT INTO sessions "
                "(session_id, role, turns_completed, max_turns, resume_mode, resume_context, started_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            (session_id, role, 0, max_turns, resume_mode, resume_context, _now_iso()),
        )
    conn.close()
    logger.info("Created session %s (role=%s)", session_id, role)
    return session_id


def end_session(session_id: str, db: str | Path = _DEFAULT_DB) -> None:
    """Mark a session as ended."""
    conn = _get_conn(db)
    with conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
            (_now_iso(), session_id),
        )
    conn.close()


def add_turn(
    db: str | Path = _DEFAULT_DB,
    session_id: str = "",
    speaker: str = "candidate",
    content: str = "",
) -> None:
    """
    Append one turn (interviewer or candidate) to the turns table.

    Parameters
    ----------
    speaker : str
        Must be ``'candidate'`` or ``'interviewer'``.
    """
    if speaker not in ("candidate", "interviewer"):
        raise ValueError(f"speaker must be 'candidate' or 'interviewer', got {speaker!r}")

    conn = _get_conn(db)
    with conn:
        conn.execute(
            "INSERT INTO turns (session_id, speaker, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, speaker, content, _now_iso()),
        )
    conn.close()


def get_turns(
    session_id: str, db: str | Path = _DEFAULT_DB
) -> list[dict[str, Any]]:
    """
    Retrieve all turns for a session, ordered by turn_id (insertion order).

    Returns
    -------
    list[dict]
        Each dict has keys: ``turn_id``, ``session_id``, ``speaker``,
        ``content``, ``timestamp``.
    """
    conn = _get_conn(db)
    cur = conn.execute(
        "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_id",
        (session_id,),
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def save_scores(
    db: str | Path = _DEFAULT_DB,
    session_id: str = "",
    scores: list[dict[str, Any]] | None = None,
) -> None:
    """
    Persist a list of dimension scores for a session.

    Parameters
    ----------
    scores : list[dict]
        Each dict must have keys: ``dimension``, ``score``, ``justification``.
    """
    if not scores:
        return

    conn = _get_conn(db)
    with conn:
        conn.executemany(
            "INSERT INTO scores (session_id, dimension, score, justification) VALUES (?, ?, ?, ?)",
            [
                (session_id, s.get("dimension", ""), s.get("score"), s.get("justification", ""))
                for s in scores
            ],
        )
    conn.close()


def get_scores(
    session_id: str, db: str | Path = _DEFAULT_DB
) -> list[dict[str, Any]]:
    """
    Retrieve all dimension scores for a session.

    Returns
    -------
    list[dict]
        Each dict has keys: ``session_id``, ``dimension``, ``score``,
        ``justification``.
    """
    conn = _get_conn(db)
    cur = conn.execute(
        "SELECT * FROM scores WHERE session_id = ?",
        (session_id,),
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_session(
    session_id: str, db: str | Path = _DEFAULT_DB
) -> dict[str, Any] | None:
    """Return session metadata row or None if not found."""
    conn = _get_conn(db)
    cur = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_sessions(db: str | Path = _DEFAULT_DB) -> list[dict[str, Any]]:
    """Retrieve all sessions ordered by started_at DESC (most recent first)."""
    conn = _get_conn(db)
    cur = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def increment_turns_completed(
    session_id: str, db: str | Path = _DEFAULT_DB
) -> int:
    """
    Increment and return the session's turns_completed counter.
    """
    conn = _get_conn(db)
    with conn:
        conn.execute(
            "UPDATE sessions SET turns_completed = COALESCE(turns_completed, 0) + 1 WHERE session_id = ?",
            (session_id,),
        )
        cur = conn.execute(
            "SELECT turns_completed FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
    conn.close()
    return int(row["turns_completed"]) if row else 0


def save_ats_score(
    session_id: str,
    ats_data: dict[str, Any] | None,
    db: str | Path = _DEFAULT_DB,
) -> None:
    """Save or update ATS score analysis in SQLite tied to session_id."""
    if not ats_data or ats_data.get("score") is None:
        return
    import json
    score = ats_data.get("score")
    matched = json.dumps(ats_data.get("matched", []))
    missing = json.dumps(ats_data.get("missing", []))
    suggestions = json.dumps(ats_data.get("suggestions", []))

    conn = _get_conn(db)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO ats_scores (session_id, score, matched_keywords, missing_keywords, suggestions) VALUES (?, ?, ?, ?, ?)",
            (session_id, score, matched, missing, suggestions),
        )
    conn.close()


def get_ats_score(
    session_id: str,
    db: str | Path = _DEFAULT_DB,
) -> dict[str, Any] | None:
    """Retrieve ATS score dict for a session or None if not found."""
    import json
    conn = _get_conn(db)
    cur = conn.execute(
        "SELECT * FROM ats_scores WHERE session_id = ?",
        (session_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None

    try:
        score = row["score"]
        matched = json.loads(row["matched_keywords"]) if row["matched_keywords"] else []
        missing = json.loads(row["missing_keywords"]) if row["missing_keywords"] else []
        suggestions = json.loads(row["suggestions"]) if row["suggestions"] else []

        matched_str = ", ".join(matched[:8]) if matched else "None"
        missing_str = ", ".join(missing[:8]) if missing else "None"
        sug_str = suggestions[0] if suggestions else "No specific suggestions."

        formatted_md = f"""### 🎯 Resume ATS Score: **{score}%** Match
- **Matched Keywords ({len(matched)}):** `{matched_str}`
- **Missing Keywords ({len(missing)}):** `{missing_str}`
- **ATS Suggestion:** {sug_str}"""

        return {
            "score": score,
            "matched": matched,
            "missing": missing,
            "suggestions": suggestions,
            "formatted_md": formatted_md,
        }
    except Exception as exc:
        logger.warning("Error parsing ATS scores row: %s", exc)
        return None
