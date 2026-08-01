-- memory/schema.sql
-- Exact schema from architecture.md §6
-- Applied by memory/db.py at session init via executescript()

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    role         TEXT,           -- e.g. "Backend Engineer"
    turns_completed INTEGER DEFAULT 0,
    max_turns    INTEGER DEFAULT 5,
    resume_mode  TEXT DEFAULT 'generic',
    resume_context TEXT DEFAULT '',
    started_at   TEXT,
    ended_at     TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT REFERENCES sessions(session_id),
    speaker      TEXT CHECK(speaker IN ('candidate','interviewer')),
    content      TEXT,
    timestamp    TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    session_id      TEXT REFERENCES sessions(session_id),
    dimension       TEXT,   -- e.g. "technical_depth"
    score           INTEGER,
    justification   TEXT
);

CREATE TABLE IF NOT EXISTS ats_scores (
    session_id       TEXT PRIMARY KEY REFERENCES sessions(session_id),
    score            INTEGER,
    matched_keywords TEXT,
    missing_keywords TEXT,
    suggestions      TEXT
);
