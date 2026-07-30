-- memory/schema.sql
-- Exact schema from architecture.md §6
-- Applied by memory/db.py at session init via executescript()

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    role         TEXT,           -- e.g. "Backend Engineer"
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
