"""
tests/conftest.py
-----------------
Shared pytest fixtures for the AI Interviewer test suite.
"""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

import pytest


# ── WAV fixture helpers ───────────────────────────────────────────────────────

def _make_wav_bytes(num_frames: int = 8000, sample_rate: int = 16000, amplitude: int = 0) -> bytes:
    """
    Generate a synthetic WAV file as bytes.

    amplitude=0 → silence (used for silence tests).
    amplitude>0 → a simple sine-like signal (used for non-empty audio tests).
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        if amplitude == 0:
            frames = b"\x00\x00" * num_frames
        else:
            import math
            freq = 440  # Hz
            frames = b"".join(
                struct.pack("<h", int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate)))
                for i in range(num_frames)
            )
        wf.writeframes(frames)
    return buf.getvalue()


@pytest.fixture
def silent_audio_bytes() -> bytes:
    """Near-silent WAV audio (all zero samples, 0.5 s)."""
    return _make_wav_bytes(num_frames=8000, amplitude=0)


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """
    Short synthetic WAV with audible signal.
    Note: faster-whisper will transcribe this as something (or empty string on
    minimal signal) — tests only assert len > 0 on a *real* fixture; for unit
    tests we mock the model call instead.
    """
    return _make_wav_bytes(num_frames=32000, amplitude=10000)


# ── DB fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Return a path to a fresh, empty SQLite DB inside tmp_path."""
    return tmp_path / "test_sessions.db"


# ── Report fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_session() -> dict:
    """A complete session dict with turns and a valid scorecard."""
    return {
        "session_id": "test-session-001",
        "role": "Backend Engineer",
        "turns": [
            {"speaker": "interviewer", "content": "Tell me about a challenging bug."},
            {"speaker": "candidate", "content": "I once debugged a race condition in a distributed lock."},
            {"speaker": "interviewer", "content": "How did you identify the root cause?"},
            {"speaker": "candidate", "content": "I added detailed trace logging and replicated the issue under load."},
        ],
        "scorecard": {
            "session_id": "test-session-001",
            "overall_score": 4.0,
            "dimensions": [
                {"name": "technical_depth", "score": 4, "justification": "Good depth on concurrency."},
                {"name": "communication_clarity", "score": 4, "justification": "Clear and structured."},
                {"name": "confidence_fluency", "score": 4, "justification": "Minimal filler words."},
                {"name": "star_completeness", "score": 4, "justification": "STAR format mostly followed."},
                {"name": "problem_solving", "score": 4, "justification": "Logical debugging approach."},
            ],
            "summary": "Strong candidate with solid debugging skills.",
        },
    }


@pytest.fixture
def session_with_no_scores() -> dict:
    """A session dict with no scorecard dimensions (tests graceful PDF handling)."""
    return {
        "session_id": "test-session-002",
        "role": "HR Round",
        "turns": [
            {"speaker": "interviewer", "content": "Tell me about yourself."},
            {"speaker": "candidate", "content": "I am a software engineer with 3 years experience."},
        ],
        "scorecard": {
            "session_id": "test-session-002",
            "overall_score": None,
            "dimensions": [],
            "summary": "",
        },
    }
