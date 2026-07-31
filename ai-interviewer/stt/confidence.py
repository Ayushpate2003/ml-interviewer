"""
stt/confidence.py
------------------
Lightweight transcript heuristic scanner for real-time live session feedback.
Scans filler words ("um", "uh", "like") and hedging phrases ("I guess", "sort of").
Calculates frequency per 100 words and maps to a visual confidence gauge badge.

NOTE: This is a live-session UX signal only. It is explicitly separate from
the final Gemma 4 LLM rubric evaluation.
"""

from __future__ import annotations

import re

FILLER_WORDS = {"um", "uh", "er", "ah", "like"}
HEDGING_PHRASES = [
    "i guess",
    "sort of",
    "kind of",
    "maybe",
    "i think",
    "pretty much",
    "somewhat",
    "probably",
]


def analyze_transcript_fluency(transcript: str) -> tuple[float, str, str]:
    """
    Analyze transcript for filler words and hedging frequency per 100 words.

    Parameters
    ----------
    transcript : str
        Candidate answer transcript text.

    Returns
    -------
    tuple[float, str, str]
        (filler_rate_per_100w, level_name, gauge_badge_md)
    """
    if not transcript or not transcript.strip():
        return 0.0, "Steady", "📊 **Fluency Signal:** 🟡 Steady (No answer text yet)"

    text_lower = transcript.lower()

    # Clean words for word count
    words = re.findall(r"\b[a-z']+\b", text_lower)
    word_count = len(words)
    if word_count == 0:
        return 0.0, "Steady", "📊 **Fluency Signal:** 🟡 Steady (Empty transcript)"

    # Count filler words
    filler_count = sum(1 for w in words if w in FILLER_WORDS)

    # Count hedging phrases
    hedging_count = sum(text_lower.count(phrase) for phrase in HEDGING_PHRASES)

    total_occurrences = filler_count + hedging_count
    rate_per_100w = (total_occurrences / float(word_count)) * 100.0

    if rate_per_100w <= 3.5:
        level = "Confident"
        badge = f"📊 **Fluency Signal:** 🟢 **Confident** (Low fillers/hedging: {rate_per_100w:.1f}/100w)"
    elif rate_per_100w <= 8.5:
        level = "Steady"
        badge = f"📊 **Fluency Signal:** 🟡 **Steady** (Moderate fillers/hedging: {rate_per_100w:.1f}/100w)"
    else:
        level = "Hesitant"
        badge = f"📊 **Fluency Signal:** 🟠 **Hesitant** (Frequent fillers/hedging: {rate_per_100w:.1f}/100w)"

    return rate_per_100w, level, badge
