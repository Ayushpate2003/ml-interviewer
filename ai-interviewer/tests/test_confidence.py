"""
tests/test_confidence.py
-------------------------
Unit tests for stt/confidence.py (Live transcript fluency signal).
"""

from stt.confidence import analyze_transcript_fluency


def test_confident_clean_transcript():
    clean_text = (
        "I designed a distributed rate limiter using Redis token bucket algorithm. "
        "We deployed it on Kubernetes with high availability and benchmarked 5000 requests per second."
    )
    rate, level, badge = analyze_transcript_fluency(clean_text)
    assert level == "Confident"
    assert "🟢" in badge
    assert rate <= 3.5


def test_hesitant_filler_heavy_transcript():
    filler_text = (
        "Um, like, I guess we used Redis, uh, sort of for caching, but maybe, like, "
        "um, I think it was kind of, uh, difficult to configure."
    )
    rate, level, badge = analyze_transcript_fluency(filler_text)
    assert level == "Hesitant"
    assert "🟠" in badge
    assert rate > 8.5


def test_empty_transcript():
    rate, level, badge = analyze_transcript_fluency("")
    assert level == "Steady"
    assert rate == 0.0
