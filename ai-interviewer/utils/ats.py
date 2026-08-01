"""
utils/ats.py
------------
Offline ATS (Applicant Tracking System) resume analyzer.
Evaluates resume text against role-specific keyword criteria using 100% local processing.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

ROLE_KEYWORDS = {
    "Backend Engineer": [
        "python", "java", "golang", "c++", "sql", "postgresql", "mysql", "redis",
        "mongodb", "rest", "api", "microservices", "docker", "kubernetes", "git",
        "ci/cd", "unit testing", "system design", "distributed systems", "kafka",
        "rabbitmq", "aws", "gcp"
    ],
    "System Design": [
        "scalability", "load balancing", "caching", "sharding", "replication",
        "microservices", "message queues", "kafka", "redis", "cdn",
        "database indexing", "nosql", "sql", "fault tolerance", "high availability",
        "latency", "throughput", "consistency", "cap theorem", "api gateway"
    ],
    "HR Round": [
        "leadership", "teamwork", "communication", "conflict resolution", "problem solving",
        "adaptability", "time management", "collaboration", "project management",
        "mentorship", "agile", "scrum", "initiative", "stakeholder management", "work ethic"
    ],
}


def calculate_ats_score(resume_text: str | None, role: str = "Backend Engineer") -> dict:
    """
    Calculate ATS match percentage and keyword breakdown locally and offline.

    Returns
    -------
    dict
        {
            "score": int | None,
            "matched": list[str],
            "missing": list[str],
            "suggestions": list[str],
            "formatted_md": str
        }
    """
    if not resume_text or len(resume_text.strip()) < 20:
        return {
            "score": None,
            "matched": [],
            "missing": [],
            "suggestions": [],
            "formatted_md": "",
        }

    keywords = ROLE_KEYWORDS.get(role, ROLE_KEYWORDS["Backend Engineer"])
    text_lower = resume_text.lower()

    matched = []
    missing = []

    for kw in keywords:
        if kw.lower() in text_lower:
            matched.append(kw)
        else:
            missing.append(kw)

    total = len(keywords)
    score = round((len(matched) / total) * 100) if total > 0 else 0

    suggestions = []
    if missing:
        top_missing = ", ".join(missing[:4])
        suggestions.append(f"Consider highlighting skills like {top_missing} if applicable.")
    if score < 70:
        suggestions.append("Incorporate specific tools, frameworks, and system metrics to improve ATS parsing.")
    else:
        suggestions.append("Strong alignment with core role keywords!")

    matched_str = ", ".join(matched[:8]) if matched else "None"
    missing_str = ", ".join(missing[:8]) if missing else "None"

    formatted_md = f"""### 🎯 ATS Resume Score: **{score}%** Match for *{role}*
- **Matched Keywords ({len(matched)}):** `{matched_str}`
- **Missing Keywords ({len(missing)}):** `{missing_str}`
- **ATS Suggestion:** {suggestions[0]}"""

    return {
        "score": score,
        "matched": matched,
        "missing": missing,
        "suggestions": suggestions,
        "formatted_md": formatted_md,
    }
