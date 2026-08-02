"""
utils/ats.py
------------
Offline ATS (Applicant Tracking System) resume analyzer.
Evaluates resume text against role-specific keyword criteria using 100% local processing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROLE_KEYWORDS = {
    "Backend Engineer": [
        "python", "java", "golang", "c++", "sql", "postgresql", "mysql", "redis",
        "mongodb", "rest", "api", "microservices", "docker", "kubernetes", "git",
        "ci/cd", "unit testing", "system design", "distributed systems", "kafka",
        "rabbitmq", "aws", "gcp"
    ],
    "Frontend Engineer": [
        "react", "next.js", "vue", "angular", "typescript", "javascript", "css", "html",
        "html5", "css3", "tailwind", "sass", "redux", "zustand", "web performance",
        "vite", "webpack", "responsive design", "accessibility", "a11y", "rest api",
        "graphql", "dom", "browser security", "cross-browser"
    ],
    "DevOps / SRE": [
        "docker", "kubernetes", "terraform", "ansible", "ci/cd", "github actions",
        "jenkins", "prometheus", "grafana", "linux", "bash", "python", "helm",
        "aws", "gcp", "azure", "observability", "incident management",
        "infrastructure as code", "iac", "site reliability", "sre"
    ],
    "Cloud Computing": [
        "aws", "gcp", "azure", "terraform", "iam", "vpc", "ec2", "s3", "lambda",
        "serverless", "cloudformation", "kubernetes", "docker", "cloud security",
        "load balancing", "cost optimization", "multi-region", "disaster recovery",
        "cloud architecture", "cloud migration"
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


def calculate_ats_score(
    resume_file_or_text: Any,
    role: str = "Backend Engineer",
    jd_custom_input: Any = None,
) -> dict:
    """
    Calculate ATS match percentage and keyword breakdown locally and offline
    against generic role JD or a provided custom Job Description.
    """
    if not resume_file_or_text:
        return {
            "score": None,
            "matched": [],
            "missing": [],
            "suggestions": [],
            "formatted_md": "",
        }

    from utils.resume import extract_text_from_file, extract_text_from_file_or_string

    resume_text = ""
    file_path_str = None

    if isinstance(resume_file_or_text, (str, Path)):
        s_val = str(resume_file_or_text)
        if len(s_val) > 250 or "\n" in s_val:
            resume_text = s_val.strip()
        else:
            try:
                p = Path(s_val)
                if p.exists() and p.is_file():
                    file_path_str = str(p)
                    resume_text = extract_text_from_file(p)
                else:
                    resume_text = s_val.strip()
            except OSError:
                resume_text = s_val.strip()
    elif hasattr(resume_file_or_text, "name"):
        try:
            p = Path(str(resume_file_or_text.name))
            if p.exists() and p.is_file():
                file_path_str = str(p)
                resume_text = extract_text_from_file(p)
        except OSError:
            pass

    if not resume_text and not file_path_str:
        resume_text = extract_text_from_file(resume_file_or_text)

    if not resume_text or len(resume_text.strip()) < 20:
        return {
            "score": None,
            "matched": [],
            "missing": [],
            "suggestions": [],
            "formatted_md": "⚠️ ATS scoring unavailable (insufficient resume text)",
        }

    custom_jd_text = extract_text_from_file_or_string(jd_custom_input) if jd_custom_input else ""

    role_jd_map = {
        "Backend Engineer": "Backend Engineer proficient in Python, Java, SQL, PostgreSQL, REST APIs, Microservices, Docker, Kubernetes, System Design, Caching, Git, CI/CD, Distributed Systems.",
        "Frontend Engineer": "Frontend Engineer proficient in React, Next.js, TypeScript, JavaScript, CSS, HTML5, State Management, Web Performance, Responsive Design, Accessibility, REST APIs, GraphQL.",
        "DevOps / SRE": "DevOps Engineer / SRE proficient in Docker, Kubernetes, Terraform, CI/CD pipelines, Linux, Prometheus, Grafana, Ansible, AWS, Infrastructure as Code, Observability, Incident Management.",
        "Cloud Computing": "Cloud Architect / Computing Engineer proficient in AWS, GCP, Azure, Serverless Lambda, Terraform, VPC Networking, Cloud Security, IAM Policies, EC2, S3, Multi-region High Availability.",
        "System Design": "System Design Architect specializing in Scalability, Load Balancing, Caching, Sharding, Microservices, Message Queues, Fault Tolerance, Latency, Throughput, Database Indexing.",
        "HR Round": "HR Round candidate demonstrating Leadership, Teamwork, Communication, Conflict Resolution, Problem Solving, Adaptability, Collaboration, Agile Project Management.",
    }
    target_jd_text = custom_jd_text if custom_jd_text else role_jd_map.get(role, f"{role} role requiring domain expertise and technical skills.")
    target_label = "provided Job Description" if custom_jd_text else f"*{role}*"

    try:
        if file_path_str:
            import ats_resume_checker
            res = ats_resume_checker.analyze_resume(file_path_str, target_jd_text)
            score = int(round(res.get("ats_score", 0)))
            matched = res.get("matched_keywords", [])
            missing = res.get("missing_keywords", [])
            suggestions = res.get("suggestions", [])
        else:
            return _calculate_ats_score_fallback(resume_text, role, custom_jd_text=custom_jd_text)
    except Exception as exc:
        logger.warning("ats-resume-checker failed: %s; using local fallback matcher", exc)
        return _calculate_ats_score_fallback(resume_text, role, custom_jd_text=custom_jd_text)

    matched_str = ", ".join(matched[:8]) if matched else "None"
    missing_str = ", ".join(missing[:8]) if missing else "None"
    sug_str = suggestions[0] if suggestions else "No specific suggestions."

    formatted_md = f"""### 🎯 Resume ATS Score: **{score}%** Match for {target_label}
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


def _calculate_ats_score_fallback(resume_text: str, role: str, custom_jd_text: str = "") -> dict:
    if not resume_text or len(resume_text.strip()) < 20:
        return {
            "score": None,
            "matched": [],
            "missing": [],
            "suggestions": [],
            "formatted_md": "⚠️ ATS scoring unavailable",
        }

    if custom_jd_text and len(custom_jd_text.strip()) >= 20:
        # Extract keywords from custom JD text
        tokens = re.findall(r"\b[a-zA-Z0-9\+\#\.\-]{3,}\b", custom_jd_text.lower())
        stopwords = {"the", "and", "for", "with", "that", "this", "from", "you", "will", "our", "are", "have", "has", "must", "work", "team", "role", "ability", "experience", "strong", "looking", "years"}
        jd_keywords = sorted(list({t for t in tokens if t not in stopwords and len(t) >= 3}))[:20]
        keywords = jd_keywords if jd_keywords else ROLE_KEYWORDS.get(role, ROLE_KEYWORDS["Backend Engineer"])
        target_label = "provided Job Description"
    else:
        keywords = ROLE_KEYWORDS.get(role, ROLE_KEYWORDS["Backend Engineer"])
        target_label = f"*{role}*"

    text_lower = resume_text.lower()

    matched = [kw for kw in keywords if kw.lower() in text_lower]
    missing = [kw for kw in keywords if kw.lower() not in text_lower]

    total = len(keywords)
    score = round((len(matched) / total) * 100) if total > 0 else 0

    suggestions = []
    if missing:
        top_missing = ", ".join(missing[:4])
        suggestions.append(f"Consider highlighting skills like {top_missing} if applicable.")
    if score < 70:
        suggestions.append("Incorporate specific tools, frameworks, and metrics from the job posting to improve ATS parsing.")
    else:
        suggestions.append("Strong alignment with core job requirements!")

    matched_str = ", ".join(matched[:8]) if matched else "None"
    missing_str = ", ".join(missing[:8]) if missing else "None"

    formatted_md = f"""### 🎯 Resume ATS Score: **{score}%** Match for {target_label}
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
