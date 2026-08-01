"""
utils/resume.py
---------------
Resume & Job Description text extraction and Gemma 4 skill highlight extraction.
Extracts top technical skills and project highlights to ground all interview turns.
"""

from __future__ import annotations

import logging
from pathlib import Path
import requests

from llm.client import OLLAMA_BASE_URL, get_active_model_tag

logger = logging.getLogger(__name__)


def extract_text_from_file(file_path: str | Path | dict | Any) -> str:
    """
    Extract raw text from PDF or plain text / markdown file.
    Supports file paths, Path objects, Gradio File objects, and dict wrappers.
    """
    real_path_str = ""
    if isinstance(file_path, (str, Path)):
        real_path_str = str(file_path)
    elif isinstance(file_path, dict):
        real_path_str = file_path.get("name") or file_path.get("path") or ""
    elif hasattr(file_path, "path"):
        real_path_str = str(file_path.path)
    elif hasattr(file_path, "name"):
        real_path_str = str(file_path.name)

    if not real_path_str:
        logger.warning("Empty or invalid file_path provided for resume extraction.")
        return ""

    path = Path(real_path_str)
    if not path.exists():
        logger.warning("Resume file does not exist at path: %s", path)
        return ""

    ext = path.suffix.lower()
    if ext == ".pdf":
        # 1. Try pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            text_parts = [page.extract_text() or "" for page in reader.pages]
            full_text = "\n".join(text_parts).strip()
            if full_text:
                logger.info("Extracted %d characters from PDF resume via pypdf: %s", len(full_text), path.name)
                return full_text
        except Exception as exc:
            logger.warning("Failed PDF extraction via pypdf: %s", exc)

        # 2. Try PyPDF2
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(str(path))
            text_parts = [page.extract_text() or "" for page in reader.pages]
            full_text = "\n".join(text_parts).strip()
            if full_text:
                logger.info("Extracted %d characters from PDF resume via PyPDF2: %s", len(full_text), path.name)
                return full_text
        except Exception as exc:
            logger.warning("Failed PDF extraction via PyPDF2: %s", exc)

        # 3. Try pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                text_parts = [p.extract_text() or "" for p in pdf.pages]
                full_text = "\n".join(text_parts).strip()
                if full_text:
                    logger.info("Extracted %d characters from PDF resume via pdfplumber: %s", len(full_text), path.name)
                    return full_text
        except Exception as exc:
            logger.warning("Failed PDF extraction via pdfplumber: %s", exc)

        logger.warning("All PDF extraction libraries failed for %s", path.name)
        return ""

    elif ext in (".txt", ".md"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        except Exception as exc:
            logger.warning("Failed to read text resume file: %s", exc, exc_info=True)
            print(f"[RESUME PARSING ERROR] Failed to read text file: {exc}")
            return ""
    else:
        logger.warning("Unsupported resume file extension: %s", ext)
        return ""


def detect_resume_role_and_highlights(
    file_path: str | Path | dict | Any,
    valid_roles: list[str] | None = None,
) -> tuple[str, str | None, str]:
    """
    Extract resume highlights and classify the single best matching role from valid_roles.

    Returns
    -------
    tuple[str, str | None, str]
        (highlights_summary, detected_role_or_None, status_notice)
    """
    if not file_path:
        return "", None, ""

    raw_text = extract_text_from_file(file_path)
    if not raw_text or len(raw_text.strip()) < 20:
        logger.warning("Resume file unreadable or contains insufficient text.")
        return "", None, "⚠️ Resume file unreadable — continuing with standard interview role questions."

    # Limit prompt input length to 3000 chars for speed
    truncated_text = raw_text[:3000]
    roles_list = valid_roles or ["Backend Engineer", "HR Round", "System Design"]
    roles_formatted = ", ".join(f'"{r}"' for r in roles_list)

    prompt = (
        "Analyze this candidate resume / job description text.\n"
        "1. List the top 4 key technical skills and 2 major project highlights in 2 concise sentences.\n"
        f"2. Classify the candidate's single best matching role from this EXACT list: [{roles_formatted}].\n\n"
        f"RESUME TEXT:\n{truncated_text}\n\n"
        "Respond ONLY with a JSON object:\n"
        '{"highlights": "<2 sentence summary>", "role": "<exact string from list or null>"}'
    )

    try:
        model_tag = get_active_model_tag()
        payload = {
            "model": model_tag,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 180},
        }

        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=30.0)
        resp.raise_for_status()

        data = resp.json()
        raw_reply = data.get("message", {}).get("content", "").strip()

        highlights = ""
        detected_role = None

        # Parse JSON reply defensively
        try:
            import json
            import re

            match = re.search(r"\{.*\}", raw_reply, re.DOTALL)
            json_str = match.group(0) if match else raw_reply
            parsed = json.loads(json_str)
            highlights = parsed.get("highlights", "").strip()
            role_cand = parsed.get("role", "")
            if role_cand and isinstance(role_cand, str):
                for r in roles_list:
                    if r.lower() == role_cand.strip().lower():
                        detected_role = r
                        break
        except Exception:
            logger.warning("Could not parse JSON role response; attempting direct match.")
            highlights = raw_reply
            for r in roles_list:
                if r.lower() in raw_reply.lower():
                    detected_role = r
                    break

        status_notice = ""
        if detected_role:
            status_notice = f"📄 Auto-detected from your resume: **{detected_role}** — change if needed."
        elif highlights:
            status_notice = f"📄 Resume highlights seeded: {highlights[:100]}…"
        else:
            status_notice = "⚠️ Could not extract resume highlights — continuing with standard questions."

        return highlights, detected_role, status_notice

    except Exception as exc:
        logger.warning("Gemma 4 resume extraction failed: %s", exc, exc_info=True)
        return "", None, f"⚠️ Resume parsing fallback ({exc}): continuing with standard questions."


def extract_resume_highlights(file_path: str | Path | dict | Any) -> tuple[str, str]:
    """
    Backward-compatible wrapper for extract_resume_highlights.
    Returns (highlights_summary, status_notice).
    """
    highlights, _, notice = detect_resume_role_and_highlights(file_path)
    return highlights, notice
