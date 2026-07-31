"""
utils/resume.py
---------------
Resume & Job Description text extraction and Gemma 4 skill highlight extraction.
Extracts top technical skills and project highlights to seed Question 1 & 2.
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
        try:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            text_parts = [page.extract_text() or "" for page in reader.pages]
            full_text = "\n".join(text_parts).strip()
            logger.info("Extracted %d characters from PDF resume: %s", len(full_text), path.name)
            return full_text
        except Exception as exc:
            logger.warning("Failed to extract PDF text via pypdf: %s", exc, exc_info=True)
            print(f"[RESUME PARSING ERROR] Failed to extract PDF text: {exc}")
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


def extract_resume_highlights(file_path: str | Path | dict | Any) -> tuple[str, str]:
    """
    Extract resume highlights using Gemma 4.

    Returns
    -------
    tuple[str, str]
        (highlights_summary, status_notice)
    """
    if not file_path:
        return "", ""

    raw_text = extract_text_from_file(file_path)
    if not raw_text or len(raw_text.strip()) < 20:
        logger.warning("Resume file unreadable or contains insufficient text.")
        return "", "⚠️ Resume file unreadable — continuing with standard interview role questions."

    # Limit prompt input length to 3000 chars for speed
    truncated_text = raw_text[:3000]

    prompt = (
        "Analyze this candidate resume / job description text and list the top 4 key technical skills "
        "and 2 major project highlights in 2 concise sentences.\n\n"
        f"RESUME TEXT:\n{truncated_text}\n\n"
        "Concise Highlights:"
    )

    try:
        model_tag = get_active_model_tag()
        payload = {
            "model": model_tag,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 128},
        }

        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=30.0)
        resp.raise_for_status()

        data = resp.json()
        highlights = data.get("message", {}).get("content", "").strip()

        if highlights:
            logger.info("Extracted Gemma 4 resume highlights successfully: %s", highlights)
            print(f"[RESUME PARSED SUCCESSFULLY] {highlights}")
            return highlights, f"📄 Resume highlights seeded: {highlights[:100]}…"
        else:
            return "", "⚠️ Could not extract highlights — continuing with standard questions."

    except Exception as exc:
        logger.warning("Gemma 4 resume highlight extraction failed: %s", exc, exc_info=True)
        print(f"[RESUME EXTRACTION ERROR] Gemma 4 call failed: {exc}")
        return "", f"⚠️ Resume parsing fallback ({exc}): continuing with standard questions."
