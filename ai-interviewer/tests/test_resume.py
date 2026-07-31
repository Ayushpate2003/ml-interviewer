"""
tests/test_resume.py
---------------------
Unit tests for utils/resume.py (Resume text extraction & Gemma 4 seeding).
"""

from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

from utils.resume import extract_resume_highlights, extract_text_from_file


def test_extract_text_from_txt(tmp_path):
    txt_file = tmp_path / "resume.txt"
    txt_file.write_text("Senior Python Developer with 5 years experience in PyTorch and Docker.")

    text = extract_text_from_file(txt_file)
    assert "Senior Python Developer" in text
    assert "PyTorch" in text


def test_extract_text_from_missing_file():
    text = extract_text_from_file("non_existent_file.pdf")
    assert text == ""


@patch("utils.resume.get_active_model_tag")
@patch("utils.resume.requests.post")
def test_extract_resume_highlights_success(mock_post, mock_model, tmp_path):
    mock_model.return_value = "gemma4:12b"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {
            "content": "Key skills: PyTorch, Docker, PostgreSQL. Recent role: Backend Lead at Tech Corp."
        }
    }
    mock_post.return_value = mock_resp

    txt_file = tmp_path / "resume.txt"
    txt_file.write_text("Experienced engineer specializing in high-throughput microservices and PyTorch.")

    highlights, status = extract_resume_highlights(txt_file)
    assert "PyTorch" in highlights
    assert "Resume highlights seeded" in status


def test_extract_resume_highlights_none():
    highlights, status = extract_resume_highlights(None)
    assert highlights == ""
    assert status == ""
