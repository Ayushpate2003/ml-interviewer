"""
tests/test_phase1_role_detect.py
---------------------------------
Unit tests for Phase 1: Resume-driven role/domain auto-detection.
"""

from unittest.mock import patch
import pytest

from utils.resume import detect_resume_role_and_highlights


@patch("utils.resume.get_active_model_tag")
@patch("utils.resume.requests.post")
@patch("utils.resume.extract_text_from_file")
def test_detect_resume_role_backend_engineer(mock_extract_text, mock_post, mock_model_tag):
    mock_extract_text.return_value = "Experienced Python developer with PostgreSQL and microservices background."
    mock_model_tag.return_value = "gemma4:e4b"

    mock_resp = mock_post.return_value
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "message": {
            "content": '{"highlights": "Strong Python and database experience.", "role": "Backend Engineer"}'
        }
    }

    valid_roles = ["Backend Engineer", "HR Round", "System Design"]
    highlights, detected_role, notice = detect_resume_role_and_highlights("dummy.pdf", valid_roles)

    assert detected_role == "Backend Engineer"
    assert "Auto-detected" in notice
    assert "Backend Engineer" in notice


@patch("utils.resume.get_active_model_tag")
@patch("utils.resume.requests.post")
@patch("utils.resume.extract_text_from_file")
def test_detect_resume_role_fallback_on_parse_failure(mock_extract_text, mock_post, mock_model_tag):
    mock_extract_text.return_value = "Random unstructured text without role match."
    mock_model_tag.return_value = "gemma4:e4b"

    mock_resp = mock_post.return_value
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "message": {
            "content": "Not JSON output"
        }
    }

    valid_roles = ["Backend Engineer", "HR Round", "System Design"]
    highlights, detected_role, notice = detect_resume_role_and_highlights("dummy.txt", valid_roles)

    assert detected_role is None
    assert highlights == "Not JSON output"


def test_detect_resume_role_missing_file():
    highlights, detected_role, notice = detect_resume_role_and_highlights(None)
    assert highlights == ""
    assert detected_role is None
    assert notice == ""
