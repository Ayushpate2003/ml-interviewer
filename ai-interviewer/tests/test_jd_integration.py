"""
tests/test_jd_integration.py
----------------------------
Integration and unit tests for Job Description (JD) input, role auto-detection,
intersection questioning mode, custom ATS scoring, and Report tab transparency.
"""

from unittest.mock import MagicMock, patch
import pytest

from app import start_interview, generate_final_report
from llm.client import get_next_question
from llm.prompts import build_system_prompt
from utils.ats import calculate_ats_score
from utils.resume import detect_unified_context_and_role, extract_jd_highlights


@patch("utils.resume._chat", return_value='{"highlights": "Docker, Kubernetes, Terraform", "role": "DevOps / SRE"}')
def test_extract_jd_highlights(mock_chat):
    jd_raw = "We are seeking a Senior DevOps Engineer with Docker, Kubernetes, Terraform, and AWS experience."
    jd_ctx, detected_role, notice = extract_jd_highlights(jd_raw, ["Backend Engineer", "DevOps / SRE"])
    assert "Docker, Kubernetes, Terraform" in jd_ctx
    assert detected_role == "DevOps / SRE"
    assert "DevOps / SRE" in notice


@patch("utils.resume._chat", return_value='{"highlights": "FastAPI, Docker", "role": "DevOps / SRE"}')
def test_detect_unified_context_and_role_both(mock_chat, tmp_path):
    resume_file = tmp_path / "resume.txt"
    resume_file.write_text("Senior Python Backend Developer with FastAPI, PostgreSQL, Redis experience.")
    
    jd_text = "We need a DevOps Engineer proficient in Kubernetes, Terraform, AWS, Docker."
    
    resume_ctx, jd_ctx, detected_role, notice, mode = detect_unified_context_and_role(
        str(resume_file), jd_text, ["Backend Engineer", "DevOps / SRE"]
    )
    assert mode == "both"
    # JD role takes 1st priority over resume role
    assert detected_role == "DevOps / SRE"
    assert "Auto-detected from Job Description" in notice or "DevOps / SRE" in notice


@patch("utils.resume._chat", return_value='{"highlights": "React, Next.js", "role": "Frontend Engineer"}')
def test_detect_unified_context_and_role_jd_only(mock_chat):
    jd_text = "Frontend Engineer needed with React, Next.js, TypeScript."
    resume_ctx, jd_ctx, detected_role, notice, mode = detect_unified_context_and_role(
        None, jd_text, ["Backend Engineer", "Frontend Engineer"]
    )
    assert mode == "jd_only"
    assert detected_role == "Frontend Engineer"
    assert "Frontend Engineer" in notice


def test_detect_unified_context_and_role_neither():
    resume_ctx, jd_ctx, detected_role, notice, mode = detect_unified_context_and_role(
        None, None, ["Backend Engineer"]
    )
    assert mode == "generic"
    assert detected_role is None
    assert notice == ""


def test_build_system_prompt_intersection_mode():
    prompt = build_system_prompt(
        role="Backend Engineer",
        resume_context="Python, FastAPI, SQL",
        jd_context="Docker, Kubernetes, AWS, FastAPI",
    )
    assert "INTERSECTION MODE" in prompt
    assert "RESUME CONTEXT" in prompt
    assert "JOB DESCRIPTION (JD) CONTEXT" in prompt


def test_calculate_ats_score_with_custom_jd():
    resume_text = "Experienced Backend Engineer skilled in Python, SQL, REST APIs, Docker, and PostgreSQL."
    custom_jd = "Looking for a Backend Developer proficient in Python, SQL, Docker, Redis, and Microservices."
    
    res = calculate_ats_score(resume_text, role="Backend Engineer", jd_custom_input=custom_jd)
    assert res["score"] is not None
    assert res["score"] > 0
    assert "provided Job Description" in res["formatted_md"]


def test_calculate_ats_score_suppressed_when_no_resume():
    res = calculate_ats_score(None, role="Backend Engineer", jd_custom_input="Some JD text")
    assert res["score"] is None
    assert res["formatted_md"] == ""


@patch("utils.resume._chat", return_value='{"highlights": "FastAPI, Docker", "role": "Backend Engineer"}')
@patch("app.check_ollama_ready", return_value=(True, ""))
@patch("app.get_next_question", return_value=("How do you use Docker with FastAPI?", None))
@patch("app.speak", return_value=b"fake-audio")
def test_start_interview_both_mode(mock_speak, mock_get_next, mock_check, mock_chat, tmp_path):
    resume_file = tmp_path / "resume.txt"
    resume_file.write_text("Python, FastAPI, Redis developer experience with microservices and PostgreSQL.")
    jd_text = "Senior Developer with FastAPI, Docker, CI/CD"
    
    state, q_text, audio, turn_lbl, setup_err, resume_status, timer_html, tab = start_interview(
        role="Backend Engineer",
        resume_file=str(resume_file),
        jd_input=jd_text,
    )
    assert state["personalization_mode"] == "both"
    assert "Resume & 📋 JD mode enabled" in resume_status["value"]
    assert "Grounded in your resume & Job Description" in q_text


@patch("utils.resume._chat", return_value='{"highlights": "Kubernetes, Terraform", "role": "DevOps / SRE"}')
@patch("app.check_ollama_ready", return_value=(True, ""))
@patch("app.get_next_question", return_value=("Explain Kubernetes Pods.", None))
@patch("app.speak", return_value=b"fake-audio")
def test_start_interview_jd_only_mode(mock_speak, mock_get_next, mock_check, mock_chat):
    jd_text = "DevOps Engineer role requiring Kubernetes, Terraform, Prometheus"
    
    state, q_text, audio, turn_lbl, setup_err, resume_status, timer_html, tab = start_interview(
        role="DevOps / SRE",
        resume_file=None,
        jd_input=jd_text,
    )
    assert state["personalization_mode"] == "jd_only"
    assert state["ats_info"] is None  # ATS omitted for JD-only
    assert "Grounded in target Job Description" in q_text
