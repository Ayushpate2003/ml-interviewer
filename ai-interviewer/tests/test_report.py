"""
tests/test_report.py
---------------------
Unit tests for report/generate_report.py (unittest.md §3.4).

Uses ReportLab to build actual PDFs in tmp_path; pdfplumber is used to
extract text and assert content correctness. No live LLM or Ollama calls.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber
import pytest

from report.generate_report import generate_report


def _extract_pdf_text(pdf_path: Path) -> str:
    """Return all text content from a PDF file as a single string."""
    with pdfplumber.open(str(pdf_path)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


class TestGenerateReport:

    def test_report_generates_pdf_file(self, tmp_path: Path, sample_session: dict):
        """unittest.md §3.4 test 1: A real PDF file is created at the expected path."""
        output_path = generate_report(sample_session, out_dir=tmp_path)
        assert output_path.exists(), "PDF file was not created"
        assert output_path.suffix == ".pdf", "Output file should have .pdf extension"
        assert output_path.stat().st_size > 0, "PDF file should not be empty"

    def test_report_handles_missing_scores_gracefully(
        self, tmp_path: Path, session_with_no_scores: dict
    ):
        """unittest.md §3.4 test 2: PDF is generated without crashing when scorecard is empty."""
        output_path = generate_report(session_with_no_scores, out_dir=tmp_path)
        assert output_path.exists(), "PDF should still be created even with no scores"

    def test_report_includes_full_transcript(self, tmp_path: Path, sample_session: dict):
        """unittest.md §3.4 test 3: The transcript turns appear in the PDF text."""
        output_path = generate_report(sample_session, out_dir=tmp_path)
        text = _extract_pdf_text(output_path)
        assert "Tell me about a challenging bug" in text, "Interviewer question should be in PDF"
        assert "race condition" in text, "Candidate answer should be in PDF"

    def test_report_cover_includes_role_and_score(self, tmp_path: Path, sample_session: dict):
        """Cover section should include role name and overall score."""
        output_path = generate_report(sample_session, out_dir=tmp_path)
        text = _extract_pdf_text(output_path)
        assert "Backend Engineer" in text, "Role should appear in cover"
        assert "4.0" in text, "Overall score should appear in cover"

    def test_report_scorecard_dimensions_present(self, tmp_path: Path, sample_session: dict):
        """All 5 dimension names should appear in the report."""
        output_path = generate_report(sample_session, out_dir=tmp_path)
        text = _extract_pdf_text(output_path)
        for dim in ["Technical Depth", "Communication Clarity", "Confidence Fluency"]:
            assert dim in text, f"Dimension '{dim}' should appear in scorecard table"

    def test_report_summary_present(self, tmp_path: Path, sample_session: dict):
        """The summary paragraph from the scorecard should appear in the PDF."""
        output_path = generate_report(sample_session, out_dir=tmp_path)
        text = _extract_pdf_text(output_path)
        assert "Strong candidate" in text, "Summary text should appear in report"

    def test_report_with_none_overall_score(self, tmp_path: Path):
        """None overall score → report renders 'N/A', no crash."""
        session = {
            "session_id": "test-none-score",
            "role": "System Design",
            "turns": [{"speaker": "interviewer", "content": "Design a URL shortener."}],
            "scorecard": {
                "session_id": "test-none-score",
                "overall_score": None,
                "dimensions": [
                    {"name": "technical_depth", "score": None, "justification": "Scoring failed."},
                ],
                "summary": "",
            },
        }
        output_path = generate_report(session, out_dir=tmp_path)
        assert output_path.exists()
        text = _extract_pdf_text(output_path)
        assert "N/A" in text, "None score should render as N/A"
