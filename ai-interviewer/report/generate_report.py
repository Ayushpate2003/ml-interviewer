"""
report/generate_report.py
--------------------------
ReportLab PDF report builder (architecture.md §8.6 / system-design.md §1.8).

PDF structure (4 sections):
  1. Cover   — role, date, session ID, overall score.
  2. Scorecard table — dimension / score / one-line justification.
  3. Full transcript appendix — Q&A turn-by-turn.
  4. Summary & suggested next steps (from scorecard["summary"]).

Design decisions:
- Must NOT crash on missing/partial scores (unittest.md §3.4); None scores
  are rendered as "N/A" and missing sections are skipped gracefully.
- Output PDF is written to ``out_dir`` (default: data/ folder); the returned
  Path is passed to Gradio's gr.File download component.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_DEFAULT_OUT_DIR = _HERE.parent / "data"

# ── Colour palette ─────────────────────────────────────────────────────────────
_BRAND_DARK = colors.HexColor("#1B1F3B")    # deep navy — cover background
_BRAND_ACCENT = colors.HexColor("#4F8EF7")  # blue accent — table headers
_SCORE_GREEN = colors.HexColor("#2ECC71")
_SCORE_AMBER = colors.HexColor("#F39C12")
_SCORE_RED = colors.HexColor("#E74C3C")


def _score_colour(score: int | None) -> Any:
    if score is None:
        return colors.grey
    if score >= 4:
        return _SCORE_GREEN
    if score >= 3:
        return _SCORE_AMBER
    return _SCORE_RED


# ── Public API ────────────────────────────────────────────────────────────────

def generate_report(
    session: dict[str, Any],
    out_dir: str | Path = _DEFAULT_OUT_DIR,
) -> Path:
    """
    Build a PDF report for an interview session.

    Parameters
    ----------
    session : dict
        Must contain:
        - ``session_id`` (str)
        - ``role`` (str)
        - ``turns`` (list[dict] — each with ``speaker``, ``content``)
        - ``scorecard`` (dict matching system-design.md §1.6, may have None values)

    out_dir : str | Path
        Directory to write the PDF into.

    Returns
    -------
    Path
        Absolute path to the generated PDF file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session_id = session.get("session_id", "unknown")
    role = session.get("role", "Unknown Role")
    turns = session.get("turns") or []
    scorecard = session.get("scorecard") or {}
    overall_score = scorecard.get("overall_score")
    dimensions = scorecard.get("dimensions") or []
    summary = scorecard.get("summary", "")

    out_path = out_dir / f"report_{session_id[:8]}.pdf"
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Section 1: Cover ──────────────────────────────────────────────────────
    story.extend(_build_cover(styles, role, session_id, overall_score))

    # ── Section 2: Scorecard table ────────────────────────────────────────────
    if dimensions:
        story.extend(_build_scorecard(styles, dimensions))
    else:
        story.append(Paragraph("Scorecard unavailable for this session.", styles["Normal"]))
        story.append(Spacer(1, 0.5 * cm))

    # ── ATS Analysis Section ──────────────────────────────────────────────────
    ats_info = session.get("ats_info")
    if ats_info and ats_info.get("score") is not None:
        story.extend(_build_ats_section(styles, ats_info))

    # ── Section 3: Full transcript ────────────────────────────────────────────
    if turns:
        story.extend(_build_transcript(styles, turns))

    # ── Section 4: Summary & next steps ──────────────────────────────────────
    if summary:
        story.extend(_build_summary(styles, summary))

    doc.build(story)
    logger.info("Report written to %s", out_path)
    return out_path


# ── Section builders ──────────────────────────────────────────────────────────

def _build_ats_section(styles: Any, ats_info: dict) -> list:
    score = ats_info.get("score")
    matched = ", ".join(ats_info.get("matched", [])[:6]) or "None"
    missing = ", ".join(ats_info.get("missing", [])[:6]) or "None"
    suggestions = ats_info.get("suggestions", [""])
    sug_str = suggestions[0] if suggestions else ""

    head_style = ParagraphStyle("ATSHead", parent=styles["Heading2"], textColor=_BRAND_DARK, spaceAfter=6)
    norm_style = styles["Normal"]

    return [
        Paragraph(f"🎯 ATS Resume Match Score: {score}%", head_style),
        Paragraph(f"<b>Matched Keywords:</b> {matched}", norm_style),
        Paragraph(f"<b>Missing Keywords:</b> {missing}", norm_style),
        Paragraph(f"<b>Suggestion:</b> {sug_str}", norm_style),
        Spacer(1, 0.5 * cm),
    ]

def _build_cover(
    styles: Any,
    role: str,
    session_id: str,
    overall_score: float | None,
) -> list:
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontSize=24,
        spaceAfter=12,
        textColor=_BRAND_DARK,
    )
    sub_style = ParagraphStyle(
        "CoverSub",
        parent=styles["Normal"],
        fontSize=12,
        spaceAfter=6,
        textColor=colors.grey,
    )

    score_str = f"{overall_score:.1f} / 5.0" if overall_score is not None else "N/A"
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return [
        Paragraph("🔒 Privacy-First AI Interviewer", title_style),
        Paragraph(f"Role: {role}", sub_style),
        Paragraph(f"Date: {date_str}", sub_style),
        Paragraph(f"Session ID: {session_id}", sub_style),
        Paragraph(f"Overall Score: <b>{score_str}</b>", sub_style),
        Spacer(1, 1 * cm),
    ]


def _build_scorecard(styles: Any, dimensions: list[dict]) -> list:
    heading = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        textColor=_BRAND_DARK,
        spaceAfter=6,
    )

    header_row = ["Dimension", "Score", "Justification"]
    data = [header_row]
    row_colours = []

    for i, dim in enumerate(dimensions, start=1):
        score = dim.get("score")
        score_str = str(score) if score is not None else "N/A"
        justification = dim.get("justification", "")
        data.append([
            dim.get("name", "").replace("_", " ").title(),
            score_str,
            Paragraph(justification, styles["Normal"]) if justification else "",
        ])
        row_colours.append(("BACKGROUND", (1, i), (1, i), _score_colour(score)))

    table = Table(data, colWidths=[5 * cm, 2 * cm, 10 * cm])
    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _BRAND_ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        *row_colours,
    ])
    table.setStyle(ts)

    return [
        Paragraph("Scorecard", heading),
        table,
        Spacer(1, 0.8 * cm),
    ]


def _build_transcript(styles: Any, turns: list[dict]) -> list:
    heading = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        textColor=_BRAND_DARK,
        spaceAfter=6,
    )
    interviewer_style = ParagraphStyle(
        "Interviewer",
        parent=styles["Normal"],
        leftIndent=0,
        spaceAfter=4,
        textColor=_BRAND_DARK,
        fontName="Helvetica-Bold",
    )
    candidate_style = ParagraphStyle(
        "Candidate",
        parent=styles["Normal"],
        leftIndent=1 * cm,
        spaceAfter=8,
    )

    elements = [Paragraph("Full Transcript", heading)]
    for turn in turns:
        speaker = turn.get("speaker", "unknown")
        content = turn.get("content", "")
        if speaker == "interviewer":
            elements.append(Paragraph(f"Interviewer: {content}", interviewer_style))
        else:
            elements.append(Paragraph(f"Candidate: {content}", candidate_style))
    elements.append(Spacer(1, 0.8 * cm))
    return elements


def _build_summary(styles: Any, summary: str) -> list:
    heading = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        textColor=_BRAND_DARK,
        spaceAfter=6,
    )
    return [
        Paragraph("Summary & Next Steps", heading),
        Paragraph(summary, styles["Normal"]),
        Spacer(1, 0.5 * cm),
    ]
