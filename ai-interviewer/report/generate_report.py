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
    per_question_feedback = session.get("per_question_feedback") or []

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

    # ── Section 3: Per-question feedback ───────────────────────────────────
    if per_question_feedback:
        story.extend(_build_per_question_section(styles, per_question_feedback))

    # ── ATS Analysis Section ──────────────────────────────────────────────────
    ats_info = session.get("ats_info")
    if ats_info and ats_info.get("score") is not None:
        story.extend(_build_ats_section(styles, ats_info))

    resume_improvements = session.get("resume_improvements")
    if resume_improvements:
        story.extend(_build_resume_improvements_section(styles, resume_improvements))

    # ── Section 4: Full transcript ────────────────────────────────────────────
    model_answers = session.get("model_answers")
    if turns:
        story.extend(_build_transcript(styles, turns, model_answers=model_answers))

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


def _build_resume_improvements_section(styles: Any, improvements: list[str]) -> list:
    if not improvements:
        return []
    head_style = ParagraphStyle("ResImpHead", parent=styles["Heading2"], textColor=_BRAND_DARK, spaceAfter=6)
    bullet_style = ParagraphStyle("ResImpBullet", parent=styles["Normal"], leftIndent=0.5 * cm, spaceAfter=4)

    elements = [
        Paragraph("📄 Resume Improvement & Rewrite Suggestions", head_style),
        Paragraph("Synthesized from ATS keyword analysis and live interview technical performance:", styles["Normal"]),
        Spacer(1, 0.2 * cm),
    ]
    for imp in improvements:
        elements.append(Paragraph(f"• {imp}", bullet_style))
    elements.append(Spacer(1, 0.5 * cm))
    return elements

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


def _build_transcript(styles: Any, turns: list[dict], model_answers: list[dict] | None = None) -> list:
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
    model_head_style = ParagraphStyle(
        "ModelHead",
        parent=styles["Normal"],
        leftIndent=0.5 * cm,
        spaceAfter=2,
        textColor=_BRAND_ACCENT,
        fontName="Helvetica-Bold",
        fontSize=9,
    )
    model_bullet_style = ParagraphStyle(
        "ModelBullet",
        parent=styles["Normal"],
        leftIndent=1 * cm,
        spaceAfter=2,
        fontSize=9,
        textColor=colors.HexColor("#374151"),
    )

    model_map = {}
    if model_answers:
        for item in model_answers:
            t_idx = item.get("turn_index")
            if t_idx:
                model_map[t_idx] = item.get("bullets", [])

    elements = [Paragraph("Full Transcript & Model Answers", heading)]
    interviewer_count = 0
    for turn in turns:
        speaker = turn.get("speaker", "unknown")
        content = turn.get("content", "")
        if speaker == "interviewer":
            interviewer_count += 1
            elements.append(Paragraph(f"Interviewer: {content}", interviewer_style))
            bullets = turn.get("model_answer_bullets") or model_map.get(interviewer_count, [])
            if bullets:
                elements.append(Paragraph("💡 Model Answer Highlights:", model_head_style))
                for bullet in bullets:
                    elements.append(Paragraph(f"• {bullet}", model_bullet_style))
                elements.append(Spacer(1, 0.2 * cm))
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


def _build_per_question_section(styles: Any, questions: list[dict]) -> list:
    """
    Build a PDF section with per-question detailed feedback.

    Each question gets:
    - A heading with question number, overall score, difficulty, and readiness level.
    - A 5-row dimension score table.
    - Strengths, weaknesses, how-to-improve, and practice tips.
    """
    heading_style = ParagraphStyle(
        "PQHead", parent=styles["Heading2"], textColor=_BRAND_DARK, spaceAfter=6
    )
    q_head_style = ParagraphStyle(
        "PQQHead", parent=styles["Heading3"], textColor=_BRAND_ACCENT, spaceAfter=4
    )
    norm_style = styles["Normal"]
    bullet_style = ParagraphStyle(
        "PQBullet", parent=styles["Normal"], leftIndent=0.5 * cm, spaceAfter=3, fontSize=9
    )
    label_style = ParagraphStyle(
        "PQLabel", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9, spaceAfter=2
    )

    _DIM_DISPLAY = [
        ("technical_depth",       "Technical Depth"),
        ("communication_clarity", "Communication Clarity"),
        ("confidence_fluency",    "Confidence & Fluency"),
        ("star_completeness",     "STAR Completeness"),
        ("problem_solving",       "Problem Solving"),
    ]

    elements: list = [Paragraph("Per-Question Detailed Feedback", heading_style)]

    for q in questions:
        q_num = q.get("question_number", "?")
        overall = q.get("overall_score", 1)
        difficulty = q.get("difficulty_level", "Medium")
        readiness = q.get("readiness_level", "Beginner")
        priority = q.get("priority_for_improvement", "High")
        q_text = (q.get("question") or "")[:200]
        ans_summary = q.get("candidate_answer_summary", "")
        dims = q.get("dimensions", {})
        strengths = q.get("strengths", [])
        weaknesses = q.get("weaknesses", [])
        how_to_improve = q.get("how_to_improve", "")
        practice_tips = q.get("practice_tips", [])
        impact = q.get("estimated_impact", "")

        header_text = (
            f"Q{q_num} | Score: {overall}/10 | {difficulty} | {readiness} | Priority: {priority}"
        )
        elements.append(Paragraph(header_text, q_head_style))
        elements.append(Paragraph(f"<b>Question:</b> {q_text}", norm_style))
        elements.append(Spacer(1, 0.15 * cm))
        elements.append(Paragraph(f"<b>Candidate Answer:</b> {ans_summary}", norm_style))
        elements.append(Spacer(1, 0.2 * cm))

        # Dimension score mini-table
        dim_header = ["Dimension", "Score", "Key Recommendation"]
        dim_data = [dim_header]
        dim_row_colours = []
        for i, (key, label) in enumerate(_DIM_DISPLAY, start=1):
            d = dims.get(key, {})
            score = d.get("score", 1)
            rec = d.get("recommendation") or d.get("suggestion") or d.get("alternative_approaches", "")
            dim_data.append([label, str(score), Paragraph(rec[:120], norm_style) if rec else ""])
            # Colour score on 1-10 scale (map to 5-pt colours: ≥8→green, ≥5→amber, <5→red)
            colour = _score_colour(round(score / 2) if score else 1)
            dim_row_colours.append(("BACKGROUND", (1, i), (1, i), colour))

        dim_table = Table(dim_data, colWidths=[4 * cm, 1.5 * cm, 11.5 * cm])
        dim_ts = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _BRAND_ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            *dim_row_colours,
        ])
        dim_table.setStyle(dim_ts)
        elements.append(dim_table)
        elements.append(Spacer(1, 0.2 * cm))

        if strengths:
            elements.append(Paragraph("Strengths:", label_style))
            for s in strengths:
                elements.append(Paragraph(f"• {s}", bullet_style))
        if weaknesses:
            elements.append(Paragraph("Weaknesses:", label_style))
            for w in weaknesses:
                elements.append(Paragraph(f"• {w}", bullet_style))
        if how_to_improve:
            elements.append(Paragraph("How to Improve:", label_style))
            elements.append(Paragraph(how_to_improve[:300], bullet_style))
        if practice_tips:
            elements.append(Paragraph("Practice Tips:", label_style))
            for tip in practice_tips:
                elements.append(Paragraph(f"• {tip}", bullet_style))
        if impact:
            elements.append(Paragraph(f"Estimated Impact: {impact}", bullet_style))

        elements.append(Spacer(1, 0.5 * cm))

    return elements
