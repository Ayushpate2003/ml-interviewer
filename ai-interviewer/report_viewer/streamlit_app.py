"""
report_viewer/streamlit_app.py
-------------------------------
Streamlit companion web application for Privacy-First AI Interviewer.
Displays interactive charts, metric cards, dimension justifications, and full transcripts
reading directly from the local SQLite database (`memory/db.py`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from memory.db import get_all_sessions, get_scores, get_session, get_turns
from report.generator import generate_report

st.set_page_config(
    page_title="Privacy-First AI Interviewer — Visual Reports",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Privacy-First AI Interviewer — Graphical Report Viewer")
st.markdown(
    "Interactive visualization companion for local interview sessions stored in SQLite (`data/interview_sessions.db`)."
)

# Fetch all sessions from SQLite
sessions = get_all_sessions()

if not sessions:
    st.info("ℹ️ **No interview sessions found yet.** Please complete a session in the main Gradio app (`app.py`) to generate a visual report.")
    st.stop()

# Session selector dropdown
session_options = {
    f"{s.get('started_at', '')[:19]} | {s.get('role', 'General')} (ID: {s['session_id'][:8]}…)": s["session_id"]
    for s in sessions
}

selected_label = st.selectbox(
    "Select Interview Session:",
    options=list(session_options.keys()),
    index=0,
)
selected_session_id = session_options[selected_label]

session_meta = get_session(selected_session_id)
scores_list = get_scores(selected_session_id)
turns_list = get_turns(selected_session_id)

st.divider()

# Session Metadata Header
col_meta1, col_meta2, col_meta3 = st.columns(3)
with col_meta1:
    st.subheader(f"Role: `{session_meta.get('role', 'N/A')}`")
with col_meta2:
    st.caption(f"**Session ID:** `{selected_session_id}`")
with col_meta3:
    st.caption(f"**Started At:** `{session_meta.get('started_at', 'N/A')[:19]}`")

if not scores_list:
    st.warning("⚠️ No score data recorded for this session yet.")
else:
    # Calculate overall score
    valid_scores = [s["score"] for s in scores_list if s.get("score") is not None]
    overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    st.metric("Overall Rubric Score", f"{overall_score:.1f} / 5.0")

    # Prepare data for charts
    dim_names = [s.get("dimension", "").replace("_", " ").title() for s in scores_list]
    dim_values = [s.get("score") or 0.0 for s in scores_list]

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("### 🕸️ 5-Dimension Rubric Radar Chart")
        fig_radar = go.Figure(
            data=go.Scatterpolar(
                r=dim_values + [dim_values[0]] if dim_values else [],
                theta=dim_names + [dim_names[0]] if dim_names else [],
                fill="toself",
                name="Score",
                line_color="#4F46E5",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_chart2:
        st.markdown("### 📊 Dimension Scores Bar Chart")
        fig_bar = px.bar(
            x=dim_names,
            y=dim_values,
            labels={"x": "Dimension", "y": "Score (0 - 5)"},
            color=dim_values,
            color_continuous_scale="Viridis",
            range_y=[0, 5.5],
        )
        fig_bar.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed Justifications Section
    st.markdown("### 📝 Dimension Justifications & Feedback")
    for s in scores_list:
        name = s.get("dimension", "").replace("_", " ").title()
        score = s.get("score", "N/A")
        just = s.get("justification", "No justification provided.")
        with st.expander(f"**{name}**: `{score} / 5`", expanded=True):
            st.write(just)

# Full Transcript Section
st.divider()
st.markdown("### 📜 Full Interview Transcript")

if not turns_list:
    st.info("No turns recorded for this session.")
else:
    with st.expander("Show Turn-by-Turn Transcript", expanded=True):
        for idx, turn in enumerate(turns_list, 1):
            speaker = turn.get("speaker", "").capitalize()
            content = turn.get("content", "")
            if speaker == "Interviewer":
                st.markdown(f"🤖 **Interviewer (Turn {idx // 2 + 1}):**")
                st.info(content)
            else:
                st.markdown(f"🗣️ **Candidate:**")
                st.write(content)

# Download PDF Section
st.divider()
st.markdown("### 📄 Export PDF Report")

scorecard_dict = {
    "overall_score": round(overall_score, 1) if scores_list else 0.0,
    "dimensions": scores_list,
    "summary": f"Completed 5-turn session for role: {session_meta.get('role', 'N/A')}",
}

session_data = {
    "session_id": selected_session_id,
    "role": session_meta.get("role", "N/A"),
    "turns": turns_list,
    "scorecard": scorecard_dict,
}

try:
    pdf_path = generate_report(session_data)
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="⬇️ Download Official PDF Report",
            data=pdf_bytes,
            file_name=f"interview_report_{selected_session_id[:8]}.pdf",
            mime="application/pdf",
            type="primary",
        )
except Exception as exc:
    st.warning(f"Could not generate PDF download: {exc}")
