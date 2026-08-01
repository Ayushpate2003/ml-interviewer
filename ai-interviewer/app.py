"""
app.py — Privacy-First AI Interviewer
======================================
Gradio entrypoint. Assembles the 3-screen UI per userflow.md §2.

Screen 1 — Setup
  Role/domain dropdown, mic check button, offline badge.
Screen 2 — Live Interview
  Question text + TTS audio, Record/Stop, live transcript, turn counter, Skip.
Screen 3 — Report
  Scorecard table, expandable full transcript, PDF download, New Session.

Edge cases (userflow.md §3):
  - Silence / empty transcript → "I didn't catch that" prompt, no LLM call.
  - Skip → [skipped] placeholder added to history.
  - Mic permission denied → troubleshooting hint shown.
  - Long answers → no truncation; Gemma 4 context window handles it.
  - Refresh → new session.

Startup check (system-design.md §3):
  Ollama reachability and model availability are verified before the UI loads.
  A clear error banner is shown if either check fails — never mid-session.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

import gradio as gr

from llm.client import check_ollama_ready, get_next_question, score_session
from memory.db import add_turn, create_session, end_session, increment_turns_completed, save_scores
from report.generate_report import generate_report
from stt.confidence import analyze_transcript_fluency
from stt.gemma_audio import transcribe_native_gemma
from stt.transcribe import TranscriptionError, load_stt_model, transcribe
from stt.vad import check_end_of_speech, load_vad_model
from tts.speak import TTSError, load_tts_model, speak
from utils.ats import calculate_ats_score
from utils.resume import detect_resume_role_and_highlights, extract_resume_highlights, extract_text_from_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
MAX_TURNS = 5
DB_PATH = Path(__file__).parent / "data" / "interview_sessions.db"

ROLES = ["Backend Engineer", "HR Round", "System Design"]

# ── Module init (loaded once at startup) ─────────────────────────────────────
_STT_READY = False
_TTS_READY = False
_OLLAMA_ERROR = ""


def _init_models():
    global _STT_READY, _TTS_READY, _OLLAMA_ERROR

    # STT
    try:
        load_stt_model()
        _STT_READY = True
    except Exception as exc:
        logger.error("STT model load failed: %s", exc)

    # TTS
    try:
        load_tts_model()
        _TTS_READY = True
    except Exception as exc:
        logger.warning("TTS model load failed (will degrade gracefully): %s", exc)

    # Ollama / Gemma 4
    ok, err = check_ollama_ready()
    if not ok:
        _OLLAMA_ERROR = err
        logger.error("Ollama check failed: %s", err)


_init_models()

# ── Session state helpers ──────────────────────────────────────────────────────
def _new_state(
    role: str,
    gemma_audio_all_turns: bool = False,
    max_turns: int = MAX_TURNS,
    timer_seconds: int = 90,
    *,
    resume_mode: str = "generic",
    resume_context: str = "",
) -> dict:
    """Create a fresh session state dict stored in gr.State."""
    session_id = create_session(
        DB_PATH,
        role=role,
        max_turns=max_turns,
        resume_mode=resume_mode,
        resume_context=resume_context,
    )
    return {
        "session_id": session_id,
        "role": role,
        "history": [],          # list of {speaker, content}
        "turn_index": 0,
        "turns_completed": 0,
        "max_turns": max_turns,
        "timer_seconds": timer_seconds,
        "finished": False,
        "gemma_audio_all_turns": gemma_audio_all_turns,
        "resume_mode": resume_mode,
        "resume_context": resume_context,
        "pending_transcript": "",
    }


def _add_to_history(state: dict, speaker: str, content: str) -> None:
    """Append to in-memory history and mirror to SQLite."""
    state["history"].append({"speaker": speaker, "content": content})
    add_turn(DB_PATH, session_id=state["session_id"], speaker=speaker, content=content)


def _format_question_md(q: str, topic: str | None = None, is_resume_anchored: bool = False) -> str:
    """Format question text, resume badge, and optional probing chip for display in the Markdown component."""
    if not q or not q.strip():
        return "<div class='warning-box'>⚠️ <strong>No question generated.</strong> Please check if Ollama is running.</div>"
    if q.startswith("⚠️") or q.startswith("Error") or "Error" in q:
        return f"<div class='warning-box'>⚠️ <strong>Interviewer Error:</strong> {q}</div>"
    if q.startswith("###") or q.startswith("Session"):
        return q

    badges = []
    if is_resume_anchored:
        badges.append("📄 **Based on your resume**")
    if topic:
        badges.append(f"🔍 **Probing deeper on:** `{topic}`")

    badge_md = ("\n\n".join(badges) + "\n\n") if badges else ""
    return f"## 💬 Interviewer's Question\n\n{badge_md}### **{q}**"


def _question_audio_update(audio: bytes | None):
    """Only show the player after TTS has produced playable question audio."""
    return gr.update(value=audio, visible=audio is not None)


def _normalize_audio_input(audio_input: Any) -> bytes | str | None:
    """
    Normalize Gradio audio payloads to bytes or filepath.
    Supports filepath strings, bytes, dict wrappers, and numpy tuple payloads.
    """
    if audio_input is None:
        return None
    if isinstance(audio_input, (bytes, str)):
        return audio_input
    if isinstance(audio_input, dict):
        maybe_path = audio_input.get("path") or audio_input.get("name")
        maybe_bytes = audio_input.get("bytes")
        if isinstance(maybe_bytes, bytes):
            return maybe_bytes
        if isinstance(maybe_path, str):
            return maybe_path
        return None
    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        try:
            import io
            import numpy as np
            import soundfile as sf

            sample_rate, samples = audio_input
            arr = np.asarray(samples)
            if arr.size == 0:
                return b""
            buf = io.BytesIO()
            sf.write(buf, arr, int(sample_rate), format="WAV")
            return buf.getvalue()
        except Exception:
            logger.exception("Failed to normalize tuple audio payload")
            return None
    return None


def _build_timer_html(timer_seconds: int = 90) -> str:
    sec = int(timer_seconds) if timer_seconds else 90
    return f"""<div id="timer-display" style="padding:8px 14px; border-radius:6px; background:#1e293b; color:#10b981; font-weight:bold; font-size:15px; border:1px solid #334155; display:inline-block; margin-bottom:10px;">
    ⏱️ Time Remaining: <span id="timer-counter">{sec}</span>s / {sec}s
</div>
<script>
(function() {{
    if (window._qTimer) clearInterval(window._qTimer);
    var total = {sec};
    var remaining = total;
    var display = document.getElementById("timer-counter");
    var box = document.getElementById("timer-display");
    if (!display || !box) return;

    function playTone(freq, duration) {{
        try {{
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.frequency.value = freq;
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            gain.gain.exponentialRampToValueAtTime(0.00001, ctx.currentTime + duration);
            osc.stop(ctx.currentTime + duration);
        }} catch(e) {{}}
    }}

    window._qTimer = setInterval(function() {{
        remaining--;
        if (display) display.innerText = Math.max(0, remaining);
        var pct = ((total - remaining) / total) * 100;
        if (pct >= 90) {{
            if (box) {{ box.style.color = "#ef4444"; box.style.borderColor = "#ef4444"; }}
            if (remaining === Math.floor(total * 0.10)) playTone(880, 0.3);
        }} else if (pct >= 66) {{
            if (box) {{ box.style.color = "#f59e0b"; box.style.borderColor = "#f59e0b"; }}
            if (remaining === Math.floor(total * 0.34)) playTone(660, 0.2);
        }}
        if (remaining <= 0) {{
            clearInterval(window._qTimer);
            if (box) {{
                box.innerHTML = "⏱️ <strong>Time's up — wrap up when ready</strong>";
                box.style.color = "#ef4444";
                box.style.borderColor = "#ef4444";
            }}
        }}
    }}, 1000);
}})();
</script>"""


def start_interview(
    role: str,
    gemma_audio_all_turns: bool = False,
    resume_file: str | None = None,
    num_questions: int = 5,
    timer_seconds: int = 90,
) -> tuple:
    """
    Initialise a new session and generate the first question.
    Returns: (state, question_text, audio_bytes, turn_label, setup_error_update, resume_status_update, timer_html, tab_update)
    """
    ok, err = check_ollama_ready()
    if not ok:
        logger.error("start_interview failed health check: %s", err)
        err_html = (
            f'<div class="warning-box">'
            f'⚠️ <strong>Ollama Not Detected / Model Unavailable:</strong> {err}<br>'
            f'Please run <code>ollama serve</code> and <code>ollama pull gemma4:12b</code> in a terminal, then refresh.'
            f'</div>'
        )
        return (
            None,
            _format_question_md(f"⚠️ Cannot start: {err}"),
            _question_audio_update(None),
            "",
            gr.update(value=err_html, visible=True),
            gr.update(visible=False),
            "",
            gr.Tabs(selected="setup"),
        )

    try:
        max_turns = int(num_questions) if num_questions else 5
        t_sec = int(timer_seconds) if timer_seconds else 90
        resume_context = ""
        resume_mode = "generic"
        if resume_file:
            resume_context, _ = extract_resume_highlights(resume_file)
            if resume_context.strip():
                resume_mode = "resume"

        state = _new_state(
            role,
            gemma_audio_all_turns,
            max_turns=max_turns,
            timer_seconds=t_sec,
            resume_mode=resume_mode,
            resume_context=resume_context,
        )

        if resume_file:
            raw_txt = extract_text_from_file(resume_file)
            state["ats_info"] = calculate_ats_score(raw_txt, role)

        if resume_mode == "resume":
            resume_status = f"📄 Resume mode enabled. Questions will be grounded in your resume context for all {max_turns} turns."
        else:
            resume_status = f"⚪ Generic mode (no resume detected). Role-based questions will be used for all {max_turns} turns."

        question, topic = get_next_question(
            [],
            role,
            resume_context=resume_context,
            time_allotted_seconds=t_sec,
            current_turn=1,
            total_turns=max_turns,
        )
        _add_to_history(state, "interviewer", question)

        audio = None
        if _TTS_READY:
            try:
                audio = speak(question)
            except Exception as exc:
                logger.warning("TTS failed on first question: %s", exc)

        turn_label = f"Question 1 of {max_turns}"
        timer_html = _build_timer_html(t_sec)
        resume_status_update = gr.update(value=resume_status, visible=bool(resume_status))
        is_anchored = state.get("resume_mode") == "resume"
        return (
            state,
            _format_question_md(question, topic, is_resume_anchored=is_anchored),
            _question_audio_update(audio),
            turn_label,
            gr.update(visible=False),
            resume_status_update,
            timer_html,
            gr.Tabs(selected="interview"),
        )
    except Exception as exc:
        logger.exception("start_interview error: %s", exc)
        err_html = (
            f'<div class="warning-box">'
            f'⚠️ <strong>Error starting interview:</strong> {exc}<br>'
            f'Please check if Ollama is running.'
            f'</div>'
        )
        return (
            None,
            _format_question_md(f"⚠️ Error starting interview: {exc}"),
            _question_audio_update(None),
            "Error",
            gr.update(value=err_html, visible=True),
            gr.update(visible=False),
            "",
            gr.Tabs(selected="setup"),
        )


def _transcribe_candidate_audio(audio_input: Any, state: dict) -> tuple[str, str]:
    """Transcribe answer audio with Gemma-native-first strategy and fallback."""
    payload = _normalize_audio_input(audio_input)
    if not payload:
        return "", "🎙️ STT Engine: No audio detected"

    use_gemma_native = state.get("gemma_audio_all_turns", False) or (state.get("turn_index", 0) == 0)

    if use_gemma_native:
        try:
            return transcribe_native_gemma(payload)
        except Exception:
            logger.exception("Gemma native audio failed; trying faster-whisper fallback")
            try:
                return transcribe(payload), "⚡ faster-whisper (fallback)"
            except Exception:
                logger.exception("Fallback faster-whisper transcription failed")
                return "", "⚠️ STT Engine: Error (see logs)"

    try:
        return transcribe(payload), "⚡ STT Engine: faster-whisper"
    except TranscriptionError:
        logger.exception("faster-whisper transcription failed")
        return "", "⚠️ STT Engine: Error (see logs)"


def process_recording_stop(audio_input: Any, state: dict):
    """
    Stop-recording entry point for transcript + live analysis.
    Yields:
      1. Immediate unified loading indicator ("⏳ Analyzing your answer...").
      2. Final atomic grouped outputs (state, transcript, stt_badge, fluency_badge, vad_status, submit_error, analysis_status_update).
    """
    if state is None or state.get("finished"):
        yield (
            state,
            "",
            "⚡ STT Engine: Standby",
            "📊 Fluency Signal: Standby",
            "🎙️ **VAD Status:** Listening...",
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
        )
        return

    normalized = _normalize_audio_input(audio_input)
    if not normalized:
        yield (
            state,
            "",
            "⚡ STT Engine: Standby",
            "📊 Fluency Signal: Standby",
            "🎙️ **VAD Status:** Listening...",
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
        )
        return

    # Step 1: Yield immediate unified loading state
    yield (
        state,
        "",
        "⚡ STT Engine: Processing...",
        "📊 Fluency Signal: Analyzing...",
        "🎙️ **VAD Status:** Analyzing...",
        gr.update(value="", visible=False),
        gr.update(value="⏳ **Analyzing your answer...**", visible=True),
    )

    # Step 2: Run pipeline with error isolation
    try:
        vad_msg = "🎙️ Listening..."
        if normalized:
            try:
                _, vad_msg = check_end_of_speech(normalized)
            except Exception:
                logger.exception("VAD analysis failed; continuing without blocking transcript")
                vad_msg = "🎙️ Listening (VAD unavailable)"

        transcript, stt_badge = _transcribe_candidate_audio(audio_input, state)

        try:
            try:
                _, _, fluency_badge = analyze_transcript_fluency(transcript)
            except Exception:
                logger.exception("Fluency analysis failed during submit; continuing with transcript")
                fluency_badge = "📊 **Fluency Signal:** ⚪ Unavailable"
        except Exception:
            logger.exception("Fluency analysis failed; returning transcript anyway")
            fluency_badge = "📊 **Fluency Signal:** ⚪ Unavailable"

        state["pending_transcript"] = transcript.strip()
        if transcript.strip():
            submit_error = gr.update(value="", visible=False)
        else:
            submit_error = gr.update(
                value="⚠️ No transcript captured yet. Please record again before submitting.",
                visible=True,
            )

        yield (
            state,
            transcript,
            stt_badge,
            fluency_badge,
            f"🎙️ **VAD Status:** {vad_msg}",
            submit_error,
            gr.update(value="", visible=False),
        )
    except Exception:
        logger.exception("Stop-recording pipeline failed with unhandled exception")
        yield (
            state,
            "",
            "⚠️ STT Engine: Error (see logs)",
            "📊 **Fluency Signal:** ⚪ Unavailable",
            "🎙️ **VAD Status:** Error",
            gr.update(
                value="⚠️ Recording processing failed unexpectedly. Please try recording again.",
                visible=True,
            ),
            gr.update(value="", visible=False),
        )


def process_answer(transcript_input: str | None, state: dict) -> tuple:
    """
    Submit one transcript and return the next question.
    Returns: (state, transcript_text, stt_badge, fluency_badge, question_text, audio_bytes, turn_label, finish_flag)
    """
    if state is None or state.get("finished"):
        return state, "", "⚡ STT Engine: Standby", "📊 Fluency Signal: Standby", "", _question_audio_update(None), "", True

    max_turns = state.get("max_turns", MAX_TURNS)
    transcript = (transcript_input or state.get("pending_transcript") or "").strip()
    _, _, fluency_badge = analyze_transcript_fluency(transcript)
    stt_badge = "⚡ STT Engine: Ready"

    if not transcript:
        last_q = state["history"][-1]["content"] if state["history"] else ""
        is_anchored = state.get("resume_mode") == "resume"
        return (
            state,
            "",
            stt_badge,
            fluency_badge,
            _format_question_md(last_q, is_resume_anchored=is_anchored),
            _question_audio_update(None),
            f"Question {state['turn_index'] + 1} of {max_turns}",
            False,
        )

    _add_to_history(state, "candidate", transcript)
    state["pending_transcript"] = ""
    state["turns_completed"] = increment_turns_completed(state["session_id"], DB_PATH)
    state["turn_index"] = state["turns_completed"]

    if state["turn_index"] >= max_turns:
        state["finished"] = True
        return (
            state,
            transcript,
            stt_badge,
            fluency_badge,
            "### Session complete — generating your report…",
            _question_audio_update(None),
            f"Question {state['turn_index']} of {max_turns}",
            True,
        )

    try:
        resume_ctx = state.get("resume_context")
        t_sec = state.get("timer_seconds", 90)
        cur_turn = state["turn_index"] + 1
        question, topic = get_next_question(
            state["history"],
            state["role"],
            resume_context=resume_ctx,
            time_allotted_seconds=t_sec,
            current_turn=cur_turn,
            total_turns=max_turns,
        )
        _add_to_history(state, "interviewer", question)
    except Exception as exc:
        logger.error("LLM call failed in process_answer: %s", exc)
        question, topic = f"⚠️ LLM Error: {exc}. Please check if Ollama is running.", None

    audio = None
    if _TTS_READY and not question.startswith("⚠️"):
        try:
            audio = speak(question)
        except Exception as exc:
            logger.warning("TTS failed on turn %d: %s", state["turn_index"], exc)

    turn_label = f"Question {state['turn_index'] + 1} of {max_turns}"
    is_anchored = state.get("resume_mode") == "resume"
    return state, transcript, stt_badge, fluency_badge, _format_question_md(question, topic, is_resume_anchored=is_anchored), _question_audio_update(audio), turn_label, False


def skip_question(state: dict) -> tuple:
    """Add [skipped] placeholder and advance to the next question."""
    if state is None or state.get("finished"):
        return state, "", "", _question_audio_update(None), "", True

    max_turns = (state or {}).get("max_turns", MAX_TURNS)
    _add_to_history(state, "candidate", "[skipped]")
    state["turns_completed"] = increment_turns_completed(state["session_id"], DB_PATH)
    state["turn_index"] = state["turns_completed"]

    if state["turn_index"] >= max_turns:
        state["finished"] = True
        return state, "[skipped]", "Session complete — generating your report…", _question_audio_update(None), f"Question {state['turn_index']} of {max_turns}", True

    try:
        resume_ctx = state.get("resume_context")
        t_sec = state.get("timer_seconds", 90)
        cur_turn = state["turn_index"] + 1
        question, topic = get_next_question(
            state["history"],
            state["role"],
            resume_context=resume_ctx,
            time_allotted_seconds=t_sec,
            current_turn=cur_turn,
            total_turns=max_turns,
        )
        _add_to_history(state, "interviewer", question)
    except Exception as exc:
        logger.error("LLM call failed in skip_question: %s", exc)
        question, topic = f"⚠️ LLM Error: {exc}. Please check if Ollama is running.", None

    audio = None
    if _TTS_READY and not question.startswith("⚠️"):
        try:
            audio = speak(question)
        except Exception as exc:
            logger.warning("TTS failed: %s", exc)

    turn_label = f"Question {state['turn_index'] + 1} of {max_turns}"
    is_anchored = state.get("resume_mode") == "resume"
    return state, "[skipped]", _format_question_md(question, topic, is_resume_anchored=is_anchored), _question_audio_update(audio), turn_label, False


def generate_final_report(state: dict) -> tuple:
    """Score the session with Gemma 4 and build the PDF report."""
    if state is None:
        return None, "No active session.", None, None, None, gr.update()

    end_session(state["session_id"], DB_PATH)

    scorecard = score_session(state["history"], state["session_id"], state["role"])
    save_scores(DB_PATH, state["session_id"], scorecard.get("dimensions", []))

    session_data = {
        "session_id": state["session_id"],
        "role": state["role"],
        "turns": state["history"],
        "scorecard": scorecard,
    }
    pdf_path = generate_report(session_data)

    # Build a markdown scorecard summary and Plotly figures
    summary_md = _format_scorecard_md(scorecard)
    fig_radar, fig_bar = _create_report_charts(scorecard)

    return state, summary_md, fig_radar, fig_bar, str(pdf_path), gr.Tabs(selected="report")


def _create_report_charts(scorecard: dict):
    if not scorecard:
        return None, None

    dims = scorecard.get("dimensions") or []
    if not dims:
        return None, None

    try:
        dim_names = [
            str(d.get("name") or d.get("dimension") or "").replace("_", " ").title()
            for d in dims
        ]
        dim_scores = [float(d.get("score") or 0.0) for d in dims]

        import plotly.express as px
        import plotly.graph_objects as go

        # Radar chart
        fig_radar = go.Figure(
            data=go.Scatterpolar(
                r=dim_scores + [dim_scores[0]],
                theta=dim_names + [dim_names[0]],
                fill="toself",
                name="Rubric Score",
                line_color="#6366F1",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=False,
            margin=dict(l=30, r=30, t=30, b=30),
            title="5-Dimension Rubric Radar Chart",
        )

        # Bar chart
        fig_bar = px.bar(
            x=dim_names,
            y=dim_scores,
            labels={"x": "Dimension", "y": "Score (0-5)"},
            title="Rubric Dimension Breakdown",
            color=dim_scores,
            color_continuous_scale="Viridis",
            range_y=[0, 5.5],
        )
        fig_bar.update_layout(margin=dict(l=30, r=30, t=30, b=30))

        return fig_radar, fig_bar
    except Exception as exc:
        logger.warning("Failed to render Plotly report charts (degrading gracefully to table): %s", exc)
        return None, None


def _format_scorecard_md(scorecard: dict) -> str:
    lines = [f"### Overall Score: {scorecard.get('overall_score', 'N/A')} / 5.0\n"]
    lines.append("| Dimension | Score | Justification |")
    lines.append("|---|---|---|")
    for dim in scorecard.get("dimensions") or []:
        name = dim.get("name", "").replace("_", " ").title()
        score = dim.get("score", "N/A")
        justification = dim.get("justification", "")
        lines.append(f"| {name} | {score} | {justification} |")
    if scorecard.get("summary"):
        lines.append(f"\n**Summary:** {scorecard['summary']}")
    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_CSS = """
    .offline-badge {
        background: linear-gradient(135deg, #1B1F3B, #2d3561);
        color: #4ff0b2;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: bold;
        text-align: center;
        font-size: 14px;
        letter-spacing: 0.5px;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        border-radius: 4px;
    }
"""

_THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Privacy-First AI Interviewer", theme=_THEME, css=_CSS) as demo:

        # ── Persistent offline badge ──────────────────────────────────────────
        gr.Markdown(
            '<div class="offline-badge">🔒 100% Offline — nothing you say leaves this device</div>',
        )
        gr.Markdown("# 🎙️ Privacy-First AI Interviewer")

        # ── Session state ─────────────────────────────────────────────────────
        state = gr.State(None)

        with gr.Tabs() as tabs:
            # ── Screen 1: Setup ───────────────────────────────────────────────
            with gr.Tab("Setup", id="setup"):
                setup_error_box = gr.Markdown(
                    f"""
                    <div class="warning-box">
                    ⚠️ <strong>Ollama Not Detected / Model Unavailable:</strong> {_OLLAMA_ERROR}<br>
                    Please run <code>ollama serve</code> and <code>ollama pull gemma4:12b</code> in a terminal, then refresh this page.
                    </div>
                    """
                    if _OLLAMA_ERROR
                    else "",
                    visible=bool(_OLLAMA_ERROR),
                )
                gr.Markdown("## 1. Choose your interview type")
                detected_role_state = gr.State(value=None)
                role_dropdown = gr.Dropdown(
                    choices=ROLES,
                    value=ROLES[0],
                    label="Interview Role / Domain",
                    interactive=True,
                )
                role_mismatch_box = gr.Markdown(visible=False)
                switch_role_btn = gr.Button("⚡ Switch to Auto-Detected Role", visible=False)
                gr.Markdown("## 2. Options")
                num_questions_dropdown = gr.Dropdown(
                    choices=[3, 5, 7, 10],
                    value=5,
                    label="Number of Questions",
                    info="Select 3, 5, 7, or 10 questions for your interview session.",
                    interactive=True,
                )
                timer_allotment_dropdown = gr.Dropdown(
                    choices=[60, 90, 120],
                    value=90,
                    label="Per-Question Timer (seconds)",
                    info="Warning tones fire at 66% (amber) and 90% (red) elapsed time.",
                    interactive=True,
                )
                gemma_audio_checkbox = gr.Checkbox(
                    label="🎙️ Enable Gemma 4 Native Audio Perception for ALL turns",
                    value=False,
                    info="Turn 1 uses Gemma 4 native audio by default. Check this to use Gemma 4 native audio for all turns.",
                )
                resume_file = gr.File(
                    label="📄 Upload Resume / JD (.pdf, .txt, .md) — enables resume-grounded questioning",
                    file_types=[".pdf", ".txt", ".md"],
                    type="filepath",
                )
                resume_status_box = gr.Markdown(visible=False)
                gr.Markdown("## 3. Test your microphone")
                mic_test = gr.Audio(
                    sources=["microphone"],
                    label="Record 2 seconds and listen back to confirm your mic works",
                    type="filepath",
                )
                gr.Markdown(
                    "> **Microphone permission:** If your browser asks for microphone access, click **Allow**. "
                    "If the Record button is greyed out, check your browser's site permissions and reload."
                )
                start_btn = gr.Button("▶ Start Interview", variant="primary", size="lg")

            # ── Screen 2: Live Interview ───────────────────────────────────────
            with gr.Tab("Live Interview", id="interview"):
                turn_counter = gr.Markdown("Question 1 of 5")
                question_timer_box = gr.HTML(_build_timer_html(90))
                question_text = gr.Markdown("## 💬 Interviewer's Question\n\n*Your first question will appear here after starting the interview.*")
                # The output audio component renders Gradio's full player once a
                # question is generated: play/pause, seek, playback speed, and
                # replay controls.  Keeping it non-interactive prevents users
                # from replacing the interviewer question with their own file.
                question_audio = gr.Audio(
                    label="Question audio player",
                    autoplay=True,
                    interactive=False,
                    elem_id="question-audio-player",
                    visible=False,
                )
                gr.Markdown("---")
                gr.Markdown("### Your Answer")
                analysis_status_component = gr.Markdown(visible=False)
                stt_badge_component = gr.Markdown("🎙️ **STT Engine:** Gemma 4 is listening directly (Turn 1)")
                fluency_badge_component = gr.Markdown("📊 **Fluency Signal:** 🟢 Confident (Low fillers/hedging)")
                vad_status = gr.Markdown("🎙️ **VAD Status:** Listening...")
                answer_audio = gr.Audio(
                    sources=["microphone"],
                    label="🎤 Click Record, speak your answer (VAD auto-detects end of speech or click Stop)",
                    type="filepath",
                )
                transcript_box = gr.Textbox(
                    label="📝 Your answer (transcribed)",
                    interactive=False,
                    lines=4,
                    placeholder="Transcript will appear here after you stop recording…",
                )
                submit_error_box = gr.Markdown(visible=False)
                with gr.Row():
                    submit_btn = gr.Button("✅ Submit Answer", variant="primary")
                    skip_btn = gr.Button("⏭ Skip Question", variant="secondary")
                finish_btn = gr.Button(
                    "🏁 Finish & Generate Report",
                    variant="stop",
                    visible=False,
                )

            # ── Screen 3: Report ──────────────────────────────────────────────
            with gr.Tab("Report", id="report"):
                gr.Markdown("## 📊 Your Interview Report")
                scorecard_md = gr.Markdown("*Scorecard will appear here after your session.*")

                with gr.Row():
                    radar_plot = gr.Plot(label="🕸️ 5-Dimension Rubric Radar Chart")
                    bar_plot = gr.Plot(label="📊 Dimension Scores Bar Chart")

                with gr.Accordion("📃 Full Transcript", open=False):
                    transcript_full = gr.Markdown("*Transcript will appear here.*")

                pdf_download = gr.File(label="⬇️ Download PDF Report", visible=False)

                new_session_btn = gr.Button("🔄 Start New Session", variant="secondary")

        # ── Event handlers ────────────────────────────────────────────────────

        # Stop-recording pipeline: transcript + optional signals.
        def _on_audio_change(audio, st):
            yield from process_recording_stop(audio, st)

        answer_audio.change(
            fn=_on_audio_change,
            inputs=[answer_audio, state],
            outputs=[
                state,
                transcript_box,
                stt_badge_component,
                fluency_badge_component,
                vad_status,
                submit_error_box,
                analysis_status_component,
            ],
            show_progress="hidden",
        )

        # Auto-start if user switches to Live Interview tab directly
        def _on_tab_select(evt: gr.SelectData, st, role, gemma_all, r_file, num_q, t_sec):
            if evt.value == "Live Interview" and st is None:
                st, q_text, q_audio, turn_lbl, setup_err, r_status, t_html, _ = start_interview(role, gemma_all, r_file, num_q, t_sec)
                return st, q_text, q_audio, turn_lbl, t_html
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

        tabs.select(
            fn=_on_tab_select,
            inputs=[state, role_dropdown, gemma_audio_checkbox, resume_file, num_questions_dropdown, timer_allotment_dropdown],
            outputs=[state, question_text, question_audio, turn_counter, question_timer_box],
        )

        # Auto-detect role & extract highlights on resume upload
        def _on_resume_upload(file_path, current_role):
            if not file_path:
                return gr.update(), gr.update(value="", visible=False), None, gr.update(value="", visible=False), gr.update(visible=False)
            highlights, detected_role, status_notice = detect_resume_role_and_highlights(file_path, ROLES)
            selected_role = detected_role if detected_role else current_role

            raw_txt = extract_text_from_file(file_path)
            ats_info = calculate_ats_score(raw_txt, selected_role)
            if ats_info.get("formatted_md"):
                status_notice += f"\n\n{ats_info['formatted_md']}"

            status_update = gr.update(value=status_notice, visible=bool(status_notice))
            return selected_role, status_update, detected_role, gr.update(value="", visible=False), gr.update(visible=False)

        resume_file.change(
            fn=_on_resume_upload,
            inputs=[resume_file, role_dropdown],
            outputs=[role_dropdown, resume_status_box, detected_role_state, role_mismatch_box, switch_role_btn],
        )

        # Role dropdown manual selection change check against detected_role
        def _on_role_change(selected_role, detected_role):
            if detected_role and selected_role != detected_role:
                msg = f"💡 **Suggestion:** Your resume matches **{detected_role}** best, but **{selected_role}** is selected."
                return gr.update(value=msg, visible=True), gr.update(visible=True)
            return gr.update(value="", visible=False), gr.update(visible=False)

        role_dropdown.change(
            fn=_on_role_change,
            inputs=[role_dropdown, detected_role_state],
            outputs=[role_mismatch_box, switch_role_btn],
        )

        # Switch role button click
        def _on_switch_role(detected_role):
            if detected_role:
                return detected_role, gr.update(value="", visible=False), gr.update(visible=False)
            return gr.skip(), gr.update(value="", visible=False), gr.update(visible=False)

        switch_role_btn.click(
            fn=_on_switch_role,
            inputs=[detected_role_state],
            outputs=[role_dropdown, role_mismatch_box, switch_role_btn],
        )

        # Start interview
        start_btn.click(
            fn=start_interview,
            inputs=[role_dropdown, gemma_audio_checkbox, resume_file, num_questions_dropdown, timer_allotment_dropdown],
            outputs=[state, question_text, question_audio, turn_counter, setup_error_box, resume_status_box, question_timer_box, tabs],
        )

        # Submit answer
        def _on_submit(transcript_text, st):
            st, transcript, stt_badge, fluency_badge, question, audio_out, turn_lbl, finished = process_answer(transcript_text, st)
            finish_visible = gr.update(visible=finished)
            submit_visible = gr.update(visible=not finished)
            skip_visible = gr.update(visible=not finished)
            audio_reset = gr.update(value=None)

            if not transcript.strip():
                return (
                    st,
                    transcript,
                    stt_badge,
                    fluency_badge,
                    question,
                    audio_out,
                    turn_lbl,
                    finish_visible,
                    submit_visible,
                    skip_visible,
                    audio_reset,
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.update(
                        value="⚠️ Transcript is empty. Please record a response before submitting.",
                        visible=True,
                    ),
                    gr.skip(),
                )

            if finished:
                st, scorecard, fig_radar, fig_bar, pdf_path, tab_update = generate_final_report(st)
                transcript_md = "\n\n".join(
                    f"**{'Interviewer' if t['speaker'] == 'interviewer' else 'Candidate'}:** {t['content']}"
                    for t in (st or {}).get("history", [])
                )
                pdf_visible = gr.update(value=pdf_path, visible=bool(pdf_path))
                return (
                    st,
                    transcript,
                    stt_badge,
                    fluency_badge,
                    question,
                    audio_out,
                    turn_lbl,
                    finish_visible,
                    submit_visible,
                    skip_visible,
                    audio_reset,
                    scorecard,
                    fig_radar,
                    fig_bar,
                    transcript_md,
                    pdf_visible,
                    gr.update(value="", visible=False),
                    tab_update,
                )

            return (
                st,
                transcript,
                stt_badge,
                fluency_badge,
                question,
                audio_out,
                turn_lbl,
                finish_visible,
                submit_visible,
                skip_visible,
                audio_reset,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.update(value="", visible=False),
                gr.skip(),
            )

        submit_btn.click(
            fn=_on_submit,
            inputs=[transcript_box, state],
            outputs=[
                state,
                transcript_box,
                stt_badge_component,
                fluency_badge_component,
                question_text,
                question_audio,
                turn_counter,
                finish_btn,
                submit_btn,
                skip_btn,
                answer_audio,
                scorecard_md,
                radar_plot,
                bar_plot,
                transcript_full,
                pdf_download,
                submit_error_box,
                tabs,
            ],
        )

        # Skip question
        def _on_skip(st):
            st, _, question, audio_out, turn_lbl, finished = skip_question(st)
            finish_visible = gr.update(visible=finished)
            submit_visible = gr.update(visible=not finished)
            skip_visible = gr.update(visible=not finished)

            if finished:
                st, scorecard, fig_radar, fig_bar, pdf_path, tab_update = generate_final_report(st)
                transcript_md = "\n\n".join(
                    f"**{'Interviewer' if t['speaker'] == 'interviewer' else 'Candidate'}:** {t['content']}"
                    for t in (st or {}).get("history", [])
                )
                pdf_visible = gr.update(value=pdf_path, visible=bool(pdf_path))
                return st, "[skipped]", question, audio_out, turn_lbl, finish_visible, submit_visible, skip_visible, scorecard, fig_radar, fig_bar, transcript_md, pdf_visible, gr.update(value="", visible=False), tab_update

            return st, "[skipped]", question, audio_out, turn_lbl, finish_visible, submit_visible, skip_visible, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(value="", visible=False), gr.skip()

        skip_btn.click(
            fn=_on_skip,
            inputs=[state],
            outputs=[
                state,
                transcript_box,
                question_text,
                question_audio,
                turn_counter,
                finish_btn,
                submit_btn,
                skip_btn,
                scorecard_md,
                radar_plot,
                bar_plot,
                transcript_full,
                pdf_download,
                submit_error_box,
                tabs,
            ],
        )

        # Finish & generate report
        def _on_finish(st):
            st, scorecard, fig_radar, fig_bar, pdf_path, tab_update = generate_final_report(st)
            # Build transcript text for expandable section
            transcript_md = "\n\n".join(
                f"**{'Interviewer' if t['speaker'] == 'interviewer' else 'Candidate'}:** {t['content']}"
                for t in (st or {}).get("history", [])
            )
            pdf_visible = gr.update(value=pdf_path, visible=bool(pdf_path))
            return st, scorecard, fig_radar, fig_bar, transcript_md, pdf_visible, gr.update(value="", visible=False), tab_update

        finish_btn.click(
            fn=_on_finish,
            inputs=[state],
            outputs=[state, scorecard_md, radar_plot, bar_plot, transcript_full, pdf_download, submit_error_box, tabs],
        )

        # New session → reset to Setup tab
        new_session_btn.click(
            fn=lambda: (None, gr.Tabs(selected="setup")),
            inputs=[],
            outputs=[state, tabs],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
