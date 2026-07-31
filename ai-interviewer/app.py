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
  - Refresh → new session (resume is a stretch goal, out of MVP scope).

Startup check (system-design.md §3):
  Ollama reachability and model availability are verified before the UI loads.
  A clear error banner is shown if either check fails — never mid-session.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import gradio as gr

from llm.client import check_ollama_ready, get_next_question, score_session
from memory.db import add_turn, create_session, end_session, save_scores
from report.generate_report import generate_report
from stt.transcribe import TranscriptionError, load_stt_model, transcribe
from tts.speak import TTSError, load_tts_model, speak

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
MAX_TURNS = int(os.environ.get("MAX_TURNS", "5"))
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
def _new_state(role: str) -> dict:
    """Create a fresh session state dict stored in gr.State."""
    session_id = create_session(DB_PATH, role=role)
    return {
        "session_id": session_id,
        "role": role,
        "history": [],          # list of {speaker, content}
        "turn_index": 0,
        "finished": False,
    }


def _add_to_history(state: dict, speaker: str, content: str) -> None:
    """Append to in-memory history and mirror to SQLite."""
    state["history"].append({"speaker": speaker, "content": content})
    add_turn(DB_PATH, session_id=state["session_id"], speaker=speaker, content=content)


def _format_question_md(q: str) -> str:
    """Format question text for display in the Markdown component."""
    if not q or not q.strip():
        return "<div class='warning-box'>⚠️ <strong>No question generated.</strong> Please check if Ollama is running.</div>"
    if q.startswith("⚠️") or q.startswith("Error") or "Error" in q:
        return f"<div class='warning-box'>⚠️ <strong>Interviewer Error:</strong> {q}</div>"
    if q.startswith("###") or q.startswith("Session"):
        return q
    return f"## 💬 Interviewer's Question\n\n### **{q}**"


def _question_audio_update(audio: bytes | None):
    """Only show the player after TTS has produced playable question audio."""
    return gr.update(value=audio, visible=audio is not None)


def start_interview(role: str) -> tuple:
    """
    Initialise a new session and generate the first question.
    Returns: (state, question_text, audio_bytes, turn_label, tab_update)
    """
    ok, err = check_ollama_ready()
    if not ok:
        logger.error("start_interview failed health check: %s", err)
        return None, _format_question_md(f"⚠️ Cannot start: {err}"), _question_audio_update(None), "", gr.Tabs(selected="setup")

    try:
        state = _new_state(role)
        question = get_next_question([], role)
        _add_to_history(state, "interviewer", question)

        audio = None
        if _TTS_READY:
            try:
                audio = speak(question)
            except Exception as exc:
                logger.warning("TTS failed on first question: %s", exc)

        turn_label = f"Question 1 of ~{MAX_TURNS}"
        return state, _format_question_md(question), _question_audio_update(audio), turn_label, gr.Tabs(selected="interview")
    except Exception as exc:
        logger.exception("start_interview error: %s", exc)
        return None, _format_question_md(f"⚠️ Error starting interview: {exc}"), _question_audio_update(None), "Error", gr.Tabs(selected="setup")


def process_answer(audio_input: bytes | str | None, state: dict) -> tuple:
    """
    Process one candidate answer and return the next question.
    Returns: (state, transcript_text, question_text, audio_bytes, turn_label, finish_flag)
    """
    if state is None or state.get("finished"):
        return state, "", "", _question_audio_update(None), "", True

    # Transcribe
    transcript = ""
    if audio_input:
        try:
            transcript = transcribe(audio_input)
        except TranscriptionError as exc:
            logger.warning("TranscriptionError: %s", exc)
            transcript = ""

    if not transcript.strip():
        # Silence / empty — prompt retry, no LLM call
        last_q = state["history"][-1]["content"] if state["history"] else ""
        return state, "[I didn't catch that — please try again]", _format_question_md(last_q), _question_audio_update(None), f"Question {state['turn_index'] + 1} of ~{MAX_TURNS}", False

    _add_to_history(state, "candidate", transcript)
    state["turn_index"] += 1

    # End of session?
    if state["turn_index"] >= MAX_TURNS:
        state["finished"] = True
        return state, transcript, "### Session complete — generating your report…", _question_audio_update(None), f"Question {state['turn_index']} of {MAX_TURNS}", True

    # Next question
    try:
        question = get_next_question(state["history"], state["role"])
        _add_to_history(state, "interviewer", question)
    except Exception as exc:
        logger.error("LLM call failed in process_answer: %s", exc)
        question = f"⚠️ LLM Error: {exc}. Please check if Ollama is running."

    audio = None
    if _TTS_READY and not question.startswith("⚠️"):
        try:
            audio = speak(question)
        except Exception as exc:
            logger.warning("TTS failed: %s", exc)

    turn_label = f"Question {state['turn_index'] + 1} of ~{MAX_TURNS}"
    return state, transcript, _format_question_md(question), _question_audio_update(audio), turn_label, False


def skip_question(state: dict) -> tuple:
    """Add [skipped] placeholder and advance to the next question."""
    if state is None or state.get("finished"):
        return state, "", "", _question_audio_update(None), "", True

    _add_to_history(state, "candidate", "[skipped]")
    state["turn_index"] += 1

    if state["turn_index"] >= MAX_TURNS:
        state["finished"] = True
        return state, "[skipped]", "Session complete — generating your report…", _question_audio_update(None), f"Question {state['turn_index']} of {MAX_TURNS}", True

    try:
        question = get_next_question(state["history"], state["role"])
        _add_to_history(state, "interviewer", question)
    except Exception as exc:
        logger.error("LLM call failed in skip_question: %s", exc)
        question = f"⚠️ LLM Error: {exc}. Please check if Ollama is running."

    audio = None
    if _TTS_READY and not question.startswith("⚠️"):
        try:
            audio = speak(question)
        except Exception as exc:
            logger.warning("TTS failed: %s", exc)

    turn_label = f"Question {state['turn_index'] + 1} of ~{MAX_TURNS}"
    return state, "[skipped]", _format_question_md(question), _question_audio_update(audio), turn_label, False


def generate_final_report(state: dict) -> tuple:
    """Score the session with Gemma 4 and build the PDF report."""
    if state is None:
        return None, "No active session.", None, gr.update()

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

    # Build a markdown scorecard summary
    summary_md = _format_scorecard_md(scorecard)

    return state, summary_md, str(pdf_path), gr.Tabs(selected="report")


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
    with gr.Blocks(title="Privacy-First AI Interviewer") as demo:

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
                if _OLLAMA_ERROR:
                    gr.Markdown(
                        f"""
                        <div class="warning-box">
                        ⚠️ <strong>Ollama Not Detected / Model Unavailable:</strong> {_OLLAMA_ERROR}<br>
                        Please run <code>ollama serve</code> and <code>ollama pull gemma4:12b</code> in a terminal, then refresh this page.
                        </div>
                        """
                    )
                gr.Markdown("## 1. Choose your interview type")
                role_dropdown = gr.Dropdown(
                    choices=ROLES,
                    value=ROLES[0],
                    label="Interview Role / Domain",
                    interactive=True,
                )
                gr.Markdown("## 2. Test your microphone")
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
                turn_counter = gr.Markdown("Question 1 of ~5")
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
                answer_audio = gr.Audio(
                    sources=["microphone"],
                    label="🎤 Click Record, speak your answer, then click Stop",
                    type="filepath",
                )
                transcript_box = gr.Textbox(
                    label="📝 Your answer (transcribed)",
                    interactive=False,
                    lines=4,
                    placeholder="Transcript will appear here after you stop recording…",
                )
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
                with gr.Accordion("📃 Full Transcript", open=False):
                    transcript_full = gr.Markdown("*Transcript will appear here.*")
                pdf_download = gr.File(label="⬇️ Download PDF Report", visible=False)
                new_session_btn = gr.Button("🔄 Start New Session", variant="secondary")

        # ── Event handlers ────────────────────────────────────────────────────

        # Auto-start if user switches to Live Interview tab directly
        def _on_tab_select(evt: gr.SelectData, st, role):
            if evt.value == "Live Interview" and st is None:
                st, q_text, q_audio, turn_lbl, _ = start_interview(role)
                return st, q_text, q_audio, turn_lbl
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()

        tabs.select(
            fn=_on_tab_select,
            inputs=[state, role_dropdown],
            outputs=[state, question_text, question_audio, turn_counter],
        )

        # Start interview
        start_btn.click(
            fn=start_interview,
            inputs=[role_dropdown],
            outputs=[state, question_text, question_audio, turn_counter, tabs],
        )

        # Submit answer
        def _on_submit(audio, st):
            st, transcript, question, audio_out, turn_lbl, finished = process_answer(audio, st)
            finish_visible = gr.update(visible=finished)
            submit_visible = gr.update(visible=not finished)
            skip_visible = gr.update(visible=not finished)
            return st, transcript, question, audio_out, turn_lbl, finish_visible, submit_visible, skip_visible

        submit_btn.click(
            fn=_on_submit,
            inputs=[answer_audio, state],
            outputs=[state, transcript_box, question_text, question_audio, turn_counter, finish_btn, submit_btn, skip_btn],
        )

        # Skip question
        def _on_skip(st):
            st, _, question, audio_out, turn_lbl, finished = skip_question(st)
            finish_visible = gr.update(visible=finished)
            submit_visible = gr.update(visible=not finished)
            skip_visible = gr.update(visible=not finished)
            return st, "[skipped]", question, audio_out, turn_lbl, finish_visible, submit_visible, skip_visible

        skip_btn.click(
            fn=_on_skip,
            inputs=[state],
            outputs=[state, transcript_box, question_text, question_audio, turn_counter, finish_btn, submit_btn, skip_btn],
        )

        # Finish & generate report
        def _on_finish(st):
            st, scorecard, pdf_path, tab_update = generate_final_report(st)
            # Build transcript text for expandable section
            transcript_md = "\n\n".join(
                f"**{'Interviewer' if t['speaker'] == 'interviewer' else 'Candidate'}:** {t['content']}"
                for t in (st or {}).get("history", [])
            )
            pdf_visible = gr.update(value=pdf_path, visible=bool(pdf_path))
            return st, scorecard, transcript_md, pdf_visible, tab_update

        finish_btn.click(
            fn=_on_finish,
            inputs=[state],
            outputs=[state, scorecard_md, transcript_full, pdf_download, tabs],
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
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=_THEME,
        css=_CSS,
    )
