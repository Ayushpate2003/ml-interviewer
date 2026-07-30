"""
stt/transcribe.py
-----------------
Speech-to-Text module using faster-whisper.

Design decisions (architecture.md §8.2):
- Model loaded ONCE at module init (not per-turn) to avoid repeated load latency.
- Uses 'small' model by default; 'base.en' is a valid alternative for
  English-only demos on very constrained hardware.
- Compute type is auto-detected: 'int8' on Apple Silicon/CPU, 'float16' if
  CUDA is available.
- Empty / near-silent audio → returns "" (do NOT call the LLM for empty input).
- Malformed / non-audio bytes → raises TranscriptionError.
"""

from __future__ import annotations

import io
import logging
import os
import platform
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Model configuration ────────────────────────────────────────────────────────
_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")

def _choose_compute_type() -> str:
    """Pick the fastest compute type available on this machine."""
    try:
        import ctranslate2  # bundled with faster-whisper
        if ctranslate2.get_cuda_device_count() > 0:
            return "float16"
    except Exception:
        pass
    # Apple Silicon and generic CPU
    return "int8"

_DEVICE = "cpu"
_COMPUTE_TYPE = _choose_compute_type()

# ── Singleton model ────────────────────────────────────────────────────────────
_model = None


class TranscriptionError(Exception):
    """Raised when audio bytes cannot be transcribed (malformed / corrupt data)."""


def load_stt_model():
    """
    Load faster-whisper model into the module-level singleton.
    Call once at application startup; subsequent calls are no-ops.
    """
    global _model
    if _model is not None:
        return _model

    from faster_whisper import WhisperModel  # noqa: PLC0415

    logger.info(
        "Loading faster-whisper '%s' model (device=%s, compute_type=%s) …",
        _MODEL_SIZE,
        _DEVICE,
        _COMPUTE_TYPE,
    )
    _model = WhisperModel(_MODEL_SIZE, device=_DEVICE, compute_type=_COMPUTE_TYPE)
    logger.info("faster-whisper model loaded.")
    return _model


def transcribe(audio_input: bytes | str | Path | None) -> str:
    """
    Transcribe raw audio bytes or a file path to text.

    Parameters
    ----------
    audio_input : bytes | str | Path | None
        Raw audio data bytes or a file path string (as returned by Gradio's
        gr.Audio component with type="filepath"), or None.

    Returns
    -------
    str
        Transcribed text, stripped of leading/trailing whitespace.
        Returns "" for silence / near-silent input.

    Raises
    ------
    TranscriptionError
        If audio_input is malformed or cannot be decoded by whisper.
    """
    if not audio_input:
        logger.debug("transcribe(): received empty audio input — returning ''.")
        return ""

    model = load_stt_model()

    tmp_path = None
    created_tmp = False

    try:
        if isinstance(audio_input, (str, Path)):
            tmp_path = str(audio_input)
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                return ""
        elif isinstance(audio_input, bytes):
            if len(audio_input) == 0:
                return ""
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_input)
                tmp_path = tmp.name
                created_tmp = True
        else:
            raise TranscriptionError(f"Unsupported audio input type: {type(audio_input)}")

        segments, info = model.transcribe(tmp_path, beam_size=5)
        text = " ".join(seg.text for seg in segments).strip()
        logger.debug("transcribe(): '%s' (lang=%s)", text[:80], info.language)
        return text

    except Exception as exc:
        err_msg = str(exc).lower()
        if "no speech" in err_msg or "silent" in err_msg:
            return ""
        raise TranscriptionError(f"Failed to transcribe audio: {exc}") from exc

    finally:
        if created_tmp and tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def get_model() -> object | None:
    """Return the loaded model singleton (used by tests)."""
    return _model
