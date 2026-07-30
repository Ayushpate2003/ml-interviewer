"""
tts/speak.py
------------
Text-to-Speech module using Piper (architecture.md §8.5 / system-design.md §1.7).

Design decisions:
- One pre-downloaded voice model is loaded once at startup.
- The function returns raw WAV bytes so Gradio's gr.Audio output component can
  play it directly without writing to disk.
- CLI fallback: if the ``piper`` Python package is not importable (common on
  macOS), we attempt to call the Piper binary found on PATH. This allows the
  macOS homebrew or manually-installed Piper binary to work transparently.
- Verification is manual only (unittest.md §5 — no automated assertions on
  audio naturalness).

Default voice: en_US-lessac-medium
Override via the PIPER_VOICE environment variable (voice model name).
Override binary path via PIPER_BIN (e.g. /opt/homebrew/bin/piper).
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
_VOICE_NAME = os.environ.get("PIPER_VOICE", "en_US-lessac-medium")
_PIPER_BIN = os.environ.get("PIPER_BIN", "piper")

# Voice model cache directory
_VOICE_DIR = Path(__file__).parent.parent / "data" / "piper_voices"
_VOICE_DIR.mkdir(parents=True, exist_ok=True)

_tts = None  # Piper Python API instance (if available)
_use_cli = False  # True if falling back to CLI binary


class TTSError(Exception):
    """Raised when TTS synthesis fails."""


# ── Model loading ─────────────────────────────────────────────────────────────

def load_tts_model() -> None:
    """
    Load the Piper TTS model.
    Tries the Python API first; falls back to CLI binary.
    Call once at startup.
    """
    global _tts, _use_cli

    if _tts is not None or _use_cli:
        return  # already loaded

    # Attempt 1: Python API
    try:
        from piper import PiperVoice  # noqa: PLC0415

        model_path = _ensure_voice_downloaded(_VOICE_NAME)
        _tts = PiperVoice.load(str(model_path))
        logger.info("Piper TTS loaded via Python API (voice=%s).", _VOICE_NAME)
        return
    except ImportError:
        logger.warning("piper Python package not found — trying CLI binary '%s'.", _PIPER_BIN)
    except Exception as exc:
        logger.warning("Piper Python API failed: %s — trying CLI.", exc)

    # Attempt 2: CLI binary
    bin_path = shutil.which(_PIPER_BIN)
    if bin_path:
        _use_cli = True
        logger.info("Piper TTS will use CLI binary at '%s'.", bin_path)
    else:
        logger.error(
            "Piper TTS is unavailable (no Python package and '%s' not on PATH). "
            "TTS will be skipped. Install piper-tts or place the piper binary on PATH.",
            _PIPER_BIN,
        )


def _ensure_voice_downloaded(voice_name: str) -> Path:
    """
    Return path to the .onnx voice model, downloading if necessary.
    Uses the Piper HTTPS release URL.
    """
    onnx_path = _VOICE_DIR / f"{voice_name}.onnx"
    json_path = _VOICE_DIR / f"{voice_name}.onnx.json"

    if onnx_path.exists() and json_path.exists():
        return onnx_path

    base_url = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        + _voice_path_fragment(voice_name)
    )
    import urllib.request  # noqa: PLC0415

    logger.info("Downloading Piper voice '%s' …", voice_name)
    for url, dest in [
        (f"{base_url}/{voice_name}.onnx", onnx_path),
        (f"{base_url}/{voice_name}.onnx.json", json_path),
    ]:
        urllib.request.urlretrieve(url, dest)  # one-time download at startup
        logger.info("Downloaded %s → %s", url, dest)

    return onnx_path


def _voice_path_fragment(voice_name: str) -> str:
    """
    Build the HuggingFace subfolder path from the voice name.
    e.g. "en_US-lessac-medium" → "en/en_US/lessac/medium"
    """
    parts = voice_name.split("-")
    lang_country = parts[0]  # "en_US"
    lang = lang_country.split("_")[0]  # "en"
    speaker = parts[1] if len(parts) > 1 else "default"
    quality = parts[2] if len(parts) > 2 else "medium"
    return f"{lang}/{lang_country}/{speaker}/{quality}"


# ── Public API ────────────────────────────────────────────────────────────────

def speak(text: str) -> str | None:
    """
    Synthesise ``text`` to speech and return path to a temporary WAV file.

    Returns
    -------
    str | None
        Path to temporary WAV file, or None if TTS is unavailable.
    """
    if not text or not text.strip():
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    # Python API path
    if _tts is not None:
        try:
            import wave
            with wave.open(tmp_path, "wb") as wav_file:
                _tts.synthesize_wav(text, wav_file)
            return tmp_path
        except Exception as exc:
            logger.warning("Piper Python API synthesis failed: %s — trying CLI", exc)

    # CLI path
    if _use_cli:
        return _speak_cli_file(text, tmp_path)

    # TTS unavailable — graceful degradation
    logger.warning("speak(): TTS is unavailable; returning None.")
    return None


def _speak_cli_file(text: str, out_path: str) -> str | None:
    """Generate audio via Piper CLI binary into out_path."""
    onnx_path = _VOICE_DIR / f"{_VOICE_NAME}.onnx"
    try:
        result = subprocess.run(
            [_PIPER_BIN, "--model", str(onnx_path), "--output_file", out_path],
            input=text.encode(),
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise TTSError(f"piper CLI error: {result.stderr.decode()}")
        return out_path
    except Exception as exc:
        logger.warning("piper CLI error: %s", exc)
        return None
