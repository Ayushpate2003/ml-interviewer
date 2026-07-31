"""
stt/gemma_audio.py
------------------
Gemma 4 native audio transcription & understanding via Ollama API.
Sends raw audio (base64-encoded) directly to Gemma 4 for native perception.
Includes 30-second duration check with automatic fallback to faster-whisper.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import requests
import soundfile as sf

from llm.client import OLLAMA_BASE_URL, get_active_model_tag
from stt.transcribe import transcribe

logger = logging.getLogger(__name__)

MAX_NATIVE_AUDIO_DURATION_SEC = 30.0


class GemmaAudioError(Exception):
    """Raised when Gemma 4 native audio processing fails."""


def get_audio_duration(audio_input: str | Path | bytes) -> float:
    """
    Get duration of audio file or bytes in seconds.
    """
    if isinstance(audio_input, (str, Path)):
        info = sf.info(str(audio_input))
        return info.duration
    elif isinstance(audio_input, bytes):
        data, sr = sf.read(io.BytesIO(audio_input), dtype="float32")
        return len(data) / float(sr)
    return 0.0


def transcribe_native_gemma(
    audio_input: str | Path | bytes,
    prompt: str = "Listen carefully to this audio recording of a candidate answer. Provide an accurate transcript of what was spoken.",
) -> tuple[str, str]:
    """
    Transcribe audio natively using Gemma 4 multimodal capability.

    Returns
    -------
    tuple[str, str]
        (transcript_text, badge_label)
    """
    try:
        duration = get_audio_duration(audio_input)
        logger.info("Audio duration: %.2fs (Max for Gemma 4 native: %.2fs)", duration, MAX_NATIVE_AUDIO_DURATION_SEC)

        if duration > MAX_NATIVE_AUDIO_DURATION_SEC:
            logger.info("Audio duration %.2fs > 30s limit; falling back to faster-whisper.", duration)
            fallback_text = transcribe(audio_input)
            return fallback_text, "⚡ faster-whisper (auto fallback: >30s answer)"

        # Read audio bytes & base64 encode
        if isinstance(audio_input, (str, Path)):
            with open(audio_input, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(audio_input, bytes):
            raw_bytes = audio_input
        else:
            raise GemmaAudioError(f"Invalid audio type: {type(audio_input)}")

        b64_audio = base64.b64encode(raw_bytes).decode("utf-8")
        model_tag = get_active_model_tag()

        # Send multimodal request to Ollama chat endpoint
        payload = {
            "model": model_tag,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64_audio],
                }
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256},
        }

        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=20.0)
        resp.raise_for_status()

        data = resp.json()
        transcript = data.get("message", {}).get("content", "").strip()

        if transcript:
            logger.info("Gemma 4 native audio transcription successful.")
            return transcript, "🎙️ Gemma 4 Native Audio Perception"
        else:
            logger.warning("Gemma 4 returned empty native transcript; falling back to faster-whisper.")
            fallback_text = transcribe(audio_input)
            return fallback_text, "⚡ faster-whisper (fallback: empty Gemma output)"

    except Exception as exc:
        logger.warning("Gemma 4 native audio attempt failed: %s. Falling back to faster-whisper.", exc)
        try:
            fallback_text = transcribe(audio_input)
            return fallback_text, "⚡ faster-whisper (fallback)"
        except Exception as fallback_exc:
            logger.error("Both Gemma 4 native and faster-whisper failed: %s", fallback_exc)
            raise GemmaAudioError(f"Audio transcription failed: {exc}") from exc
