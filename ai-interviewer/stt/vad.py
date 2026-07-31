"""
stt/vad.py
----------
Silero VAD & Energy-based Voice Activity Detection wrapper.
Detects trailing silence after candidate speech (default 1.5s threshold).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
# pyrefly: ignore [missing-import]
import numpy as np

logger = logging.getLogger(__name__)

_VAD_MODEL = None
_VAD_UTILS = None


class VADError(Exception):
    """Raised when VAD processing fails."""


def load_vad_model():
    """
    Load the Silero VAD model using PyTorch.
    Loaded once globally at app startup.
    """
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is not None:
        return _VAD_MODEL, _VAD_UTILS

    try:
        from silero_vad import load_silero_vad, get_speech_timestamps

        _VAD_MODEL = load_silero_vad()
        _VAD_UTILS = {"get_speech_timestamps": get_speech_timestamps}
        logger.info("Silero VAD model loaded successfully.")
        return _VAD_MODEL, _VAD_UTILS
    except Exception as exc:
        logger.warning("Silero VAD load failed (will degrade gracefully): %s", exc)
        return None, None


def read_audio_data(audio_input: str | Path | bytes) -> tuple[np.ndarray, int]:
    """
    Load audio input into float32 numpy array and sample rate.
    """
    import soundfile as sf

    if isinstance(audio_input, (str, Path)):
        data, sr = sf.read(str(audio_input), dtype="float32")
    elif isinstance(audio_input, bytes):
        data, sr = sf.read(io.BytesIO(audio_input), dtype="float32")
    else:
        raise VADError(f"Unsupported audio input type: {type(audio_input)}")

    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo to mono

    return data, sr


def check_end_of_speech(
    audio_input: str | Path | bytes,
    silence_threshold_sec: float = 1.5,
    min_speech_duration_sec: float = 0.5,
) -> tuple[bool, str]:
    """
    Check if speech/sound has occurred and is followed by >= silence_threshold_sec trailing silence.
    """
    model, utils = load_vad_model()

    try:
        data, sr = read_audio_data(audio_input)
        if len(data) == 0:
            return False, "🎙️ Listening..."

        duration_sec = len(data) / float(sr)
        if duration_sec < (min_speech_duration_sec + silence_threshold_sec):
            # Clip is shorter than min speech + trailing silence threshold
            return False, "🎙️ Listening..."

        # Frame-wise energy analysis (100ms frames)
        frame_len = int(sr * 0.1)
        num_frames = len(data) // frame_len
        if num_frames == 0:
            return False, "🎙️ Listening..."

        energies = [
            np.sqrt(np.mean(data[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(num_frames)
        ]

        # Calculate noise floor dynamically (bottom 20th percentile)
        noise_floor = float(np.percentile(energies, 20))
        speech_energy_threshold = max(noise_floor * 3.0, 0.01)

        # Detect frames with active speech / sound
        speech_frames = [idx for idx, e in enumerate(energies) if e > speech_energy_threshold]

        if not speech_frames:
            # Check with Silero VAD if energy threshold missed human speech
            if model is not None and utils is not None:
                import torch
                tensor_data = torch.from_numpy(data)
                get_speech_ts = utils["get_speech_timestamps"]
                timestamps = get_speech_ts(tensor_data, model, sampling_rate=sr, threshold=0.3)
                if timestamps:
                    last_end_sec = timestamps[-1]["end"] / sr
                    trailing_silence = duration_sec - last_end_sec
                    if trailing_silence >= silence_threshold_sec:
                        return True, "✓ Got it (end-of-speech detected)"
            return False, "🎙️ Listening..."

        # Find the last active speech frame
        last_speech_frame = speech_frames[-1]
        last_speech_end_sec = (last_speech_frame + 1) * 0.1
        trailing_silence_sec = duration_sec - last_speech_end_sec

        total_speech_sec = len(speech_frames) * 0.1

        logger.debug(
            "VAD stats: total_speech=%.2fs, last_end=%.2fs, trailing_silence=%.2fs (duration=%.2fs)",
            total_speech_sec,
            last_speech_end_sec,
            trailing_silence_sec,
            duration_sec,
        )

        if total_speech_sec >= min_speech_duration_sec and trailing_silence_sec >= silence_threshold_sec:
            return True, "✓ Got it (end-of-speech detected)"
        else:
            return False, "🎙️ Listening..."

    except Exception as exc:
        logger.warning("VAD check error: %s (falling back to manual mode)", exc)
        return False, "🎙️ Listening (manual mode)"
