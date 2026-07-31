"""
tests/test_gemma_audio.py
--------------------------
Unit tests for stt/gemma_audio.py (Gemma 4 native audio understanding & 30s fallback).
"""

from unittest.mock import MagicMock, patch
import numpy as np
import soundfile as sf
import pytest

from stt.gemma_audio import (
    MAX_NATIVE_AUDIO_DURATION_SEC,
    get_audio_duration,
    transcribe_native_gemma,
)


def test_get_audio_duration(tmp_path):
    sr = 16000
    t = np.linspace(0, 5.0, 5 * sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    audio_file = tmp_path / "duration_5s.wav"
    sf.write(str(audio_file), audio, sr)

    duration = get_audio_duration(audio_file)
    assert abs(duration - 5.0) < 0.1


@patch("stt.gemma_audio.get_active_model_tag")
@patch("stt.gemma_audio.requests.post")
def test_transcribe_native_gemma_success(mock_post, mock_model, tmp_path):
    mock_model.return_value = "gemma4:12b"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": "I built a distributed system using Python."}
    }
    mock_post.return_value = mock_response

    sr = 16000
    t = np.linspace(0, 2.0, 2 * sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio_file = tmp_path / "short_audio.wav"
    sf.write(str(audio_file), audio, sr)

    transcript, badge = transcribe_native_gemma(audio_file)
    assert transcript == "I built a distributed system using Python."
    assert "Gemma 4 Native Audio Perception" in badge


@patch("stt.gemma_audio.transcribe")
def test_transcribe_native_gemma_fallback_over_30s(mock_transcribe, tmp_path):
    mock_transcribe.return_value = "Long 35-second answer transcript via Whisper."

    sr = 16000
    # 35 seconds audio
    t = np.linspace(0, 35.0, 35 * sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio_file = tmp_path / "long_audio.wav"
    sf.write(str(audio_file), audio, sr)

    transcript, badge = transcribe_native_gemma(audio_file)
    assert transcript == "Long 35-second answer transcript via Whisper."
    assert "faster-whisper" in badge
    mock_transcribe.assert_called_once()
