"""
tests/test_vad.py
------------------
Unit tests for stt/vad.py (Silero VAD end-of-speech detection).
"""

import numpy as np
import soundfile as sf
import pytest

from stt.vad import load_vad_model, read_audio_data, check_end_of_speech


def test_load_vad_model():
    model, utils = load_vad_model()
    assert model is not None
    assert utils is not None
    assert "get_speech_timestamps" in utils


def test_read_audio_data(tmp_path):
    # Generate 1 second of 16kHz sine wave audio
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    audio_file = tmp_path / "test.wav"
    sf.write(str(audio_file), audio, sr)

    data, rate = read_audio_data(audio_file)
    assert rate == 16000
    assert len(data) == sr


def test_check_end_of_speech_silence(tmp_path):
    # Generate 3 seconds of audio: 1s tone (speech) followed by 2s silence
    sr = 16000
    t_speech = np.linspace(0, 1.0, sr, dtype=np.float32)
    speech = 0.5 * np.sin(2 * np.pi * 440 * t_speech)
    silence = np.zeros(2 * sr, dtype=np.float32)
    full_audio = np.concatenate([speech, silence])

    audio_file = tmp_path / "speech_then_silence.wav"
    sf.write(str(audio_file), full_audio, sr)

    is_end, status = check_end_of_speech(audio_file, silence_threshold_sec=1.5)
    assert is_end is True
    assert "✓ Got it" in status


def test_check_end_of_speech_ongoing(tmp_path):
    # Generate 2 seconds of continuous tone (ongoing speech, no trailing silence)
    sr = 16000
    t = np.linspace(0, 2.0, 2 * sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    audio_file = tmp_path / "ongoing_speech.wav"
    sf.write(str(audio_file), audio, sr)

    is_end, status = check_end_of_speech(audio_file, silence_threshold_sec=1.5)
    assert is_end is False
    assert "🎙️ Listening" in status
