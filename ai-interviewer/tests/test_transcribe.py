"""
tests/test_transcribe.py
-------------------------
Unit tests for stt/transcribe.py (unittest.md §3.2).

The faster-whisper WhisperModel is mocked in all tests — no live model download
or inference required for CI-style checks.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stt.transcribe import TranscriptionError, transcribe


class TestTranscribe:

    def test_empty_audio_returns_empty_string(self, silent_audio_bytes):
        """
        Near-silent / all-zero WAV → transcribe() should return "".
        We mock the model to return no segments (as whisper does for silence).
        """
        mock_segment = MagicMock()
        mock_segment.text = ""
        mock_info = MagicMock()
        mock_info.language = "en"

        with patch("stt.transcribe.load_stt_model") as mock_load, \
             patch("stt.transcribe.tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("stt.transcribe.os.unlink"):

            mock_model = MagicMock()
            mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)
            mock_load.return_value = mock_model

            # Patch the global _model to bypass re-loading
            import stt.transcribe as mod
            mod._model = mock_model

            result = transcribe(silent_audio_bytes)
            assert result.strip() == ""

    def test_none_audio_returns_empty_string(self):
        """None input (no recording) → returns "" without calling the model."""
        result = transcribe(None)
        assert result == ""

    def test_transcribe_returns_nonempty_for_known_sample(self, sample_audio_bytes):
        """
        Non-silent audio → model returns at least one segment with text.
        We mock the model to return a canned transcript.
        """
        mock_segment = MagicMock()
        mock_segment.text = " I once debugged a race condition in a distributed lock."
        mock_info = MagicMock()
        mock_info.language = "en"

        import stt.transcribe as mod
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)
        mod._model = mock_model

        with patch("stt.transcribe.os.unlink"):
            result = transcribe(sample_audio_bytes)
            assert len(result) > 0

    def test_handles_unsupported_audio_gracefully(self):
        """
        Malformed bytes → WhisperModel raises a non-silence exception →
        transcribe() wraps it in TranscriptionError.
        """
        import stt.transcribe as mod
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("Invalid audio format")
        mod._model = mock_model

        with pytest.raises(TranscriptionError):
            transcribe(b"not-real-audio-bytes")
