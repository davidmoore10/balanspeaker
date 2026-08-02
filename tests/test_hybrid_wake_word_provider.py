"""Tests for the hybrid wake-word provider."""

import numpy as np
import pytest

from speech_recognition.provider import SpeechToTextProvider
from speech_recognition.result import TranscriptionResult
from wake_word.hybrid_provider import HybridWakeWordProvider
from wake_word.provider import WakeWordProvider
from wake_word.result import WakeWordDetection


class SilentWakeProvider(WakeWordProvider):
    """Primary provider that never detects a phrase."""

    @property
    def name(self) -> str:
        return "silent-wake"

    @property
    def sample_rate(self) -> int:
        return 16000

    async def process(
        self,
        audio: np.ndarray,
    ) -> WakeWordDetection | None:
        del audio
        return None


class StubTranscriber(SpeechToTextProvider):
    """Return configured text for rolling audio."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    @property
    def name(self) -> str:
        return "stub-transcriber"

    async def transcribe(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
    ) -> TranscriptionResult:
        assert audio.dtype == np.float32
        assert sample_rate == 16000
        self.calls += 1
        return TranscriptionResult(text=self._text)


@pytest.mark.asyncio
async def test_whisper_fallback_detects_phrase() -> None:
    """Transcribed wake phrases should activate when the fast model misses."""

    transcriber = StubTranscriber("Hey, Jarvis!")
    provider = HybridWakeWordProvider(
        primary=SilentWakeProvider(),
        transcriber=transcriber,
        window_frames=2,
        minimum_frames=2,
        transcription_interval_frames=2,
    )

    assert await provider.process(np.zeros(1280, dtype=np.int16)) is None
    detection = await provider.process(np.zeros(1280, dtype=np.int16))

    assert detection is not None
    assert detection.model_name == "hey_jarvis"
    assert detection.score == 1.0
    assert transcriber.calls == 1


@pytest.mark.asyncio
async def test_whisper_fallback_ignores_other_speech() -> None:
    """Unrelated transcriptions should not activate."""

    provider = HybridWakeWordProvider(
        primary=SilentWakeProvider(),
        transcriber=StubTranscriber("Hello there"),
        window_frames=2,
        minimum_frames=2,
        transcription_interval_frames=2,
    )

    await provider.process(np.zeros(1280, dtype=np.int16))

    assert await provider.process(np.zeros(1280, dtype=np.int16)) is None
