"""Wake detection with openWakeWord and a local Whisper fallback."""

import re
from collections import deque

import numpy as np
from numpy.typing import NDArray

from speech_recognition.errors import EmptyTranscriptionError
from speech_recognition.provider import SpeechToTextProvider
from wake_word.provider import WakeWordProvider
from wake_word.result import WakeWordDetection


class HybridWakeWordProvider(WakeWordProvider):
    """Use fast model scoring plus periodic phrase transcription."""

    def __init__(
        self,
        *,
        primary: WakeWordProvider,
        transcriber: SpeechToTextProvider,
        phrase: str = "hey jarvis",
        window_frames: int = 40,
        transcription_interval_frames: int = 16,
        minimum_frames: int = 24,
    ) -> None:
        self._primary = primary
        self._transcriber = transcriber
        self._phrase = _normalize_text(phrase)
        self._frames: deque[NDArray[np.int16]] = deque(maxlen=window_frames)
        self._transcription_interval_frames = transcription_interval_frames
        self._minimum_frames = minimum_frames
        self._frames_since_transcription = 0

    @property
    def name(self) -> str:
        """Return the combined provider name."""

        return f"hybrid:{self._primary.name}+{self._transcriber.name}"

    @property
    def sample_rate(self) -> int:
        """Return the shared audio sample rate."""

        return self._primary.sample_rate

    async def process(
        self,
        audio: NDArray[np.int16],
    ) -> WakeWordDetection | None:
        """Check the fast detector, then periodically transcribe a rolling window."""

        normalized_audio = np.asarray(audio, dtype=np.int16).reshape(-1)

        if normalized_audio.size == 0:
            return None

        self._frames.append(normalized_audio.copy())
        self._frames_since_transcription += 1

        detection = await self._primary.process(normalized_audio)

        if detection is not None:
            self._reset_window()
            return detection

        if (
            len(self._frames) < self._minimum_frames
            or self._frames_since_transcription < self._transcription_interval_frames
        ):
            return None

        self._frames_since_transcription = 0
        pcm = np.concatenate(tuple(self._frames))
        floating_audio = pcm.astype(np.float32) / 32768.0

        try:
            result = await self._transcriber.transcribe(
                audio=floating_audio,
                sample_rate=self.sample_rate,
            )
        except EmptyTranscriptionError:
            return None

        normalized_text = _normalize_text(result.text)

        if self._phrase not in normalized_text:
            return None

        self._reset_window()
        return WakeWordDetection(
            model_name=self._phrase.replace(" ", "_"),
            score=1.0,
        )

    def _reset_window(self) -> None:
        """Discard audio already associated with a detection."""

        self._frames.clear()
        self._frames_since_transcription = 0


def _normalize_text(value: str) -> str:
    """Normalize transcribed text for exact phrase matching."""

    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())
