"""Interfaces implemented by speech-to-text providers."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from speech_recognition.result import TranscriptionResult


class SpeechToTextProvider(ABC):
    """Interface for local or cloud transcription providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""

    @abstractmethod
    async def transcribe(
        self,
        *,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> TranscriptionResult:
        """Convert mono floating-point audio into text."""
