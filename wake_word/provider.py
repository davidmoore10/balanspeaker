"""Interfaces implemented by wake-word providers."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from wake_word.result import WakeWordDetection


class WakeWordProvider(ABC):
    """Detect a configured wake phrase in PCM audio."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the required audio sample rate."""

    @abstractmethod
    async def process(
        self,
        audio: NDArray[np.int16],
    ) -> WakeWordDetection | None:
        """Process one PCM audio frame."""
