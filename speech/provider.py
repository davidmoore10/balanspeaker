"""Interfaces implemented by text-to-speech providers."""

from abc import ABC, abstractmethod


class SpeechProvider(ABC):
    """Interface for local or cloud text-to-speech providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Speak text aloud."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop current speech when supported."""
