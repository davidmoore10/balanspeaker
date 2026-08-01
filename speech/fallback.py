"""Fallback coordination between speech providers."""

from speech.errors import SpeechProviderUnavailableError
from speech.provider import SpeechProvider


class FallbackSpeechProvider(SpeechProvider):
    """Use a secondary provider when the primary is unavailable."""

    def __init__(
        self,
        *,
        primary: SpeechProvider,
        fallback: SpeechProvider,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._last_provider_name: str | None = None
        self._last_used_fallback = False

    @property
    def name(self) -> str:
        """Return a description of the provider chain."""

        return f"{self._primary.name} -> {self._fallback.name}"

    @property
    def last_provider_name(self) -> str | None:
        """Return the provider used for the latest utterance."""

        return self._last_provider_name

    @property
    def last_used_fallback(self) -> bool:
        """Return whether the latest utterance used the fallback."""

        return self._last_used_fallback

    async def speak(self, text: str) -> None:
        """Speak using the primary provider when available."""

        try:
            await self._primary.speak(text)
        except SpeechProviderUnavailableError:
            self._last_provider_name = self._fallback.name
            self._last_used_fallback = True

            await self._fallback.speak(text)
            return

        self._last_provider_name = self._primary.name
        self._last_used_fallback = False

    async def stop(self) -> None:
        """Ask both providers to stop safely."""

        try:
            await self._primary.stop()
        except SpeechProviderUnavailableError:
            pass

        await self._fallback.stop()
