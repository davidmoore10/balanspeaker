"""Tests for text-to-speech fallback behaviour."""

import pytest

from speech.errors import SpeechProviderUnavailableError
from speech.fallback import FallbackSpeechProvider
from speech.provider import SpeechProvider


class FixedSpeechProvider(SpeechProvider):
    """Controllable speech provider used by tests."""

    def __init__(
        self,
        *,
        provider_name: str,
        unavailable: bool = False,
    ) -> None:
        self._provider_name = provider_name
        self._unavailable = unavailable
        self.utterances: list[str] = []
        self.stop_count = 0

    @property
    def name(self) -> str:
        """Return the provider name."""

        return self._provider_name

    async def speak(self, text: str) -> None:
        """Record or reject an utterance."""

        if self._unavailable:
            raise SpeechProviderUnavailableError("Provider unavailable")

        self.utterances.append(text)

    async def stop(self) -> None:
        """Record a stop request."""

        if self._unavailable:
            raise SpeechProviderUnavailableError("Provider unavailable")

        self.stop_count += 1


@pytest.mark.asyncio
async def test_primary_provider_is_used() -> None:
    """The primary provider should normally speak."""

    primary = FixedSpeechProvider(provider_name="primary")
    fallback = FixedSpeechProvider(provider_name="fallback")

    provider = FallbackSpeechProvider(
        primary=primary,
        fallback=fallback,
    )

    await provider.speak("Hello")

    assert primary.utterances == ["Hello"]
    assert fallback.utterances == []
    assert provider.last_provider_name == "primary"
    assert not provider.last_used_fallback


@pytest.mark.asyncio
async def test_fallback_is_used_when_primary_unavailable() -> None:
    """An unavailable primary should activate the fallback."""

    primary = FixedSpeechProvider(
        provider_name="primary",
        unavailable=True,
    )
    fallback = FixedSpeechProvider(provider_name="fallback")

    provider = FallbackSpeechProvider(
        primary=primary,
        fallback=fallback,
    )

    await provider.speak("Hello")

    assert fallback.utterances == ["Hello"]
    assert provider.last_provider_name == "fallback"
    assert provider.last_used_fallback
