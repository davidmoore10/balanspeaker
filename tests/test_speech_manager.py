"""Tests for asynchronous interruptible speech."""

import asyncio

import pytest

from speech.manager import SpeechManager
from speech.provider import SpeechProvider


class ControlledSpeechProvider(SpeechProvider):
    """Speech provider controlled by per-utterance events."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.utterances: list[str] = []
        self.stop_count = 0
        self._active_release: asyncio.Event | None = None

    @property
    def name(self) -> str:
        """Return the provider name."""

        return "controlled"

    async def speak(self, text: str) -> None:
        """Wait until the current utterance is released."""

        release = asyncio.Event()
        self._active_release = release
        self.utterances.append(text)
        self.started.set()

        await release.wait()

        if self._active_release is release:
            self._active_release = None

    async def stop(self) -> None:
        """Release only the currently active utterance."""

        self.stop_count += 1

        if self._active_release is not None:
            self._active_release.set()


@pytest.mark.asyncio
async def test_speak_returns_without_waiting_for_audio() -> None:
    """Starting speech should not block command processing."""

    provider = ControlledSpeechProvider()
    manager = SpeechManager(provider=provider)

    started = await manager.speak("Long response")

    assert started

    await provider.started.wait()

    assert manager.is_speaking
    assert manager.current_text == "Long response"

    await manager.close()


@pytest.mark.asyncio
async def test_interrupt_stops_active_speech() -> None:
    """Active speech should be interruptible."""

    provider = ControlledSpeechProvider()
    manager = SpeechManager(provider=provider)

    await manager.speak("Long response")
    await provider.started.wait()

    interrupted = await manager.interrupt()

    assert interrupted
    assert not manager.is_speaking
    assert manager.current_text is None
    assert provider.stop_count == 1


@pytest.mark.asyncio
async def test_new_speech_interrupts_old_speech() -> None:
    """Replacement speech should stop the previous utterance."""

    provider = ControlledSpeechProvider()
    manager = SpeechManager(provider=provider)

    await manager.speak("First response")
    await provider.started.wait()

    provider.started.clear()

    await manager.speak("Second response")
    await provider.started.wait()

    assert provider.stop_count == 1
    assert provider.utterances == [
        "First response",
        "Second response",
    ]
    assert manager.is_speaking
    assert manager.current_text == "Second response"

    await manager.close()


@pytest.mark.asyncio
async def test_empty_speech_is_ignored() -> None:
    """Empty text should not create a playback task."""

    manager = SpeechManager(provider=ControlledSpeechProvider())

    started = await manager.speak("   ")

    assert not started
    assert not manager.is_speaking
