"""Integration tests for media and alarm behaviour."""

from datetime import datetime
from uuid import UUID

import pytest

from ai.stub import StubChatbotProvider
from assistant.event_handler import handle_event
from config.settings import Settings
from domains.audio.state import MediaPlaybackState
from main import build_application
from models.event import Event, EventType
from speech.silent import SilentSpeechProvider

TIMER_ID = UUID("00000000-0000-0000-0000-000000000001")


def build_test_application():
    """Build the application with deterministic test settings."""

    return build_application(
        chatbot_provider=StubChatbotProvider(),
        speech_provider=SilentSpeechProvider(),
        settings=Settings(
            chatbot_provider="stub",
            speech_provider="silent",
            wake_word_enabled=False,
        ),
    )


@pytest.mark.asyncio
async def test_application_handles_media_commands() -> None:
    """The production assistant should register the media service."""

    assistant, _, context = build_test_application()

    response = await assistant.handle_text("play music")

    assert response.text == "Playing music."
    assert context.audio_manager.media_state == MediaPlaybackState.PLAYING


@pytest.mark.asyncio
async def test_timer_alarm_pauses_and_restores_media() -> None:
    """A timer alarm should interrupt and restore playback."""

    assistant, _, context = build_test_application()

    await assistant.handle_text("play music")

    event = Event(
        type=EventType.TIMER_FINISHED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
        data={
            "timer_id": str(TIMER_ID),
            "name": "Pasta",
        },
    )

    await handle_event(
        event=event,
        context=context,
    )

    assert context.audio_manager.media_state == MediaPlaybackState.PAUSED

    response = await assistant.handle_text("stop alarm")

    assert response.text == "Pasta alarm stopped."
    assert context.audio_manager.media_state == MediaPlaybackState.PLAYING


@pytest.mark.asyncio
async def test_stopping_media_prevents_alarm_restoration() -> None:
    """Stopping media during an alarm should prevent auto-resume."""

    assistant, _, context = build_test_application()

    await assistant.handle_text("play music")

    event = Event(
        type=EventType.TIMER_FINISHED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
        data={
            "timer_id": str(TIMER_ID),
            "name": "Pasta",
        },
    )

    await handle_event(
        event=event,
        context=context,
    )

    assert context.audio_manager.media_state == MediaPlaybackState.PAUSED

    stop_media_response = await assistant.handle_text("stop music")

    assert stop_media_response.text == "Music stopped."
    assert context.audio_manager.media_state == MediaPlaybackState.STOPPED

    stop_alarm_response = await assistant.handle_text("stop alarm")

    assert stop_alarm_response.text == ("Pasta alarm stopped.")
    assert context.audio_manager.media_state == MediaPlaybackState.STOPPED
