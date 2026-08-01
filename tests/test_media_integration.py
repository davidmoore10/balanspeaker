"""Integration tests for media and alarm interruption."""

from datetime import datetime
from uuid import UUID

import pytest

from assistant.event_handler import handle_event
from domains.audio.backend import SimulatedAudioBackend
from domains.audio.state import MediaPlaybackState
from main import build_application
from models.event import Event, EventType


TIMER_ID = UUID("00000000-0000-0000-0000-000000000001")


@pytest.mark.asyncio
async def test_application_handles_media_commands() -> None:
    """The production assistant should register the media service."""

    assistant, _, context = build_application()

    response = await assistant.handle_text("play music")

    assert response.text == "Playing music."
    assert context.audio_manager.media_state == MediaPlaybackState.PLAYING


@pytest.mark.asyncio
async def test_timer_alarm_pauses_and_restores_media() -> None:
    """A timer alarm should interrupt and restore playback."""

    assistant, _, context = build_application()

    backend = context.audio_manager.backend

    assert isinstance(backend, SimulatedAudioBackend)

    play_response = await assistant.handle_text("play music")

    assert play_response.text == "Playing music."

    backend.clear_operations()

    timer_event = Event(
        type=EventType.TIMER_FINISHED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
        data={
            "timer_id": str(TIMER_ID),
            "name": "Pasta",
        },
    )

    await handle_event(
        event=timer_event,
        context=context,
    )

    assert context.audio_manager.media_state == MediaPlaybackState.PAUSED
    assert backend.operations == (
        "pause_media",
        "start_alarm",
    )

    backend.clear_operations()

    stop_response = await assistant.handle_text("stop alarm")

    assert stop_response.text == "Pasta alarm stopped."
    assert context.audio_manager.media_state == MediaPlaybackState.PLAYING
    assert backend.operations == (
        "stop_alarm",
        "play_media",
    )


@pytest.mark.asyncio
async def test_manual_pause_prevents_alarm_restoration() -> None:
    """A manual pause during an alarm should prevent auto-resume."""

    assistant, _, context = build_application()

    backend = context.audio_manager.backend

    assert isinstance(backend, SimulatedAudioBackend)

    await assistant.handle_text("play music")

    timer_event = Event(
        type=EventType.TIMER_FINISHED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
        data={
            "timer_id": str(TIMER_ID),
            "name": "Timer",
        },
    )

    await handle_event(
        event=timer_event,
        context=context,
    )

    pause_response = await assistant.handle_text("pause music")

    assert pause_response.text == "No music is currently playing."

    await context.audio_manager.pause_media()
    backend.clear_operations()

    await assistant.handle_text("stop alarm")

    assert context.audio_manager.media_state == MediaPlaybackState.PAUSED
    assert backend.operations == ("stop_alarm",)
