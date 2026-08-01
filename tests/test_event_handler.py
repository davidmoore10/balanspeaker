"""Tests for application event processing."""

from datetime import datetime
from uuid import UUID

import pytest

from assistant.context import ApplicationContext
from assistant.event_handler import format_event_message, handle_event
from domains.audio.backend import SimulatedAudioBackend
from domains.audio.state import AlarmPlaybackState, MediaPlaybackState
from models.event import Event, EventType
from tests.helpers import build_test_context


TIMER_ID = UUID("00000000-0000-0000-0000-000000000001")


def get_simulated_backend(
    context: ApplicationContext,
) -> SimulatedAudioBackend:
    """Return the simulated backend from a test context."""

    backend = context.audio_manager.backend

    assert isinstance(backend, SimulatedAudioBackend)

    return backend


def test_finished_default_timer_returns_generic_message() -> None:
    """Default timers should use a generic completion message."""

    event = Event(
        type=EventType.TIMER_FINISHED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
        data={
            "timer_id": str(TIMER_ID),
            "name": "Timer",
        },
    )

    message = format_event_message(event)

    assert message == "Your timer has finished."


def test_finished_named_timer_includes_name() -> None:
    """Named timers should include the timer name."""

    event = Event(
        type=EventType.TIMER_FINISHED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
        data={
            "timer_id": str(TIMER_ID),
            "name": "Pasta",
        },
    )

    message = format_event_message(event)

    assert message == "Your Pasta timer has finished."


@pytest.mark.asyncio
async def test_handle_finished_event_starts_alarm() -> None:
    """A completed timer should create an active alarm."""

    context = build_test_context()

    event = Event(
        type=EventType.TIMER_FINISHED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
        data={
            "timer_id": str(TIMER_ID),
            "name": "Pasta",
        },
    )

    message = await handle_event(
        event=event,
        context=context,
    )

    active_alarms = context.alarm_manager.get_active_alarms()

    assert message == "Your Pasta timer has finished."
    assert len(active_alarms) == 1
    assert active_alarms[0].timer_id == TIMER_ID
    assert active_alarms[0].name == "Pasta"
    assert context.audio_manager.alarm_state == AlarmPlaybackState.ACTIVE


@pytest.mark.asyncio
async def test_finished_event_interrupts_media() -> None:
    """A timer completion should pause active media."""

    context = build_test_context()
    backend = get_simulated_backend(context)

    await context.audio_manager.play_media()
    backend.clear_operations()

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
    assert backend.operations == (
        "pause_media",
        "start_alarm",
    )


@pytest.mark.asyncio
async def test_handle_invalid_timer_id_does_not_start_alarm() -> None:
    """Malformed timer IDs should not create alarms."""

    context = build_test_context()
    backend = get_simulated_backend(context)

    event = Event(
        type=EventType.TIMER_FINISHED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
        data={
            "timer_id": "invalid-id",
            "name": "Pasta",
        },
    )

    message = await handle_event(
        event=event,
        context=context,
    )

    assert message == "Your Pasta timer has finished."
    assert context.alarm_manager.get_active_alarms() == ()
    assert backend.operations == ()


@pytest.mark.asyncio
async def test_started_alarm_event_has_no_message() -> None:
    """Alarm lifecycle events should not duplicate user messages."""

    context = build_test_context()

    event = Event(
        type=EventType.ALARM_STARTED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
        data={
            "timer_id": str(TIMER_ID),
            "name": "Pasta",
        },
    )

    message = await handle_event(
        event=event,
        context=context,
    )

    assert message is None


def test_started_timer_event_has_no_formatted_message() -> None:
    """Timer-started events are already acknowledged by the service."""

    event = Event(
        type=EventType.TIMER_STARTED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
    )

    assert format_event_message(event) is None
