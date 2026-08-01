"""Tests for alarm control commands."""

from uuid import UUID

import pytest

from domains.audio.backend import SimulatedAudioBackend
from domains.audio.state import MediaPlaybackState
from models.command import Command, CommandType
from services.alarm import AlarmService
from tests.helpers import build_test_context


TIMER_ID = UUID("00000000-0000-0000-0000-000000000001")


def test_alarm_service_declares_supported_command() -> None:
    """The service should handle stop-alarm commands."""

    service = AlarmService()

    assert service.supported_commands == frozenset({CommandType.STOP_ALARM})


@pytest.mark.asyncio
async def test_stop_alarm_without_active_alarm() -> None:
    """Stopping without an active alarm should return a clear response."""

    context = build_test_context()
    service = AlarmService()
    command = Command(type=CommandType.STOP_ALARM)

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == "There is no active alarm."


@pytest.mark.asyncio
async def test_stop_default_alarm() -> None:
    """A default timer alarm should use generic wording."""

    context = build_test_context()
    service = AlarmService()

    await context.alarm_manager.start_alarm(
        timer_id=TIMER_ID,
        name="Timer",
    )
    await context.audio_manager.start_alarm()

    command = Command(type=CommandType.STOP_ALARM)

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == "Alarm stopped."
    assert not context.alarm_manager.has_active_alarm
    assert not context.audio_manager.alarm_is_active


@pytest.mark.asyncio
async def test_stop_named_alarm() -> None:
    """A named timer alarm should be identified in the response."""

    context = build_test_context()
    service = AlarmService()

    await context.alarm_manager.start_alarm(
        timer_id=TIMER_ID,
        name="pasta",
    )
    await context.audio_manager.start_alarm()

    command = Command(type=CommandType.STOP_ALARM)

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == "Pasta alarm stopped."


@pytest.mark.asyncio
async def test_stopping_alarm_restores_interrupted_media() -> None:
    """Media paused by an alarm should resume when it stops."""

    context = build_test_context()
    service = AlarmService()

    backend = context.audio_manager.backend

    assert isinstance(backend, SimulatedAudioBackend)

    await context.audio_manager.play_media()

    await context.alarm_manager.start_alarm(
        timer_id=TIMER_ID,
        name="Timer",
    )
    await context.audio_manager.start_alarm()

    backend.clear_operations()

    command = Command(type=CommandType.STOP_ALARM)

    await service.execute(
        command=command,
        context=context,
    )

    assert context.audio_manager.media_state == MediaPlaybackState.PLAYING
    assert backend.operations == (
        "stop_alarm",
        "play_media",
    )


@pytest.mark.asyncio
async def test_alarm_service_rejects_wrong_command() -> None:
    """The service should reject unsupported command types."""

    context = build_test_context()
    service = AlarmService()
    command = Command(type=CommandType.GET_TIME)

    with pytest.raises(ValueError, match="cannot handle"):
        await service.execute(
            command=command,
            context=context,
        )
