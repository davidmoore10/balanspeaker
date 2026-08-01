"""Tests for the timer service."""

import pytest

from domains.timer.status import TimerStatus
from models.command import Command, CommandType
from models.event import EventType
from services.timer import TimerService
from tests.helpers import build_test_context


def test_timer_service_declares_supported_command() -> None:
    """The timer service should handle start-timer commands."""

    service = TimerService()

    assert service.supported_commands == frozenset({CommandType.START_TIMER})


@pytest.mark.asyncio
async def test_timer_service_creates_timer() -> None:
    """A valid command should create a running timer."""

    context = build_test_context()
    service = TimerService()

    command = Command(
        type=CommandType.START_TIMER,
        parameters={"duration_seconds": 30},
    )

    response = await service.execute(
        command=command,
        context=context,
    )

    timers = context.timer_manager.get_all_timers()

    assert len(timers) == 1
    assert timers[0].duration_seconds == 30
    assert timers[0].status == TimerStatus.RUNNING
    assert response.text == "Timer set for 30 seconds."


@pytest.mark.asyncio
async def test_timer_service_formats_one_second() -> None:
    """A one-second timer should use singular wording."""

    context = build_test_context()
    service = TimerService()

    command = Command(
        type=CommandType.START_TIMER,
        parameters={"duration_seconds": 1},
    )

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == "Timer set for 1 second."


@pytest.mark.asyncio
async def test_timer_service_formats_minutes() -> None:
    """Whole-minute durations should be presented as minutes."""

    context = build_test_context()
    service = TimerService()

    command = Command(
        type=CommandType.START_TIMER,
        parameters={"duration_seconds": 120},
    )

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == "Timer set for 2 minutes."


@pytest.mark.asyncio
async def test_timer_service_formats_hours() -> None:
    """Whole-hour durations should be presented as hours."""

    context = build_test_context()
    service = TimerService()

    command = Command(
        type=CommandType.START_TIMER,
        parameters={"duration_seconds": 3600},
    )

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == "Timer set for 1 hour."


@pytest.mark.asyncio
async def test_timer_service_publishes_started_event() -> None:
    """Creating a timer should publish a timer-started event."""

    context = build_test_context()
    service = TimerService()

    command = Command(
        type=CommandType.START_TIMER,
        parameters={"duration_seconds": 45},
    )

    await service.execute(
        command=command,
        context=context,
    )

    event = context.event_bus.get_nowait()

    assert event.type == EventType.TIMER_STARTED
    assert event.data["duration_seconds"] == 45


@pytest.mark.asyncio
async def test_timer_service_rejects_wrong_command() -> None:
    """The service should reject unsupported command types."""

    context = build_test_context()
    service = TimerService()

    command = Command(type=CommandType.GET_TIME)

    with pytest.raises(ValueError, match="cannot handle"):
        await service.execute(
            command=command,
            context=context,
        )


@pytest.mark.asyncio
async def test_timer_service_rejects_missing_duration() -> None:
    """The service should reject commands without a duration."""

    context = build_test_context()
    service = TimerService()

    command = Command(
        type=CommandType.START_TIMER,
        parameters={},
    )

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        await service.execute(
            command=command,
            context=context,
        )


@pytest.mark.asyncio
async def test_timer_service_rejects_zero_duration() -> None:
    """The service should reject zero-second timers."""

    context = build_test_context()
    service = TimerService()

    command = Command(
        type=CommandType.START_TIMER,
        parameters={"duration_seconds": 0},
    )

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        await service.execute(
            command=command,
            context=context,
        )


@pytest.mark.asyncio
async def test_timer_service_rejects_boolean_duration() -> None:
    """Boolean values should not be accepted as integer durations."""

    context = build_test_context()
    service = TimerService()

    command = Command(
        type=CommandType.START_TIMER,
        parameters={"duration_seconds": True},
    )

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        await service.execute(
            command=command,
            context=context,
        )
