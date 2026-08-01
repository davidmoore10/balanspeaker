"""Tests for the timer service."""

from datetime import timedelta

import pytest

from domains.timer.status import TimerStatus
from models.command import Command, CommandType
from models.event import EventType
from services.timer import TimerService
from tests.helpers import build_test_context


def test_timer_service_declares_supported_commands() -> None:
    service = TimerService()

    assert service.supported_commands == frozenset(
        {
            CommandType.START_TIMER,
            CommandType.LIST_TIMERS,
            CommandType.CANCEL_TIMER,
        }
    )


@pytest.mark.asyncio
async def test_timer_service_creates_timer() -> None:
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
async def test_timer_service_creates_named_timer() -> None:
    context = build_test_context()
    service = TimerService()

    command = Command(
        type=CommandType.START_TIMER,
        parameters={
            "duration_seconds": 300,
            "name": "pasta",
        },
    )

    response = await service.execute(
        command=command,
        context=context,
    )

    timer = context.timer_manager.get_running_timers()[0]

    assert timer.name == "pasta"
    assert response.text == "Pasta timer set for 5 minutes."


@pytest.mark.asyncio
async def test_timer_service_formats_combined_duration() -> None:
    context = build_test_context()
    service = TimerService()

    command = Command(
        type=CommandType.START_TIMER,
        parameters={"duration_seconds": 65},
    )

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == "Timer set for 1 minute and 5 seconds."


@pytest.mark.asyncio
async def test_timer_service_publishes_started_event() -> None:
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
async def test_list_timers_returns_no_active_timers() -> None:
    context = build_test_context()
    service = TimerService()

    command = Command(type=CommandType.LIST_TIMERS)

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == "You have no active timers."


@pytest.mark.asyncio
async def test_list_timers_returns_single_timer() -> None:
    context = build_test_context()
    service = TimerService()

    await context.timer_manager.create_timer(
        duration_seconds=90,
        name="pasta",
    )

    context.clock.advance(timedelta(seconds=25))

    command = Command(type=CommandType.LIST_TIMERS)

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == ("The pasta timer has 1 minute and 5 seconds remaining.")


@pytest.mark.asyncio
async def test_list_timers_returns_multiple_timers() -> None:
    context = build_test_context()
    service = TimerService()

    await context.timer_manager.create_timer(
        duration_seconds=30,
        name="eggs",
    )
    await context.timer_manager.create_timer(
        duration_seconds=120,
        name="pasta",
    )

    command = Command(type=CommandType.LIST_TIMERS)

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == (
        "You have 2 active timers: "
        "The eggs timer has 30 seconds remaining.; "
        "The pasta timer has 2 minutes remaining."
    )


@pytest.mark.asyncio
async def test_cancel_only_active_timer() -> None:
    context = build_test_context()
    service = TimerService()

    timer = await context.timer_manager.create_timer(
        duration_seconds=30,
    )

    command = Command(type=CommandType.CANCEL_TIMER)

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == "Timer cancelled."
    assert context.timer_manager.get_timer(timer.id).status == TimerStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_named_timer() -> None:
    context = build_test_context()
    service = TimerService()

    pasta_timer = await context.timer_manager.create_timer(
        duration_seconds=300,
        name="pasta",
    )
    await context.timer_manager.create_timer(
        duration_seconds=60,
        name="tea",
    )

    command = Command(
        type=CommandType.CANCEL_TIMER,
        parameters={"name": "Pasta"},
    )

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == "Pasta timer cancelled."
    assert (
        context.timer_manager.get_timer(pasta_timer.id).status == TimerStatus.CANCELLED
    )


@pytest.mark.asyncio
async def test_cancel_without_name_rejects_multiple_timers() -> None:
    context = build_test_context()
    service = TimerService()

    await context.timer_manager.create_timer(
        duration_seconds=300,
        name="pasta",
    )
    await context.timer_manager.create_timer(
        duration_seconds=60,
        name="tea",
    )

    command = Command(type=CommandType.CANCEL_TIMER)

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == (
        "You have more than one active timer. Please specify which timer to cancel."
    )


@pytest.mark.asyncio
async def test_cancel_unknown_named_timer() -> None:
    context = build_test_context()
    service = TimerService()

    await context.timer_manager.create_timer(
        duration_seconds=300,
        name="pasta",
    )

    command = Command(
        type=CommandType.CANCEL_TIMER,
        parameters={"name": "tea"},
    )

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == ("I couldn't find an active timer named 'tea'.")


@pytest.mark.asyncio
async def test_cancel_when_no_timers_are_active() -> None:
    context = build_test_context()
    service = TimerService()

    command = Command(type=CommandType.CANCEL_TIMER)

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == "You have no active timers."


@pytest.mark.asyncio
async def test_timer_service_rejects_wrong_command() -> None:
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
