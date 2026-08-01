"""Tests for the background timer scheduler."""

from datetime import timedelta

import pytest

from domains.timer.scheduler import TimerScheduler
from domains.timer.status import TimerStatus
from models.event import EventType
from tests.helpers import build_test_context


def test_scheduler_rejects_non_positive_interval() -> None:
    """The polling interval must be greater than zero."""

    context = build_test_context()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        TimerScheduler(
            timer_manager=context.timer_manager,
            poll_interval_seconds=0,
        )


def test_scheduler_exposes_poll_interval() -> None:
    """The configured polling interval should be available."""

    context = build_test_context()

    scheduler = TimerScheduler(
        timer_manager=context.timer_manager,
        poll_interval_seconds=0.5,
    )

    assert scheduler.poll_interval_seconds == 0.5


@pytest.mark.asyncio
async def test_run_once_does_not_finish_active_timer() -> None:
    """A timer should remain active before its finish time."""

    context = build_test_context()

    timer = await context.timer_manager.create_timer(
        duration_seconds=10,
    )

    context.event_bus.get_nowait()

    scheduler = TimerScheduler(
        timer_manager=context.timer_manager,
    )

    finished_timers = await scheduler.run_once()

    assert finished_timers == ()
    assert context.timer_manager.get_timer(timer.id).status == TimerStatus.RUNNING


@pytest.mark.asyncio
async def test_run_once_finishes_expired_timer() -> None:
    """A timer should finish after the fake clock reaches its deadline."""

    context = build_test_context()

    timer = await context.timer_manager.create_timer(
        duration_seconds=10,
    )

    context.event_bus.get_nowait()
    context.clock.advance(timedelta(seconds=10))

    scheduler = TimerScheduler(
        timer_manager=context.timer_manager,
    )

    finished_timers = await scheduler.run_once()

    assert len(finished_timers) == 1
    assert finished_timers[0].id == timer.id
    assert finished_timers[0].status == TimerStatus.FINISHED


@pytest.mark.asyncio
async def test_run_once_publishes_finished_event() -> None:
    """Finishing a timer should publish a timer-finished event."""

    context = build_test_context()

    await context.timer_manager.create_timer(
        duration_seconds=10,
        name="Tea",
    )

    context.event_bus.get_nowait()
    context.clock.advance(timedelta(seconds=10))

    scheduler = TimerScheduler(
        timer_manager=context.timer_manager,
    )

    await scheduler.run_once()

    event = context.event_bus.get_nowait()

    assert event.type == EventType.TIMER_FINISHED
    assert event.data["name"] == "Tea"
