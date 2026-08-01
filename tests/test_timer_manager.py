"""Tests for timer creation and lifecycle management."""

from datetime import datetime, timedelta
from uuid import UUID

import pytest

from core.clock import FakeClock
from core.event_bus import EventBus
from domains.timer.manager import TimerManager, TimerNotFoundError
from domains.timer.status import TimerStatus
from models.event import EventType


TIMER_ID_1 = UUID("00000000-0000-0000-0000-000000000001")
TIMER_ID_2 = UUID("00000000-0000-0000-0000-000000000002")


def build_manager(
    timer_ids: list[UUID] | None = None,
) -> tuple[TimerManager, FakeClock, EventBus]:
    """Build a timer manager with deterministic dependencies."""

    clock = FakeClock(datetime(2026, 8, 1, 20, 0))
    event_bus = EventBus()

    available_ids = iter(timer_ids or [TIMER_ID_1])

    manager = TimerManager(
        clock=clock,
        event_bus=event_bus,
        id_provider=lambda: next(available_ids),
    )

    return manager, clock, event_bus


@pytest.mark.asyncio
async def test_create_timer_stores_running_timer() -> None:
    manager, _, _ = build_manager()

    timer = await manager.create_timer(
        duration_seconds=300,
        name="Pasta",
    )

    assert timer.id == TIMER_ID_1
    assert timer.name == "Pasta"
    assert timer.duration_seconds == 300
    assert timer.status == TimerStatus.RUNNING
    assert manager.get_timer(TIMER_ID_1) is timer


@pytest.mark.asyncio
async def test_create_timer_calculates_finish_time() -> None:
    manager, _, _ = build_manager()

    timer = await manager.create_timer(
        duration_seconds=300,
    )

    assert timer.created_at == datetime(2026, 8, 1, 20, 0)
    assert timer.finishes_at == datetime(2026, 8, 1, 20, 5)


@pytest.mark.asyncio
async def test_create_timer_uses_default_name() -> None:
    manager, _, _ = build_manager()

    timer = await manager.create_timer(
        duration_seconds=30,
        name="   ",
    )

    assert timer.name == "Timer"


@pytest.mark.asyncio
async def test_create_timer_rejects_non_positive_duration() -> None:
    manager, _, _ = build_manager()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        await manager.create_timer(duration_seconds=0)


@pytest.mark.asyncio
async def test_create_timer_publishes_started_event() -> None:
    manager, _, event_bus = build_manager()

    await manager.create_timer(
        duration_seconds=60,
        name="Eggs",
    )

    event = event_bus.get_nowait()

    assert event.type == EventType.TIMER_STARTED
    assert event.data["timer_id"] == str(TIMER_ID_1)
    assert event.data["name"] == "Eggs"
    assert event.data["duration_seconds"] == 60


@pytest.mark.asyncio
async def test_manager_supports_multiple_timers() -> None:
    manager, _, _ = build_manager(timer_ids=[TIMER_ID_1, TIMER_ID_2])

    first_timer = await manager.create_timer(
        duration_seconds=60,
        name="Eggs",
    )
    second_timer = await manager.create_timer(
        duration_seconds=300,
        name="Pasta",
    )

    assert manager.get_all_timers() == (
        first_timer,
        second_timer,
    )
    assert manager.get_running_timers() == (
        first_timer,
        second_timer,
    )


@pytest.mark.asyncio
async def test_remaining_seconds_decreases_with_clock() -> None:
    manager, clock, _ = build_manager()

    await manager.create_timer(duration_seconds=60)

    clock.advance(timedelta(seconds=15))

    assert manager.remaining_seconds(TIMER_ID_1) == 45


@pytest.mark.asyncio
async def test_remaining_seconds_never_goes_below_zero() -> None:
    manager, clock, _ = build_manager()

    await manager.create_timer(duration_seconds=10)

    clock.advance(timedelta(seconds=20))

    assert manager.remaining_seconds(TIMER_ID_1) == 0


@pytest.mark.asyncio
async def test_check_expired_finishes_due_timer() -> None:
    manager, clock, _ = build_manager()

    await manager.create_timer(
        duration_seconds=60,
        name="Egg timer",
    )

    clock.advance(timedelta(seconds=60))

    finished_timers = await manager.check_expired()

    assert len(finished_timers) == 1
    assert finished_timers[0].status == TimerStatus.FINISHED
    assert manager.get_timer(TIMER_ID_1).status == TimerStatus.FINISHED


@pytest.mark.asyncio
async def test_check_expired_ignores_timer_not_yet_due() -> None:
    manager, clock, _ = build_manager()

    timer = await manager.create_timer(duration_seconds=60)

    clock.advance(timedelta(seconds=59))

    finished_timers = await manager.check_expired()

    assert finished_timers == ()
    assert manager.get_timer(timer.id).status == TimerStatus.RUNNING


@pytest.mark.asyncio
async def test_check_expired_publishes_finished_event() -> None:
    manager, clock, event_bus = build_manager()

    await manager.create_timer(
        duration_seconds=10,
        name="Tea",
    )

    event_bus.get_nowait()

    clock.advance(timedelta(seconds=10))
    await manager.check_expired()

    event = event_bus.get_nowait()

    assert event.type == EventType.TIMER_FINISHED
    assert event.data["timer_id"] == str(TIMER_ID_1)
    assert event.data["name"] == "Tea"


@pytest.mark.asyncio
async def test_check_expired_only_finishes_due_timers() -> None:
    manager, clock, _ = build_manager(timer_ids=[TIMER_ID_1, TIMER_ID_2])

    await manager.create_timer(
        duration_seconds=30,
        name="Short",
    )
    await manager.create_timer(
        duration_seconds=60,
        name="Long",
    )

    clock.advance(timedelta(seconds=30))

    finished_timers = await manager.check_expired()

    assert tuple(timer.id for timer in finished_timers) == (TIMER_ID_1,)
    assert manager.get_timer(TIMER_ID_1).status == TimerStatus.FINISHED
    assert manager.get_timer(TIMER_ID_2).status == TimerStatus.RUNNING


@pytest.mark.asyncio
async def test_finished_timer_is_not_finished_twice() -> None:
    manager, clock, event_bus = build_manager()

    await manager.create_timer(duration_seconds=10)
    event_bus.get_nowait()

    clock.advance(timedelta(seconds=10))

    first_result = await manager.check_expired()
    first_event = event_bus.get_nowait()

    second_result = await manager.check_expired()

    assert len(first_result) == 1
    assert first_event.type == EventType.TIMER_FINISHED
    assert second_result == ()
    assert event_bus.pending_count == 0


@pytest.mark.asyncio
async def test_cancel_timer_changes_status() -> None:
    manager, _, _ = build_manager()

    await manager.create_timer(
        duration_seconds=60,
        name="Pasta",
    )

    cancelled_timer = await manager.cancel_timer(TIMER_ID_1)

    assert cancelled_timer.status == TimerStatus.CANCELLED
    assert manager.get_timer(TIMER_ID_1).status == TimerStatus.CANCELLED
    assert manager.get_running_timers() == ()


@pytest.mark.asyncio
async def test_cancel_timer_publishes_event() -> None:
    manager, _, event_bus = build_manager()

    await manager.create_timer(
        duration_seconds=60,
        name="Pasta",
    )
    event_bus.get_nowait()

    await manager.cancel_timer(TIMER_ID_1)

    event = event_bus.get_nowait()

    assert event.type == EventType.TIMER_CANCELLED
    assert event.data["timer_id"] == str(TIMER_ID_1)
    assert event.data["name"] == "Pasta"


@pytest.mark.asyncio
async def test_cancelled_timer_does_not_expire() -> None:
    manager, clock, event_bus = build_manager()

    await manager.create_timer(duration_seconds=10)
    event_bus.get_nowait()

    await manager.cancel_timer(TIMER_ID_1)
    event_bus.get_nowait()

    clock.advance(timedelta(seconds=20))

    finished_timers = await manager.check_expired()

    assert finished_timers == ()
    assert manager.get_timer(TIMER_ID_1).status == TimerStatus.CANCELLED
    assert event_bus.pending_count == 0


@pytest.mark.asyncio
async def test_finished_timer_cannot_be_cancelled() -> None:
    manager, clock, _ = build_manager()

    await manager.create_timer(duration_seconds=10)
    clock.advance(timedelta(seconds=10))
    await manager.check_expired()

    with pytest.raises(
        ValueError,
        match="Only running timers",
    ):
        await manager.cancel_timer(TIMER_ID_1)


def test_unknown_timer_raises_not_found_error() -> None:
    manager, _, _ = build_manager()

    with pytest.raises(
        TimerNotFoundError,
        match="was not found",
    ):
        manager.get_timer(TIMER_ID_1)
