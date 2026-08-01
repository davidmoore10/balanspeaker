"""Tests for alarm lifecycle management."""

from datetime import datetime
from uuid import UUID

import pytest

from core.clock import FakeClock
from core.event_bus import EventBus
from domains.alarm.manager import AlarmManager
from domains.alarm.status import AlarmStatus
from models.event import EventType


TIMER_ID = UUID("00000000-0000-0000-0000-000000000001")


def build_alarm_manager() -> tuple[
    AlarmManager,
    FakeClock,
    EventBus,
]:
    """Create an alarm manager with deterministic dependencies."""

    clock = FakeClock(datetime(2026, 8, 1, 20, 0))
    event_bus = EventBus()

    manager = AlarmManager(
        clock=clock,
        event_bus=event_bus,
    )

    return manager, clock, event_bus


@pytest.mark.asyncio
async def test_start_alarm_creates_active_alarm() -> None:
    manager, _, _ = build_alarm_manager()

    alarm = await manager.start_alarm(
        timer_id=TIMER_ID,
        name="Pasta",
    )

    assert alarm.timer_id == TIMER_ID
    assert alarm.name == "Pasta"
    assert alarm.status == AlarmStatus.ACTIVE
    assert manager.has_active_alarm


@pytest.mark.asyncio
async def test_start_alarm_publishes_event() -> None:
    manager, _, event_bus = build_alarm_manager()

    await manager.start_alarm(
        timer_id=TIMER_ID,
        name="Pasta",
    )

    event = event_bus.get_nowait()

    assert event.type == EventType.ALARM_STARTED
    assert event.data["timer_id"] == str(TIMER_ID)
    assert event.data["name"] == "Pasta"


@pytest.mark.asyncio
async def test_start_alarm_is_idempotent_for_active_timer() -> None:
    manager, _, event_bus = build_alarm_manager()

    first_alarm = await manager.start_alarm(
        timer_id=TIMER_ID,
        name="Pasta",
    )

    second_alarm = await manager.start_alarm(
        timer_id=TIMER_ID,
        name="Pasta",
    )

    assert second_alarm is first_alarm
    assert len(manager.get_active_alarms()) == 1
    assert event_bus.pending_count == 1


@pytest.mark.asyncio
async def test_stop_all_stops_active_alarm() -> None:
    manager, _, _ = build_alarm_manager()

    await manager.start_alarm(
        timer_id=TIMER_ID,
        name="Pasta",
    )

    stopped_alarms = await manager.stop_all()

    assert len(stopped_alarms) == 1
    assert stopped_alarms[0].status == AlarmStatus.STOPPED
    assert manager.get_active_alarms() == ()
    assert not manager.has_active_alarm


@pytest.mark.asyncio
async def test_stop_all_publishes_stopped_event() -> None:
    manager, _, event_bus = build_alarm_manager()

    await manager.start_alarm(
        timer_id=TIMER_ID,
        name="Pasta",
    )
    event_bus.get_nowait()

    await manager.stop_all()

    event = event_bus.get_nowait()

    assert event.type == EventType.ALARM_STOPPED
    assert event.data["timer_id"] == str(TIMER_ID)


@pytest.mark.asyncio
async def test_stop_all_returns_empty_without_active_alarm() -> None:
    manager, _, event_bus = build_alarm_manager()

    stopped_alarms = await manager.stop_all()

    assert stopped_alarms == ()
    assert event_bus.pending_count == 0
