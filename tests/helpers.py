"""Shared test helpers."""

from datetime import datetime

from assistant.context import ApplicationContext
from core.clock import FakeClock
from core.event_bus import EventBus
from domains.timer.manager import TimerManager


def build_test_context(
    current_time: datetime | None = None,
) -> ApplicationContext:
    """Create a complete application context for tests."""

    clock = FakeClock(current_time or datetime(2026, 8, 1, 20, 0))
    event_bus = EventBus()
    timer_manager = TimerManager(
        clock=clock,
        event_bus=event_bus,
    )

    return ApplicationContext(
        clock=clock,
        event_bus=event_bus,
        timer_manager=timer_manager,
    )
