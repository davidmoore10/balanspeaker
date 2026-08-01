"""Shared test helpers."""

from datetime import datetime

from assistant.context import ApplicationContext
from core.clock import FakeClock
from core.event_bus import EventBus
from domains.alarm.manager import AlarmManager
from domains.audio.backend import SimulatedAudioBackend
from domains.audio.manager import AudioManager
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

    alarm_manager = AlarmManager(
        clock=clock,
        event_bus=event_bus,
    )

    audio_backend = SimulatedAudioBackend()

    audio_manager = AudioManager(
        backend=audio_backend,
    )

    return ApplicationContext(
        clock=clock,
        event_bus=event_bus,
        timer_manager=timer_manager,
        alarm_manager=alarm_manager,
        audio_manager=audio_manager,
    )
