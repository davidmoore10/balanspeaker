"""Shared application dependencies available to assistant services."""

from dataclasses import dataclass

from core.clock import Clock
from core.event_bus import EventBus
from domains.alarm.manager import AlarmManager
from domains.timer.manager import TimerManager


@dataclass(frozen=True, slots=True)
class ApplicationContext:
    """Dependencies and managers shared across the application."""

    clock: Clock
    event_bus: EventBus
    timer_manager: TimerManager
    alarm_manager: AlarmManager
