"""Events published by assistant components."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4


class EventType(StrEnum):
    """Events understood by the application."""

    TIMER_STARTED = "timer_started"
    TIMER_FINISHED = "timer_finished"
    TIMER_CANCELLED = "timer_cancelled"

    ALARM_STARTED = "alarm_started"
    ALARM_STOPPED = "alarm_stopped"


@dataclass(frozen=True, slots=True)
class Event:
    """A notification that something occurred in the application."""

    type: EventType
    occurred_at: datetime
    data: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
