"""Timer lifecycle states."""

from enum import StrEnum


class TimerStatus(StrEnum):
    """Possible timer states."""

    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
