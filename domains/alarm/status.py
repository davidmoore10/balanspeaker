"""Alarm lifecycle states."""

from enum import StrEnum


class AlarmStatus(StrEnum):
    """Possible alarm states."""

    ACTIVE = "active"
    STOPPED = "stopped"
