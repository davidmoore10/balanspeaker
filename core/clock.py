"""Clock abstractions used by time-dependent components."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta


class Clock(ABC):
    """Interface for obtaining the current date and time."""

    @abstractmethod
    def now(self) -> datetime:
        """Return the current date and time."""


class SystemClock(Clock):
    """Clock backed by the computer's local system time."""

    def now(self) -> datetime:
        """Return the current local date and time."""

        return datetime.now()


class FakeClock(Clock):
    """Controllable clock used in automated tests."""

    def __init__(self, initial_time: datetime) -> None:
        self._current_time = initial_time

    def now(self) -> datetime:
        """Return the fake clock's current time."""

        return self._current_time

    def set(self, new_time: datetime) -> None:
        """Set the fake clock to an exact time."""

        self._current_time = new_time

    def advance(self, delta: timedelta) -> None:
        """Advance the fake clock by a duration."""

        if delta.total_seconds() < 0:
            raise ValueError("FakeClock cannot move backwards.")

        self._current_time += delta
