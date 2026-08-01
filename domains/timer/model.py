"""Timer domain model."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from uuid import UUID

from domains.timer.status import TimerStatus


@dataclass(frozen=True, slots=True)
class Timer:
    """An immutable countdown timer."""

    id: UUID
    name: str
    duration_seconds: int
    created_at: datetime
    finishes_at: datetime
    status: TimerStatus

    def remaining_seconds(self, current_time: datetime) -> int:
        """Return whole seconds remaining, never below zero."""

        remaining = self.finishes_at - current_time
        return max(0, int(remaining.total_seconds()))

    def with_status(self, status: TimerStatus) -> "Timer":
        """Return a copy of the timer with a new status."""

        return replace(self, status=status)

    @classmethod
    def create(
        cls,
        *,
        timer_id: UUID,
        name: str,
        duration_seconds: int,
        created_at: datetime,
    ) -> "Timer":
        """Create a running timer from a start time and duration."""

        return cls(
            id=timer_id,
            name=name,
            duration_seconds=duration_seconds,
            created_at=created_at,
            finishes_at=created_at + timedelta(seconds=duration_seconds),
            status=TimerStatus.RUNNING,
        )
