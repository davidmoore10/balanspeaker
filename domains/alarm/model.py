"""Alarm domain model."""

from dataclasses import dataclass, replace
from datetime import datetime
from uuid import UUID

from domains.alarm.status import AlarmStatus


@dataclass(frozen=True, slots=True)
class Alarm:
    """An alarm activated by a completed timer."""

    timer_id: UUID
    name: str
    started_at: datetime
    status: AlarmStatus

    def with_status(self, status: AlarmStatus) -> "Alarm":
        """Return a copy of the alarm with a new status."""

        return replace(self, status=status)

    @classmethod
    def create(
        cls,
        *,
        timer_id: UUID,
        name: str,
        started_at: datetime,
    ) -> "Alarm":
        """Create an active alarm."""

        return cls(
            timer_id=timer_id,
            name=name,
            started_at=started_at,
            status=AlarmStatus.ACTIVE,
        )
