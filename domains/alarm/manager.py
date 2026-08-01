"""Lifecycle management for active timer alarms."""

from uuid import UUID

from core.clock import Clock
from core.event_bus import EventBus
from domains.alarm.model import Alarm
from domains.alarm.status import AlarmStatus
from models.event import Event, EventType


class AlarmManager:
    """Create, inspect, and stop timer alarms."""

    def __init__(
        self,
        *,
        clock: Clock,
        event_bus: EventBus,
    ) -> None:
        self._clock = clock
        self._event_bus = event_bus
        self._alarms: dict[UUID, Alarm] = {}

    async def start_alarm(
        self,
        *,
        timer_id: UUID,
        name: str | None = None,
    ) -> Alarm:
        """Create or return an active alarm for a completed timer."""

        existing_alarm = self._alarms.get(timer_id)

        if existing_alarm is not None and existing_alarm.status == AlarmStatus.ACTIVE:
            return existing_alarm

        alarm_name = self._normalize_name(name)
        started_at = self._clock.now()

        alarm = Alarm.create(
            timer_id=timer_id,
            name=alarm_name,
            started_at=started_at,
        )

        self._alarms[timer_id] = alarm

        await self._event_bus.publish(
            Event(
                type=EventType.ALARM_STARTED,
                occurred_at=started_at,
                data={
                    "timer_id": str(alarm.timer_id),
                    "name": alarm.name,
                },
            )
        )

        return alarm

    async def stop_all(self) -> tuple[Alarm, ...]:
        """Stop every active alarm."""

        active_alarms = self.get_active_alarms()
        stopped_alarms: list[Alarm] = []

        for alarm in active_alarms:
            stopped_alarm = alarm.with_status(AlarmStatus.STOPPED)
            self._alarms[alarm.timer_id] = stopped_alarm
            stopped_alarms.append(stopped_alarm)

            await self._event_bus.publish(
                Event(
                    type=EventType.ALARM_STOPPED,
                    occurred_at=self._clock.now(),
                    data={
                        "timer_id": str(stopped_alarm.timer_id),
                        "name": stopped_alarm.name,
                    },
                )
            )

        return tuple(stopped_alarms)

    def get_active_alarms(self) -> tuple[Alarm, ...]:
        """Return all alarms that are currently active."""

        return tuple(
            alarm
            for alarm in self._alarms.values()
            if alarm.status == AlarmStatus.ACTIVE
        )

    def get_all_alarms(self) -> tuple[Alarm, ...]:
        """Return all alarms in insertion order."""

        return tuple(self._alarms.values())

    @property
    def has_active_alarm(self) -> bool:
        """Return whether at least one alarm is active."""

        return bool(self.get_active_alarms())

    @staticmethod
    def _normalize_name(name: str | None) -> str:
        """Return a valid alarm name."""

        if name is None:
            return "Timer"

        cleaned_name = name.strip()

        if not cleaned_name:
            return "Timer"

        return cleaned_name
