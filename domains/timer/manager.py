"""Creation and lifecycle management of timers."""

from collections.abc import Callable
from uuid import UUID, uuid4

from core.clock import Clock
from core.event_bus import EventBus
from domains.timer.model import Timer
from domains.timer.status import TimerStatus
from models.event import Event, EventType


class TimerNotFoundError(LookupError):
    """Raised when a timer cannot be found."""


class TimerManager:
    """Create, inspect, cancel, and expire timers."""

    def __init__(
        self,
        *,
        clock: Clock,
        event_bus: EventBus,
        id_provider: Callable[[], UUID] | None = None,
    ) -> None:
        self._clock = clock
        self._event_bus = event_bus
        self._id_provider = id_provider or uuid4
        self._timers: dict[UUID, Timer] = {}

    async def create_timer(
        self,
        *,
        duration_seconds: int,
        name: str | None = None,
    ) -> Timer:
        """Create and store a running timer."""

        if duration_seconds <= 0:
            raise ValueError("Timer duration must be greater than zero.")

        timer_id = self._id_provider()
        timer_name = self._normalize_name(name)
        created_at = self._clock.now()

        timer = Timer.create(
            timer_id=timer_id,
            name=timer_name,
            duration_seconds=duration_seconds,
            created_at=created_at,
        )

        self._timers[timer.id] = timer

        await self._event_bus.publish(
            Event(
                type=EventType.TIMER_STARTED,
                occurred_at=created_at,
                data={
                    "timer_id": str(timer.id),
                    "name": timer.name,
                    "duration_seconds": timer.duration_seconds,
                    "finishes_at": timer.finishes_at.isoformat(),
                },
            )
        )

        return timer

    async def cancel_timer(self, timer_id: UUID) -> Timer:
        """Cancel a running timer."""

        timer = self.get_timer(timer_id)

        if timer.status != TimerStatus.RUNNING:
            raise ValueError("Only running timers can be cancelled.")

        cancelled_timer = timer.with_status(TimerStatus.CANCELLED)
        self._timers[timer_id] = cancelled_timer

        await self._event_bus.publish(
            Event(
                type=EventType.TIMER_CANCELLED,
                occurred_at=self._clock.now(),
                data={
                    "timer_id": str(cancelled_timer.id),
                    "name": cancelled_timer.name,
                },
            )
        )

        return cancelled_timer

    async def check_expired(self) -> tuple[Timer, ...]:
        """Mark expired running timers as finished and publish events."""

        current_time = self._clock.now()
        finished_timers: list[Timer] = []

        for timer_id, timer in tuple(self._timers.items()):
            if timer.status != TimerStatus.RUNNING:
                continue

            if timer.finishes_at > current_time:
                continue

            finished_timer = timer.with_status(TimerStatus.FINISHED)
            self._timers[timer_id] = finished_timer
            finished_timers.append(finished_timer)

            await self._event_bus.publish(
                Event(
                    type=EventType.TIMER_FINISHED,
                    occurred_at=current_time,
                    data={
                        "timer_id": str(finished_timer.id),
                        "name": finished_timer.name,
                    },
                )
            )

        return tuple(finished_timers)

    def get_timer(self, timer_id: UUID) -> Timer:
        """Return a timer by ID."""

        try:
            return self._timers[timer_id]
        except KeyError as error:
            raise TimerNotFoundError(f"Timer '{timer_id}' was not found.") from error

    def get_all_timers(self) -> tuple[Timer, ...]:
        """Return all timers in creation order."""

        return tuple(self._timers.values())

    def get_running_timers(self) -> tuple[Timer, ...]:
        """Return only currently running timers."""

        return tuple(
            timer
            for timer in self._timers.values()
            if timer.status == TimerStatus.RUNNING
        )

    def remaining_seconds(self, timer_id: UUID) -> int:
        """Return the number of seconds remaining for a timer."""

        timer = self.get_timer(timer_id)

        if timer.status != TimerStatus.RUNNING:
            return 0

        return timer.remaining_seconds(self._clock.now())

    @staticmethod
    def _normalize_name(name: str | None) -> str:
        """Return a valid display name for a timer."""

        if name is None:
            return "Timer"

        cleaned_name = name.strip()

        if not cleaned_name:
            return "Timer"

        return cleaned_name
