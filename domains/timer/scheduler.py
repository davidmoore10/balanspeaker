"""Background scheduling for timer expiry checks."""

import asyncio

from domains.timer.manager import TimerManager
from domains.timer.model import Timer


class TimerScheduler:
    """Periodically check the timer manager for expired timers."""

    def __init__(
        self,
        *,
        timer_manager: TimerManager,
        poll_interval_seconds: float = 0.25,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("Timer scheduler poll interval must be greater than zero.")

        self._timer_manager = timer_manager
        self._poll_interval_seconds = poll_interval_seconds

    @property
    def poll_interval_seconds(self) -> float:
        """Return the scheduler polling interval."""

        return self._poll_interval_seconds

    async def run_once(self) -> tuple[Timer, ...]:
        """Perform one timer expiry check."""

        return await self._timer_manager.check_expired()

    async def run(self) -> None:
        """Continuously check for expired timers."""

        while True:
            await self.run_once()
            await asyncio.sleep(self._poll_interval_seconds)
