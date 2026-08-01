"""Asynchronous publication and consumption of application events."""

import asyncio

from models.event import Event


class EventBus:
    """Queue-based communication channel for application events."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue()

    async def publish(self, event: Event) -> None:
        """Add an event to the queue."""

        await self._queue.put(event)

    async def get(self) -> Event:
        """Wait for and return the next available event."""

        return await self._queue.get()

    def get_nowait(self) -> Event:
        """Return the next event without waiting."""

        return self._queue.get_nowait()

    def task_done(self) -> None:
        """Mark the most recently retrieved event as processed."""

        self._queue.task_done()

    async def join(self) -> None:
        """Wait until all queued events have been processed."""

        await self._queue.join()

    @property
    def pending_count(self) -> int:
        """Return the number of events currently waiting."""

        return self._queue.qsize()
