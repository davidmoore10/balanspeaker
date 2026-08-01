"""Tests for asynchronous event publication."""

import asyncio
from datetime import datetime

import pytest

from core.event_bus import EventBus
from models.event import Event, EventType


def build_event() -> Event:
    """Create a deterministic event for tests."""

    return Event(
        type=EventType.TIMER_STARTED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
        data={"timer_id": "timer-1"},
    )


@pytest.mark.asyncio
async def test_publish_adds_event_to_queue() -> None:
    event_bus = EventBus()
    event = build_event()

    await event_bus.publish(event)

    assert event_bus.pending_count == 1


@pytest.mark.asyncio
async def test_get_returns_published_event() -> None:
    event_bus = EventBus()
    event = build_event()
    await event_bus.publish(event)

    received_event = await event_bus.get()

    assert received_event is event


@pytest.mark.asyncio
async def test_get_nowait_returns_published_event() -> None:
    event_bus = EventBus()
    event = build_event()
    await event_bus.publish(event)

    received_event = event_bus.get_nowait()

    assert received_event is event


def test_get_nowait_raises_when_queue_is_empty() -> None:
    event_bus = EventBus()

    with pytest.raises(asyncio.QueueEmpty):
        event_bus.get_nowait()


@pytest.mark.asyncio
async def test_join_completes_after_task_done() -> None:
    event_bus = EventBus()
    event = build_event()
    await event_bus.publish(event)

    await event_bus.get()
    event_bus.task_done()

    await asyncio.wait_for(
        event_bus.join(),
        timeout=0.1,
    )


@pytest.mark.asyncio
async def test_events_preserve_publication_order() -> None:
    event_bus = EventBus()

    first_event = Event(
        type=EventType.TIMER_STARTED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
    )
    second_event = Event(
        type=EventType.TIMER_FINISHED,
        occurred_at=datetime(2026, 8, 1, 20, 1),
    )

    await event_bus.publish(first_event)
    await event_bus.publish(second_event)

    assert await event_bus.get() is first_event
    assert await event_bus.get() is second_event
