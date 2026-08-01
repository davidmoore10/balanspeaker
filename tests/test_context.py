"""Tests for the shared application context."""

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from assistant.context import ApplicationContext
from core.clock import FakeClock
from core.event_bus import EventBus


def test_context_exposes_shared_dependencies() -> None:
    clock = FakeClock(datetime(2026, 8, 1, 20, 0))
    event_bus = EventBus()

    context = ApplicationContext(
        clock=clock,
        event_bus=event_bus,
    )

    assert context.clock is clock
    assert context.event_bus is event_bus


def test_context_dependencies_cannot_be_replaced() -> None:
    context = ApplicationContext(
        clock=FakeClock(datetime(2026, 8, 1, 20, 0)),
        event_bus=EventBus(),
    )

    with pytest.raises(FrozenInstanceError):
        context.clock = FakeClock(datetime(2026, 8, 2, 20, 0))
