"""Tests for user-facing event messages."""

from datetime import datetime

from assistant.event_handler import format_event_message
from models.event import Event, EventType


def test_finished_default_timer_returns_generic_message() -> None:
    """Default timers should use a generic completion message."""

    event = Event(
        type=EventType.TIMER_FINISHED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
        data={"name": "Timer"},
    )

    message = format_event_message(event)

    assert message == "Your timer has finished."


def test_finished_named_timer_includes_name() -> None:
    """Named timers should include the timer name."""

    event = Event(
        type=EventType.TIMER_FINISHED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
        data={"name": "Pasta"},
    )

    message = format_event_message(event)

    assert message == "Your Pasta timer has finished."


def test_finished_timer_without_name_returns_generic_message() -> None:
    """Missing timer names should produce a generic message."""

    event = Event(
        type=EventType.TIMER_FINISHED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
    )

    message = format_event_message(event)

    assert message == "Your timer has finished."


def test_started_event_has_no_user_facing_message() -> None:
    """Timer-started events are already acknowledged by the service."""

    event = Event(
        type=EventType.TIMER_STARTED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
    )

    assert format_event_message(event) is None


def test_cancelled_event_has_no_user_facing_message_yet() -> None:
    """Cancellation messaging will be handled by a later service."""

    event = Event(
        type=EventType.TIMER_CANCELLED,
        occurred_at=datetime(2026, 8, 1, 20, 0),
    )

    assert format_event_message(event) is None
