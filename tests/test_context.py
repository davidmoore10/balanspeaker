"""Tests for the shared application context."""

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from core.clock import FakeClock
from tests.helpers import build_test_context


def test_context_exposes_shared_dependencies() -> None:
    """The context should expose all application managers."""

    context = build_test_context(current_time=datetime(2026, 8, 1, 20, 0))

    assert isinstance(context.clock, FakeClock)
    assert context.event_bus is not None
    assert context.timer_manager is not None
    assert context.alarm_manager is not None
    assert context.audio_manager is not None
    assert context.conversation_manager is not None
    assert context.chatbot_provider is not None
    assert context.speech_provider is not None


def test_context_dependencies_cannot_be_replaced() -> None:
    """The context itself should remain immutable."""

    context = build_test_context()

    with pytest.raises(FrozenInstanceError):
        context.clock = FakeClock(datetime(2026, 8, 2, 20, 0))
