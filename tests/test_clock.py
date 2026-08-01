"""Tests for application clocks."""

from datetime import datetime, timedelta

import pytest

from core.clock import FakeClock, SystemClock


def test_fake_clock_returns_initial_time() -> None:
    initial_time = datetime(2026, 8, 1, 20, 0)
    clock = FakeClock(initial_time)

    assert clock.now() == initial_time


def test_fake_clock_can_be_set() -> None:
    clock = FakeClock(datetime(2026, 8, 1, 20, 0))
    new_time = datetime(2026, 8, 2, 9, 30)

    clock.set(new_time)

    assert clock.now() == new_time


def test_fake_clock_can_advance() -> None:
    clock = FakeClock(datetime(2026, 8, 1, 20, 0))

    clock.advance(timedelta(minutes=5, seconds=30))

    assert clock.now() == datetime(2026, 8, 1, 20, 5, 30)


def test_fake_clock_rejects_negative_advance() -> None:
    clock = FakeClock(datetime(2026, 8, 1, 20, 0))

    with pytest.raises(ValueError, match="cannot move backwards"):
        clock.advance(timedelta(seconds=-1))


def test_system_clock_returns_datetime() -> None:
    clock = SystemClock()

    assert isinstance(clock.now(), datetime)
