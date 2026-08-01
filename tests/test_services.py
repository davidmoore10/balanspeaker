"""Tests for the initial assistant services."""

from datetime import datetime

import pytest

from services.clock import ClockService
from services.greeting import GreetingService


@pytest.mark.parametrize(
    "user_text",
    [
        "hello",
        "Hi",
        "  hey  ",
        "good morning",
    ],
)
def test_greeting_service_recognises_supported_greetings(
    user_text: str,
) -> None:
    service = GreetingService()

    assert service.can_handle(user_text)


def test_greeting_service_does_not_steal_longer_request() -> None:
    service = GreetingService()

    assert not service.can_handle("hello, what time is it?")


@pytest.mark.asyncio
async def test_greeting_service_returns_response() -> None:
    service = GreetingService()

    response = await service.execute("hello")

    assert response.text == "Hello! How can I help?"


@pytest.mark.parametrize(
    "user_text",
    [
        "what time is it",
        "What's the time?",
        "tell me the time",
        "current time",
        "TIME",
    ],
)
def test_clock_service_recognises_time_requests(
    user_text: str,
) -> None:
    service = ClockService()

    assert service.can_handle(user_text)


def test_clock_service_does_not_match_timer() -> None:
    service = ClockService()

    assert not service.can_handle("timer 5")


@pytest.mark.asyncio
async def test_clock_service_returns_injected_time() -> None:
    fixed_time = datetime(2026, 8, 1, 19, 30)

    service = ClockService(
        now_provider=lambda: fixed_time,
    )

    response = await service.execute("what time is it")

    assert response.text == "The current time is 19:30."
