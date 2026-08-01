"""Tests for the initial assistant services."""

from datetime import datetime

import pytest

from assistant.context import ApplicationContext
from core.clock import FakeClock
from core.event_bus import EventBus
from models.command import Command, CommandType
from services.clock import ClockService
from services.greeting import GreetingService


def build_context(
    current_time: datetime | None = None,
) -> ApplicationContext:
    """Create an application context for service tests."""

    return ApplicationContext(
        clock=FakeClock(current_time or datetime(2026, 8, 1, 20, 0)),
        event_bus=EventBus(),
    )


def test_greeting_service_declares_supported_command() -> None:
    service = GreetingService()

    assert service.supported_commands == frozenset({CommandType.GREET})


@pytest.mark.asyncio
async def test_greeting_service_returns_response() -> None:
    service = GreetingService()
    command = Command(type=CommandType.GREET)

    response = await service.execute(
        command=command,
        context=build_context(),
    )

    assert response.text == "Hello! How can I help?"


@pytest.mark.asyncio
async def test_greeting_service_rejects_wrong_command() -> None:
    service = GreetingService()
    command = Command(type=CommandType.GET_TIME)

    with pytest.raises(ValueError, match="cannot handle"):
        await service.execute(
            command=command,
            context=build_context(),
        )


def test_clock_service_declares_supported_command() -> None:
    service = ClockService()

    assert service.supported_commands == frozenset({CommandType.GET_TIME})


@pytest.mark.asyncio
async def test_clock_service_returns_context_time() -> None:
    service = ClockService()
    command = Command(type=CommandType.GET_TIME)
    context = build_context(current_time=datetime(2026, 8, 1, 19, 30))

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == "The current time is 19:30."


@pytest.mark.asyncio
async def test_clock_service_rejects_wrong_command() -> None:
    service = ClockService()
    command = Command(type=CommandType.GREET)

    with pytest.raises(ValueError, match="cannot handle"):
        await service.execute(
            command=command,
            context=build_context(),
        )
