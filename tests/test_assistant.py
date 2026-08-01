"""Tests for the core assistant."""

from datetime import datetime

import pytest

from assistant.assistant import Assistant
from assistant.context import ApplicationContext
from assistant.parser import RuleBasedCommandParser
from assistant.registry import ServiceRegistry
from core.clock import FakeClock
from core.event_bus import EventBus
from models.command import Command, CommandType
from models.response import AssistantResponse
from services.base import Service


class GreetingTestService(Service):
    """A greeting service used in assistant tests."""

    @property
    def name(self) -> str:
        return "greeting-test"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        return frozenset({CommandType.GREET})

    async def execute(
        self,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        return AssistantResponse(text="Handled greeting.")


class FailingTimeService(Service):
    """A time service that deliberately raises an exception."""

    @property
    def name(self) -> str:
        return "failing-time"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        return frozenset({CommandType.GET_TIME})

    async def execute(
        self,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        raise RuntimeError("Simulated service failure")


def build_context() -> ApplicationContext:
    """Create a test application context."""

    return ApplicationContext(
        clock=FakeClock(datetime(2026, 8, 1, 20, 0)),
        event_bus=EventBus(),
    )


def build_test_assistant(
    include_time_service: bool = False,
) -> Assistant:
    """Build an assistant configured for tests."""

    registry = ServiceRegistry()
    registry.register(GreetingTestService())

    if include_time_service:
        registry.register(FailingTimeService())

    return Assistant(
        registry=registry,
        parser=RuleBasedCommandParser(),
        context=build_context(),
        name="Balanspeaker",
    )


def test_default_assistant_name() -> None:
    assistant = Assistant(
        registry=ServiceRegistry(),
        parser=RuleBasedCommandParser(),
        context=build_context(),
    )

    assert assistant.name == "Balanspeaker"


def test_empty_assistant_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Assistant name cannot be empty"):
        Assistant(
            registry=ServiceRegistry(),
            parser=RuleBasedCommandParser(),
            context=build_context(),
            name="   ",
        )


def test_start_message_contains_assistant_name() -> None:
    assistant = build_test_assistant()

    response = assistant.start_message()

    assert response.text == "Balanspeaker is ready. Type 'exit' to quit."


@pytest.mark.asyncio
async def test_handle_text_executes_matching_command() -> None:
    assistant = build_test_assistant()

    response = await assistant.handle_text("hello")

    assert response.text == "Handled greeting."


@pytest.mark.asyncio
async def test_handle_empty_text_returns_prompt() -> None:
    assistant = build_test_assistant()

    response = await assistant.handle_text("   ")

    assert response.text == "Please enter a command."


@pytest.mark.asyncio
async def test_unknown_request_returns_fallback() -> None:
    assistant = build_test_assistant()

    response = await assistant.handle_text("play some music")

    assert response.text == "I don't know how to handle that yet."


@pytest.mark.asyncio
async def test_known_command_without_service_returns_unavailable() -> None:
    assistant = build_test_assistant()

    response = await assistant.handle_text("what time is it")

    assert response.text == "That capability is not currently available."


@pytest.mark.asyncio
async def test_service_failure_returns_controlled_error() -> None:
    assistant = build_test_assistant(include_time_service=True)

    response = await assistant.handle_text("what time is it")

    assert response.text == ("Something went wrong while handling that request.")


@pytest.mark.asyncio
async def test_timer_without_registered_service_is_unavailable() -> None:
    assistant = build_test_assistant()

    response = await assistant.handle_text("timer 5")

    assert response.text == "That capability is not currently available."


@pytest.mark.asyncio
async def test_timer_without_duration_returns_parser_error() -> None:
    assistant = build_test_assistant()

    response = await assistant.handle_text("set a timer")

    assert response.text == "Please specify how long the timer should run."
