"""Tests for the core assistant."""

import pytest

from assistant.assistant import Assistant
from assistant.context import ApplicationContext
from assistant.parser import RuleBasedCommandParser
from assistant.registry import ServiceRegistry
from models.command import Command, CommandType
from models.response import AssistantResponse
from services.base import Service
from tests.helpers import build_test_context


class GreetingTestService(Service):
    """A greeting service used in assistant tests."""

    @property
    def name(self) -> str:
        """Return the service name."""

        return "greeting-test"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        """Return supported command types."""

        return frozenset({CommandType.GREET})

    async def execute(
        self,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Return a successful test response."""

        return AssistantResponse(text="Handled greeting.")


class FailingTimeService(Service):
    """A time service that deliberately raises an exception."""

    @property
    def name(self) -> str:
        """Return the service name."""

        return "failing-time"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        """Return supported command types."""

        return frozenset({CommandType.GET_TIME})

    async def execute(
        self,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Simulate a service failure."""

        raise RuntimeError("Simulated service failure")


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
        context=build_test_context(),
        name="Balanspeaker",
    )


def test_default_assistant_name() -> None:
    """The default assistant name should be Balanspeaker."""

    assistant = Assistant(
        registry=ServiceRegistry(),
        parser=RuleBasedCommandParser(),
        context=build_test_context(),
    )

    assert assistant.name == "Balanspeaker"


def test_empty_assistant_name_is_rejected() -> None:
    """Blank assistant names should be rejected."""

    with pytest.raises(ValueError, match="Assistant name cannot be empty"):
        Assistant(
            registry=ServiceRegistry(),
            parser=RuleBasedCommandParser(),
            context=build_test_context(),
            name="   ",
        )


def test_start_message_contains_assistant_name() -> None:
    """The startup message should include the assistant name."""

    assistant = build_test_assistant()

    response = assistant.start_message()

    assert response.text == "Balanspeaker is ready. Type 'exit' to quit."


@pytest.mark.asyncio
async def test_handle_text_executes_matching_command() -> None:
    """A recognised command should be routed to its service."""

    assistant = build_test_assistant()

    response = await assistant.handle_text("hello")

    assert response.text == "Handled greeting."


@pytest.mark.asyncio
async def test_handle_empty_text_returns_prompt() -> None:
    """Blank input should produce a prompt rather than a service call."""

    assistant = build_test_assistant()

    response = await assistant.handle_text("   ")

    assert response.text == "Please enter a command."


@pytest.mark.asyncio
async def test_chat_without_registered_service_is_unavailable() -> None:
    """Chat requests require a registered chatbot service."""

    assistant = build_test_assistant()

    response = await assistant.handle_text("How should I store basil?")

    assert response.text == "That capability is not currently available."


@pytest.mark.asyncio
async def test_known_command_without_service_returns_unavailable() -> None:
    """A parsed command without a registered service should be unavailable."""

    assistant = build_test_assistant()

    response = await assistant.handle_text("what time is it")

    assert response.text == "That capability is not currently available."


@pytest.mark.asyncio
async def test_service_failure_returns_controlled_error() -> None:
    """A service exception should not crash the assistant."""

    assistant = build_test_assistant(include_time_service=True)

    response = await assistant.handle_text("what time is it")

    assert response.text == ("Something went wrong while handling that request.")


@pytest.mark.asyncio
async def test_timer_without_registered_service_is_unavailable() -> None:
    """A valid timer command should be unavailable before service registration."""

    assistant = build_test_assistant()

    response = await assistant.handle_text("timer 5")

    assert response.text == "That capability is not currently available."


@pytest.mark.asyncio
async def test_timer_without_duration_returns_parser_error() -> None:
    """An incomplete timer command should return a parser validation error."""

    assistant = build_test_assistant()

    response = await assistant.handle_text("set a timer")

    assert response.text == "Please specify how long the timer should run."
