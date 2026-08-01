"""Tests for the core assistant."""

import pytest

from assistant.assistant import Assistant
from assistant.parser import RuleBasedCommandParser
from assistant.registry import ServiceRegistry
from models.command import Command, CommandType
from models.response import AssistantResponse
from services.base import Service


class GreetingTestService(Service):
    @property
    def name(self) -> str:
        return "greeting-test"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        return frozenset({CommandType.GREET})

    async def execute(self, command: Command) -> AssistantResponse:
        return AssistantResponse(text="Handled greeting.")


class FailingTimeService(Service):
    @property
    def name(self) -> str:
        return "failing-time"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        return frozenset({CommandType.GET_TIME})

    async def execute(self, command: Command) -> AssistantResponse:
        raise RuntimeError("Simulated service failure")


def build_test_assistant(
    include_time_service: bool = False,
) -> Assistant:
    registry = ServiceRegistry()
    registry.register(GreetingTestService())

    if include_time_service:
        registry.register(FailingTimeService())

    return Assistant(
        registry=registry,
        parser=RuleBasedCommandParser(),
        name="Balanspeaker",
    )


def test_default_assistant_name() -> None:
    assistant = Assistant(
        registry=ServiceRegistry(),
        parser=RuleBasedCommandParser(),
    )

    assert assistant.name == "Balanspeaker"


def test_empty_assistant_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Assistant name cannot be empty"):
        Assistant(
            registry=ServiceRegistry(),
            parser=RuleBasedCommandParser(),
            name="   ",
        )


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
