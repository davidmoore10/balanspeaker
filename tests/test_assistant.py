"""Tests for the core assistant."""

import pytest

from assistant.assistant import Assistant
from assistant.registry import ServiceRegistry
from models.response import AssistantResponse
from services.base import Service


class TestService(Service):
    """A service used only by assistant tests."""

    @property
    def name(self) -> str:
        return "test"

    def can_handle(self, user_text: str) -> bool:
        return user_text.lower() == "handle me"

    async def execute(self, user_text: str) -> AssistantResponse:
        return AssistantResponse(text="Handled successfully.")


class FailingService(Service):
    """A service that simulates an internal failure."""

    @property
    def name(self) -> str:
        return "failing"

    def can_handle(self, user_text: str) -> bool:
        return user_text.lower() == "fail"

    async def execute(self, user_text: str) -> AssistantResponse:
        raise RuntimeError("Simulated service failure")


def build_test_assistant() -> Assistant:
    registry = ServiceRegistry()
    registry.register(TestService())
    registry.register(FailingService())

    return Assistant(
        registry=registry,
        name="Balanspeaker",
    )


def test_default_assistant_name() -> None:
    assistant = Assistant(registry=ServiceRegistry())

    assert assistant.name == "Balanspeaker"


def test_custom_assistant_name_is_trimmed() -> None:
    assistant = Assistant(
        registry=ServiceRegistry(),
        name="  Kitchen Assistant  ",
    )

    assert assistant.name == "Kitchen Assistant"


def test_empty_assistant_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Assistant name cannot be empty"):
        Assistant(
            registry=ServiceRegistry(),
            name="   ",
        )


def test_start_message_contains_assistant_name() -> None:
    assistant = Assistant(
        registry=ServiceRegistry(),
        name="Balanspeaker",
    )

    response = assistant.start_message()

    assert isinstance(response, AssistantResponse)
    assert response.text == "Balanspeaker is ready. Type 'exit' to quit."


@pytest.mark.asyncio
async def test_handle_text_uses_matching_service() -> None:
    assistant = build_test_assistant()

    response = await assistant.handle_text("handle me")

    assert response.text == "Handled successfully."


@pytest.mark.asyncio
async def test_handle_empty_text_returns_prompt() -> None:
    assistant = build_test_assistant()

    response = await assistant.handle_text("   ")

    assert response.text == "Please enter a command."


@pytest.mark.asyncio
async def test_unknown_request_returns_fallback() -> None:
    assistant = build_test_assistant()

    response = await assistant.handle_text("unknown request")

    assert response.text == "I don't know how to handle that yet."


@pytest.mark.asyncio
async def test_service_failure_does_not_crash_assistant() -> None:
    assistant = build_test_assistant()

    response = await assistant.handle_text("fail")

    assert response.text == ("Something went wrong while handling that request.")
