"""Tests for service registration and command routing."""

import pytest

from assistant.registry import (
    DuplicateCommandHandlerError,
    DuplicateServiceError,
    ServiceRegistry,
)
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
        return AssistantResponse(text="Hello")


class DuplicateNameService(GreetingTestService):
    @property
    def name(self) -> str:
        return "GREETING-TEST"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        return frozenset({CommandType.GET_TIME})


class DuplicateCommandService(GreetingTestService):
    @property
    def name(self) -> str:
        return "another-greeting"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        return frozenset({CommandType.GREET})


class EmptyNameService(GreetingTestService):
    @property
    def name(self) -> str:
        return "   "


def test_register_adds_service() -> None:
    registry = ServiceRegistry()
    service = GreetingTestService()

    registry.register(service)

    assert len(registry) == 1
    assert registry.get("greeting-test") is service
    assert registry.service_names == ("greeting-test",)


def test_find_handler_returns_service_for_command() -> None:
    registry = ServiceRegistry()
    service = GreetingTestService()
    registry.register(service)

    handler = registry.find_handler(CommandType.GREET)

    assert handler is service


def test_find_handler_returns_none_without_registration() -> None:
    registry = ServiceRegistry()

    handler = registry.find_handler(CommandType.GET_TIME)

    assert handler is None


def test_duplicate_service_name_is_rejected() -> None:
    registry = ServiceRegistry()
    registry.register(GreetingTestService())

    with pytest.raises(
        DuplicateServiceError,
        match="already registered",
    ):
        registry.register(DuplicateNameService())


def test_duplicate_command_handler_is_rejected() -> None:
    registry = ServiceRegistry()
    registry.register(GreetingTestService())

    with pytest.raises(
        DuplicateCommandHandlerError,
        match="already handled",
    ):
        registry.register(DuplicateCommandService())


def test_empty_service_name_is_rejected() -> None:
    registry = ServiceRegistry()

    with pytest.raises(
        ValueError,
        match="Service name cannot be empty",
    ):
        registry.register(EmptyNameService())
