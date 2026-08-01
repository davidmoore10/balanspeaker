"""Tests for service registration and selection."""

import pytest

from assistant.registry import DuplicateServiceError, ServiceRegistry
from models.response import AssistantResponse
from services.base import Service


class FirstService(Service):
    @property
    def name(self) -> str:
        return "first"

    def can_handle(self, user_text: str) -> bool:
        return user_text == "shared request"

    async def execute(self, user_text: str) -> AssistantResponse:
        return AssistantResponse(text="First service")


class SecondService(Service):
    @property
    def name(self) -> str:
        return "second"

    def can_handle(self, user_text: str) -> bool:
        return user_text in {"shared request", "second request"}

    async def execute(self, user_text: str) -> AssistantResponse:
        return AssistantResponse(text="Second service")


class DuplicateFirstService(FirstService):
    @property
    def name(self) -> str:
        return "FIRST"


class EmptyNameService(FirstService):
    @property
    def name(self) -> str:
        return "   "


def test_register_adds_service() -> None:
    registry = ServiceRegistry()
    service = FirstService()

    registry.register(service)

    assert len(registry) == 1
    assert registry.get("first") is service
    assert registry.service_names == ("first",)


def test_service_lookup_is_case_insensitive() -> None:
    registry = ServiceRegistry()
    service = FirstService()
    registry.register(service)

    assert registry.get("FIRST") is service
    assert registry.get("  First  ") is service


def test_duplicate_service_name_is_rejected() -> None:
    registry = ServiceRegistry()
    registry.register(FirstService())

    with pytest.raises(
        DuplicateServiceError,
        match="already registered",
    ):
        registry.register(DuplicateFirstService())


def test_empty_service_name_is_rejected() -> None:
    registry = ServiceRegistry()

    with pytest.raises(
        ValueError,
        match="Service name cannot be empty",
    ):
        registry.register(EmptyNameService())


def test_find_handler_returns_matching_service() -> None:
    registry = ServiceRegistry()
    first_service = FirstService()
    second_service = SecondService()

    registry.register(first_service)
    registry.register(second_service)

    assert registry.find_handler("second request") is second_service


def test_find_handler_uses_registration_order() -> None:
    registry = ServiceRegistry()
    first_service = FirstService()
    second_service = SecondService()

    registry.register(first_service)
    registry.register(second_service)

    assert registry.find_handler("shared request") is first_service


def test_find_handler_returns_none_without_match() -> None:
    registry = ServiceRegistry()
    registry.register(FirstService())

    assert registry.find_handler("unhandled request") is None
