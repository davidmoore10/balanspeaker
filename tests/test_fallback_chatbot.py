"""Tests for chatbot provider fallback behaviour."""

from datetime import datetime

import pytest

from ai.errors import ChatbotProviderUnavailableError
from ai.fallback import FallbackChatbotProvider
from ai.provider import ChatbotProvider
from domains.conversation.message import ConversationMessage
from domains.conversation.role import ConversationRole


class FixedProvider(ChatbotProvider):
    """Provider returning a fixed response or error."""

    def __init__(
        self,
        *,
        provider_name: str,
        response: str = "",
        error: Exception | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._response = response
        self._error = error
        self.call_count = 0

    @property
    def name(self) -> str:
        """Return the provider name."""

        return self._provider_name

    async def generate_response(
        self,
        *,
        history: tuple[ConversationMessage, ...],
    ) -> str:
        """Return the configured response."""

        self.call_count += 1

        if self._error is not None:
            raise self._error

        return self._response


def build_history() -> tuple[ConversationMessage, ...]:
    """Create minimal conversation history."""

    return (
        ConversationMessage.create(
            role=ConversationRole.USER,
            content="Hello",
            created_at=datetime(2026, 8, 1, 20, 0),
        ),
    )


@pytest.mark.asyncio
async def test_primary_provider_is_used_when_available() -> None:
    """The primary provider should normally handle requests."""

    primary = FixedProvider(
        provider_name="primary",
        response="Primary response",
    )
    fallback = FixedProvider(
        provider_name="fallback",
        response="Fallback response",
    )

    provider = FallbackChatbotProvider(
        primary=primary,
        fallback=fallback,
    )

    response = await provider.generate_response(history=build_history())

    assert response == "Primary response"
    assert primary.call_count == 1
    assert fallback.call_count == 0
    assert provider.last_provider_name == "primary"
    assert not provider.last_used_fallback


@pytest.mark.asyncio
async def test_fallback_is_used_when_primary_unavailable() -> None:
    """Availability failures should activate the fallback."""

    primary = FixedProvider(
        provider_name="primary",
        error=ChatbotProviderUnavailableError("Primary unavailable"),
    )
    fallback = FixedProvider(
        provider_name="fallback",
        response="Fallback response",
    )

    provider = FallbackChatbotProvider(
        primary=primary,
        fallback=fallback,
    )

    response = await provider.generate_response(history=build_history())

    assert response == "Fallback response"
    assert primary.call_count == 1
    assert fallback.call_count == 1
    assert provider.last_provider_name == "fallback"
    assert provider.last_used_fallback
