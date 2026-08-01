"""Tests for the local Ollama chatbot provider."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
import pytest

from ai.errors import (
    ChatbotProviderResponseError,
    ChatbotProviderUnavailableError,
)
from ai.ollama_provider import OllamaChatbotProvider
from domains.conversation.message import ConversationMessage
from domains.conversation.role import ConversationRole


@dataclass
class FakeResponseMessage:
    """Fake Ollama response message."""

    content: str


@dataclass
class FakeChatResponse:
    """Fake Ollama chat response."""

    message: FakeResponseMessage


class FakeOllamaClient:
    """Controllable replacement for the Ollama client."""

    def __init__(
        self,
        *,
        response_text: str = "Local response",
        error: Exception | None = None,
    ) -> None:
        self._response_text = response_text
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any],
        keep_alive: str,
    ) -> FakeChatResponse:
        """Record and respond to a chat request."""

        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "options": options,
                "keep_alive": keep_alive,
            }
        )

        if self._error is not None:
            raise self._error

        return FakeChatResponse(
            message=FakeResponseMessage(content=self._response_text)
        )


def build_message(
    role: ConversationRole,
    content: str,
) -> ConversationMessage:
    """Create a conversation message for tests."""

    return ConversationMessage.create(
        role=role,
        content=content,
        created_at=datetime(2026, 8, 1, 20, 0),
    )


def test_provider_name_includes_model() -> None:
    """Provider identity should include its model."""

    provider = OllamaChatbotProvider(
        model="llama3.2:3b",
        client=FakeOllamaClient(),
    )

    assert provider.name == "ollama:llama3.2:3b"


@pytest.mark.asyncio
async def test_provider_returns_local_response() -> None:
    """A successful Ollama response should be returned."""

    client = FakeOllamaClient(response_text="Keep basil refrigerated.")

    provider = OllamaChatbotProvider(
        model="llama3.2:3b",
        client=client,
    )

    response = await provider.generate_response(
        history=(
            build_message(
                ConversationRole.USER,
                "How should I store basil?",
            ),
        )
    )

    assert response == "Keep basil refrigerated."


@pytest.mark.asyncio
async def test_provider_sends_system_and_history_messages() -> None:
    """Conversation history should be sent in order."""

    client = FakeOllamaClient()

    provider = OllamaChatbotProvider(
        model="llama3.2:3b",
        temperature=0.2,
        keep_alive="10m",
        client=client,
    )

    await provider.generate_response(
        history=(
            build_message(
                ConversationRole.USER,
                "How should I store basil?",
            ),
            build_message(
                ConversationRole.ASSISTANT,
                "Keep it refrigerated.",
            ),
            build_message(
                ConversationRole.USER,
                "What about parsley?",
            ),
        )
    )

    call = client.calls[0]
    messages = call["messages"]

    assert messages[0]["role"] == "system"
    assert messages[1] == {
        "role": "user",
        "content": "How should I store basil?",
    }
    assert messages[2] == {
        "role": "assistant",
        "content": "Keep it refrigerated.",
    }
    assert messages[3] == {
        "role": "user",
        "content": "What about parsley?",
    }
    assert call["model"] == "llama3.2:3b"
    assert call["options"] == {
        "temperature": 0.2,
        "num_predict": 120,
        "num_ctx": 2048,
    }
    assert call["keep_alive"] == "10m"


@pytest.mark.asyncio
async def test_provider_strips_response_text() -> None:
    """Whitespace around model responses should be removed."""

    provider = OllamaChatbotProvider(
        model="llama3.2:3b",
        client=FakeOllamaClient(response_text="  Local answer.  "),
    )

    response = await provider.generate_response(
        history=(
            build_message(
                ConversationRole.USER,
                "Question",
            ),
        )
    )

    assert response == "Local answer."


@pytest.mark.asyncio
async def test_provider_rejects_empty_response() -> None:
    """Empty model responses should be rejected."""

    provider = OllamaChatbotProvider(
        model="llama3.2:3b",
        client=FakeOllamaClient(response_text="   "),
    )

    with pytest.raises(
        ChatbotProviderResponseError,
        match="empty response",
    ):
        await provider.generate_response(
            history=(
                build_message(
                    ConversationRole.USER,
                    "Question",
                ),
            )
        )


@pytest.mark.asyncio
async def test_provider_converts_connection_error() -> None:
    """Connection failures should become availability errors."""

    request = httpx.Request(
        "POST",
        "http://localhost:11434/api/chat",
    )

    client = FakeOllamaClient(
        error=httpx.ConnectError(
            "Connection failed",
            request=request,
        )
    )

    provider = OllamaChatbotProvider(
        model="llama3.2:3b",
        client=client,
    )

    with pytest.raises(
        ChatbotProviderUnavailableError,
        match="unavailable",
    ):
        await provider.generate_response(
            history=(
                build_message(
                    ConversationRole.USER,
                    "Question",
                ),
            )
        )


@pytest.mark.parametrize(
    ("model", "expected_message"),
    [
        ("", "model cannot be empty"),
        ("   ", "model cannot be empty"),
    ],
)
def test_provider_rejects_empty_model(
    model: str,
    expected_message: str,
) -> None:
    """A model name is mandatory."""

    with pytest.raises(ValueError, match=expected_message):
        OllamaChatbotProvider(
            model=model,
            client=FakeOllamaClient(),
        )


@pytest.mark.parametrize(
    "temperature",
    [-0.1, 2.1],
)
def test_provider_rejects_invalid_temperature(
    temperature: float,
) -> None:
    """Temperature must remain within the supported range."""

    with pytest.raises(
        ValueError,
        match="between zero and two",
    ):
        OllamaChatbotProvider(
            model="llama3.2:3b",
            temperature=temperature,
            client=FakeOllamaClient(),
        )
