"""Tests for the OpenAI Responses API provider."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest

from ai.errors import ChatbotProviderResponseError
from ai.openai_provider import OpenAIChatbotProvider
from ai.usage import TokenPricing, TokenUsage
from domains.conversation.message import (
    ConversationMessage,
)
from domains.conversation.role import ConversationRole


@dataclass
class FakeInputTokenDetails:
    """Fake cached-token metadata."""

    cached_tokens: int = 0


@dataclass
class FakeUsage:
    """Fake OpenAI token usage."""

    input_tokens: int
    output_tokens: int
    input_tokens_details: FakeInputTokenDetails


@dataclass
class FakeIncompleteDetails:
    """Fake incomplete-response metadata."""

    reason: str


@dataclass
class FakeResponse:
    """Fake Responses API response."""

    output_text: str
    usage: FakeUsage | None = None
    status: str = "completed"
    incomplete_details: FakeIncompleteDetails | None = None


class FakeResponsesResource:
    """Fake client responses resource."""

    def __init__(
        self,
        response: FakeResponse,
    ) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(
        self,
        **kwargs: Any,
    ) -> FakeResponse:
        """Record and return one fake response."""

        self.calls.append(kwargs)

        return self._response


class FakeOpenAIClient:
    """Fake asynchronous OpenAI client."""

    def __init__(
        self,
        response: FakeResponse,
    ) -> None:
        self.responses = FakeResponsesResource(response)


def build_message(
    role: ConversationRole,
    content: str,
) -> ConversationMessage:
    """Create a conversation message."""

    return ConversationMessage.create(
        role=role,
        content=content,
        created_at=datetime(2026, 8, 1, 20, 0),
    )


def test_provider_name_contains_model() -> None:
    """The provider should identify its model."""

    provider = OpenAIChatbotProvider(
        api_key="test-key",
        model="gpt-5-mini",
        client=FakeOpenAIClient(FakeResponse(output_text="Hello")),
    )

    assert provider.name == "openai:gpt-5-mini"


@pytest.mark.asyncio
async def test_provider_returns_output_text() -> None:
    """The response output text should be returned."""

    client = FakeOpenAIClient(
        FakeResponse(
            output_text="  Keep it refrigerated.  ",
            usage=FakeUsage(
                input_tokens=50,
                output_tokens=12,
                input_tokens_details=(FakeInputTokenDetails(cached_tokens=10)),
            ),
        )
    )

    provider = OpenAIChatbotProvider(
        api_key="test-key",
        client=client,
    )

    result = await provider.generate_response(
        history=(
            build_message(
                ConversationRole.USER,
                "How should I store basil?",
            ),
        )
    )

    assert result == "Keep it refrigerated."


@pytest.mark.asyncio
async def test_provider_sends_configuration_and_history() -> None:
    """History and generation controls should be sent."""

    client = FakeOpenAIClient(FakeResponse(output_text="Response"))

    provider = OpenAIChatbotProvider(
        api_key="test-key",
        model="gpt-5-mini",
        instructions="Use metric units.",
        max_output_tokens=500,
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

    call = client.responses.calls[0]

    assert call["model"] == "gpt-5-mini"
    assert call["instructions"] == ("Use metric units.")
    assert call["max_output_tokens"] == 500
    assert call["reasoning"] == {
        "effort": "low",
    }
    assert call["text"] == {
        "verbosity": "low",
    }
    assert call["input"] == [
        {
            "role": "user",
            "content": "How should I store basil?",
        },
        {
            "role": "assistant",
            "content": "Keep it refrigerated.",
        },
        {
            "role": "user",
            "content": "What about parsley?",
        },
    ]


@pytest.mark.asyncio
async def test_provider_tracks_usage() -> None:
    """Successful requests should update usage totals."""

    client = FakeOpenAIClient(
        FakeResponse(
            output_text="Response",
            usage=FakeUsage(
                input_tokens=100,
                output_tokens=25,
                input_tokens_details=(FakeInputTokenDetails(cached_tokens=20)),
            ),
        )
    )

    pricing = TokenPricing(
        input_per_million=1,
        cached_input_per_million=0.1,
        output_per_million=2,
    )

    provider = OpenAIChatbotProvider(
        api_key="test-key",
        client=client,
        pricing=pricing,
    )

    history = (
        build_message(
            ConversationRole.USER,
            "Question",
        ),
    )

    await provider.generate_response(history=history)
    await provider.generate_response(history=history)

    assert provider.last_usage == TokenUsage(
        input_tokens=100,
        cached_input_tokens=20,
        output_tokens=25,
        request_count=1,
    )

    assert provider.total_usage == TokenUsage(
        input_tokens=200,
        cached_input_tokens=40,
        output_tokens=50,
        request_count=2,
    )

    assert provider.estimated_total_cost_usd == pytest.approx(0.000264)


@pytest.mark.asyncio
async def test_provider_rejects_empty_response() -> None:
    """An empty completed response should be rejected."""

    provider = OpenAIChatbotProvider(
        api_key="test-key",
        client=FakeOpenAIClient(FakeResponse(output_text="   ")),
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
async def test_provider_explains_exhausted_output_budget() -> None:
    """An exhausted budget should produce a diagnostic error."""

    provider = OpenAIChatbotProvider(
        api_key="test-key",
        client=FakeOpenAIClient(
            FakeResponse(
                output_text="",
                status="incomplete",
                incomplete_details=(FakeIncompleteDetails(reason="max_output_tokens")),
            )
        ),
    )

    with pytest.raises(
        ChatbotProviderResponseError,
        match="output-token budget",
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
    "api_key",
    ["", "   "],
)
def test_provider_rejects_empty_api_key(
    api_key: str,
) -> None:
    """An API key must be supplied."""

    with pytest.raises(
        ValueError,
        match="API key cannot be empty",
    ):
        OpenAIChatbotProvider(
            api_key=api_key,
            client=FakeOpenAIClient(FakeResponse(output_text="Response")),
        )
