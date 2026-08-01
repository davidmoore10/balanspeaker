"""Tests for the zero-cost development chatbot."""

from datetime import datetime

import pytest

from ai.stub import StubChatbotProvider
from domains.conversation.message import ConversationMessage
from domains.conversation.role import ConversationRole


def build_message(
    role: ConversationRole,
    content: str,
) -> ConversationMessage:
    """Create a deterministic conversation message."""

    return ConversationMessage.create(
        role=role,
        content=content,
        created_at=datetime(2026, 8, 1, 20, 0),
    )


def test_stub_provider_name() -> None:
    """The stub provider should identify itself."""

    provider = StubChatbotProvider()

    assert provider.name == "stub"


@pytest.mark.asyncio
async def test_stub_handles_first_question() -> None:
    """The first question should receive a deterministic response."""

    provider = StubChatbotProvider()

    response = await provider.generate_response(
        history=(
            build_message(
                ConversationRole.USER,
                "How should I store basil?",
            ),
        )
    )

    assert response == (
        "I am currently using the local development chatbot. "
        "I received your question: How should I store basil?"
    )


@pytest.mark.asyncio
async def test_stub_receives_follow_up_context() -> None:
    """Follow-up responses should reference prior user context."""

    provider = StubChatbotProvider()

    response = await provider.generate_response(
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

    assert response == (
        "I received your follow-up question: "
        "What about parsley? "
        "The previous user message was: "
        "How should I store basil?"
    )
