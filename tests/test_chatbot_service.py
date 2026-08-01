"""Tests for multi-turn chatbot conversation handling."""

import pytest

from domains.conversation.role import ConversationRole
from models.command import Command, CommandType
from services.chatbot import ChatbotService
from tests.helpers import build_test_context


def test_chatbot_service_declares_supported_command() -> None:
    """The chatbot service should handle chat commands."""

    service = ChatbotService()

    assert service.supported_commands == frozenset({CommandType.CHAT})


@pytest.mark.asyncio
async def test_chatbot_service_returns_provider_response() -> None:
    """Chat requests should be delegated to the provider."""

    context = build_test_context()
    service = ChatbotService()

    command = Command(
        type=CommandType.CHAT,
        parameters={
            "message": "How should I store basil?",
        },
    )

    response = await service.execute(
        command=command,
        context=context,
    )

    assert response.text == (
        "I am currently using the local development chatbot. "
        "I received your question: How should I store basil?"
    )


@pytest.mark.asyncio
async def test_chatbot_service_stores_both_messages() -> None:
    """User and assistant messages should enter history."""

    context = build_test_context()
    service = ChatbotService()

    command = Command(
        type=CommandType.CHAT,
        parameters={"message": "Hello there"},
    )

    response = await service.execute(
        command=command,
        context=context,
    )

    history = context.conversation_manager.get_history()

    assert len(history) == 2
    assert history[0].role == ConversationRole.USER
    assert history[0].content == "Hello there"
    assert history[1].role == ConversationRole.ASSISTANT
    assert history[1].content == response.text


@pytest.mark.asyncio
async def test_chatbot_service_preserves_follow_up_context() -> None:
    """A second question should see the first conversation turn."""

    context = build_test_context()
    service = ChatbotService()

    first_command = Command(
        type=CommandType.CHAT,
        parameters={
            "message": "How should I store basil?",
        },
    )

    second_command = Command(
        type=CommandType.CHAT,
        parameters={
            "message": "What about parsley?",
        },
    )

    await service.execute(
        command=first_command,
        context=context,
    )

    second_response = await service.execute(
        command=second_command,
        context=context,
    )

    assert "What about parsley?" in second_response.text
    assert "How should I store basil?" in second_response.text
    assert len(context.conversation_manager.get_history()) == 4


@pytest.mark.asyncio
async def test_chatbot_service_rejects_missing_message() -> None:
    """Chat commands require a message."""

    context = build_test_context()
    service = ChatbotService()

    command = Command(
        type=CommandType.CHAT,
        parameters={},
    )

    with pytest.raises(
        ValueError,
        match="non-empty message",
    ):
        await service.execute(
            command=command,
            context=context,
        )


@pytest.mark.asyncio
async def test_chatbot_service_rejects_wrong_command() -> None:
    """The service should reject unsupported command types."""

    context = build_test_context()
    service = ChatbotService()

    command = Command(type=CommandType.GET_TIME)

    with pytest.raises(ValueError, match="cannot handle"):
        await service.execute(
            command=command,
            context=context,
        )
