"""Integration tests for application conversation handling."""

import pytest

from domains.conversation.role import ConversationRole
from main import build_application


@pytest.mark.asyncio
async def test_application_handles_general_question() -> None:
    """The production assistant should register the chatbot service."""

    assistant, _, context = build_application()

    response = await assistant.handle_text("How should I store fresh basil?")

    assert "How should I store fresh basil?" in response.text

    history = context.conversation_manager.get_history()

    assert len(history) == 2
    assert history[0].role == ConversationRole.USER
    assert history[1].role == ConversationRole.ASSISTANT


@pytest.mark.asyncio
async def test_application_preserves_follow_up_context() -> None:
    """Follow-up questions should see earlier conversation history."""

    assistant, _, context = build_application()

    await assistant.handle_text("How should I store fresh basil?")

    response = await assistant.handle_text("What about parsley?")

    assert "What about parsley?" in response.text
    assert "How should I store fresh basil?" in response.text
    assert len(context.conversation_manager.get_history()) == 4


@pytest.mark.asyncio
async def test_device_commands_do_not_enter_chat_history() -> None:
    """Timer and media commands should not pollute conversation history."""

    assistant, _, context = build_application()

    await assistant.handle_text("timer 5")
    await assistant.handle_text("play music")

    assert context.conversation_manager.get_history() == ()
