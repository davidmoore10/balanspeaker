"""Integration tests for application conversation handling."""

import pytest

from ai.stub import StubChatbotProvider
from config.settings import Settings
from domains.conversation.role import ConversationRole
from main import build_application
from speech.silent import SilentSpeechProvider


def build_chat_application():
    """Build the application with deterministic providers."""

    return build_application(
        chatbot_provider=StubChatbotProvider(),
        speech_provider=SilentSpeechProvider(),
        settings=Settings(
            chatbot_provider="stub",
            speech_provider="silent",
        ),
    )


@pytest.mark.asyncio
async def test_application_blocks_general_question_in_command_mode() -> None:
    """General text should not invoke AI by default."""

    assistant, _, context = build_chat_application()

    response = await assistant.handle_text("How should I store fresh basil?")

    assert response.text == (
        "I didn't recognise that command. Say 'engage AI' to start a conversation."
    )
    assert context.conversation_manager.get_history() == ()


@pytest.mark.asyncio
async def test_application_handles_general_question_in_ai_mode() -> None:
    """Explicit AI mode should enable conversational requests."""

    assistant, _, context = build_chat_application()

    await assistant.handle_text("engage AI")

    response = await assistant.handle_text("How should I store fresh basil?")

    assert "How should I store fresh basil?" in response.text

    history = context.conversation_manager.get_history()

    assert len(history) == 2
    assert history[0].role == ConversationRole.USER
    assert history[1].role == ConversationRole.ASSISTANT


@pytest.mark.asyncio
async def test_application_preserves_follow_up_context() -> None:
    """Follow-up questions should see previous AI context."""

    assistant, _, context = build_chat_application()

    await assistant.handle_text("engage AI")

    await assistant.handle_text("How should I store fresh basil?")

    response = await assistant.handle_text("What about parsley?")

    assert "What about parsley?" in response.text
    assert "How should I store fresh basil?" in response.text
    assert len(context.conversation_manager.get_history()) == 4


@pytest.mark.asyncio
async def test_device_commands_do_not_enter_chat_history() -> None:
    """Device commands should not pollute AI history."""

    assistant, _, context = build_chat_application()

    await assistant.handle_text("engage AI")
    await assistant.handle_text("timer 5")
    await assistant.handle_text("play music")

    assert context.conversation_manager.get_history() == ()
