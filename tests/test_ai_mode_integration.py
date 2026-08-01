"""Integration tests for explicit AI mode."""

import pytest

from ai.stub import StubChatbotProvider
from assistant.assistant import Assistant
from assistant.parser import RuleBasedCommandParser
from assistant.registry import ServiceRegistry
from services.chatbot import ChatbotService
from services.clock import ClockService
from services.interaction import InteractionService
from services.timer import TimerService
from tests.helpers import build_test_context


def build_assistant() -> tuple[
    Assistant,
    object,
]:
    """Build an assistant supporting AI and commands."""

    context = build_test_context(chatbot_provider=StubChatbotProvider())

    registry = ServiceRegistry()
    registry.register(ClockService())
    registry.register(TimerService())
    registry.register(ChatbotService())
    registry.register(InteractionService())

    assistant = Assistant(
        registry=registry,
        parser=RuleBasedCommandParser(),
        context=context,
    )

    return assistant, context


@pytest.mark.asyncio
async def test_chat_is_blocked_in_command_mode() -> None:
    """Unmatched speech must not call AI by default."""

    assistant, context = build_assistant()

    response = await assistant.handle_text("How should I store basil?")

    assert response.text == (
        "I didn't recognise that command. Say 'engage AI' to start a conversation."
    )
    assert context.conversation_manager.get_history() == ()


@pytest.mark.asyncio
async def test_engage_ai_enables_conversation() -> None:
    """Explicit AI mode should allow conversation."""

    assistant, context = build_assistant()

    enable_response = await assistant.handle_text("engage AI")

    chat_response = await assistant.handle_text("How should I store basil?")

    assert enable_response.text == ("AI mode enabled. What would you like to discuss?")
    assert context.interaction_manager.ai_mode_enabled
    assert "How should I store basil?" in chat_response.text
    assert len(context.conversation_manager.get_history()) == 2


@pytest.mark.asyncio
async def test_commands_still_work_in_ai_mode() -> None:
    """Deterministic commands should retain priority."""

    assistant, context = build_assistant()

    await assistant.handle_text("engage AI")

    response = await assistant.handle_text("timer 5")

    assert response.text == "Timer set for 5 seconds."
    assert context.interaction_manager.ai_mode_enabled
    assert context.conversation_manager.get_history() == ()


@pytest.mark.asyncio
async def test_disengage_ai_clears_conversation() -> None:
    """Ending AI mode should remove short-term context."""

    assistant, context = build_assistant()

    await assistant.handle_text("engage AI")
    await assistant.handle_text("How should I store basil?")

    response = await assistant.handle_text("exit AI mode")

    assert response.text == "AI mode disabled."
    assert not context.interaction_manager.ai_mode_enabled
    assert context.conversation_manager.get_history() == ()


@pytest.mark.asyncio
async def test_chat_is_blocked_again_after_exit() -> None:
    """AI should remain unavailable after disengagement."""

    assistant, _ = build_assistant()

    await assistant.handle_text("engage AI")
    await assistant.handle_text("exit AI mode")

    response = await assistant.handle_text("Tell me about basil")

    assert response.text == (
        "I didn't recognise that command. Say 'engage AI' to start a conversation."
    )
