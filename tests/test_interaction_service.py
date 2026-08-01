"""Tests for AI interaction-mode commands."""

import pytest

from models.command import Command, CommandType
from services.interaction import InteractionService
from tests.helpers import build_test_context


def test_interaction_service_supported_commands() -> None:
    """The service should expose AI mode commands."""

    service = InteractionService()

    assert service.supported_commands == frozenset(
        {
            CommandType.ENABLE_AI_MODE,
            CommandType.DISABLE_AI_MODE,
        }
    )


@pytest.mark.asyncio
async def test_enable_ai_mode() -> None:
    """AI mode should be enabled explicitly."""

    context = build_test_context()
    service = InteractionService()

    response = await service.execute(
        command=Command(type=CommandType.ENABLE_AI_MODE),
        context=context,
    )

    assert context.interaction_manager.ai_mode_enabled
    assert response.text == ("AI mode enabled. What would you like to discuss?")


@pytest.mark.asyncio
async def test_disable_ai_clears_history() -> None:
    """Ending AI mode should clear conversation context."""

    context = build_test_context()
    service = InteractionService()

    context.interaction_manager.enable_ai_mode()
    context.conversation_manager.add_user_message("How should I store basil?")

    response = await service.execute(
        command=Command(type=CommandType.DISABLE_AI_MODE),
        context=context,
    )

    assert not context.interaction_manager.ai_mode_enabled
    assert context.conversation_manager.get_history() == ()
    assert response.text == "AI mode disabled."
