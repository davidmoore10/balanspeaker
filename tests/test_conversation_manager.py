"""Tests for short-term conversation history."""

from datetime import datetime, timedelta

import pytest

from core.clock import FakeClock
from domains.conversation.manager import ConversationManager
from domains.conversation.role import ConversationRole


def build_manager(
    maximum_messages: int = 20,
) -> tuple[ConversationManager, FakeClock]:
    """Create a conversation manager and fake clock."""

    clock = FakeClock(datetime(2026, 8, 1, 20, 0))

    manager = ConversationManager(
        clock=clock,
        maximum_messages=maximum_messages,
    )

    return manager, clock


def test_manager_rejects_non_positive_limit() -> None:
    """History limits must be greater than zero."""

    clock = FakeClock(datetime(2026, 8, 1, 20, 0))

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        ConversationManager(
            clock=clock,
            maximum_messages=0,
        )


def test_add_user_message() -> None:
    """User messages should be stored with metadata."""

    manager, clock = build_manager()

    message = manager.add_user_message("How should I store basil?")

    assert message.role == ConversationRole.USER
    assert message.content == "How should I store basil?"
    assert message.created_at == clock.now()
    assert manager.get_history() == (message,)


def test_add_assistant_message() -> None:
    """Assistant messages should be retained."""

    manager, _ = build_manager()

    message = manager.add_assistant_message("Keep it refrigerated.")

    assert message.role == ConversationRole.ASSISTANT
    assert message.content == "Keep it refrigerated."


def test_message_content_is_trimmed() -> None:
    """Whitespace should not be retained around messages."""

    manager, _ = build_manager()

    message = manager.add_user_message("  hello  ")

    assert message.content == "hello"


def test_empty_message_is_rejected() -> None:
    """Blank messages should not enter conversation history."""

    manager, _ = build_manager()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        manager.add_user_message("   ")


def test_history_preserves_message_order() -> None:
    """History should remain chronological."""

    manager, clock = build_manager()

    first = manager.add_user_message("First")

    clock.advance(timedelta(seconds=1))

    second = manager.add_assistant_message("Second")

    assert manager.get_history() == (
        first,
        second,
    )


def test_history_is_trimmed_to_limit() -> None:
    """Oldest messages should be removed beyond the limit."""

    manager, _ = build_manager(maximum_messages=3)

    manager.add_user_message("One")
    manager.add_assistant_message("Two")
    manager.add_user_message("Three")
    manager.add_assistant_message("Four")

    history = manager.get_history()

    assert tuple(message.content for message in history) == (
        "Two",
        "Three",
        "Four",
    )


def test_clear_removes_history() -> None:
    """Conversation history should be clearable."""

    manager, _ = build_manager()

    manager.add_user_message("Hello")
    manager.add_assistant_message("Hi")

    manager.clear()

    assert manager.get_history() == ()
