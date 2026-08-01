"""Tests for conversational fallback parsing."""

from assistant.parser import RuleBasedCommandParser
from models.command import CommandType


def test_general_question_becomes_chat_command() -> None:
    """General questions should be routed to conversation."""

    parser = RuleBasedCommandParser()

    command = parser.parse("How should I store fresh basil?")

    assert command.type == CommandType.CHAT
    assert command.parameters == {"message": "How should I store fresh basil?"}


def test_follow_up_question_becomes_chat_command() -> None:
    """Short follow-up questions should also route to chat."""

    parser = RuleBasedCommandParser()

    command = parser.parse("What if it is frozen?")

    assert command.type == CommandType.CHAT
    assert command.parameters == {"message": "What if it is frozen?"}


def test_device_command_does_not_become_chat() -> None:
    """Recognised device commands should retain priority."""

    parser = RuleBasedCommandParser()

    command = parser.parse("timer 5")

    assert command.type == CommandType.START_TIMER
