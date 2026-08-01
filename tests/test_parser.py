"""Tests for the rule-based command parser."""

import pytest

from assistant.parser import RuleBasedCommandParser
from models.command import CommandType


@pytest.mark.parametrize(
    "user_text",
    [
        "hello",
        "Hi",
        "  hey  ",
        "good morning",
        "good evening!",
    ],
)
def test_parser_recognises_greetings(user_text: str) -> None:
    parser = RuleBasedCommandParser()

    command = parser.parse(user_text)

    assert command.type == CommandType.GREET
    assert command.original_text == user_text


@pytest.mark.parametrize(
    "user_text",
    [
        "what time is it",
        "What's the time?",
        "tell me the time",
        "current time",
        "TIME",
    ],
)
def test_parser_recognises_time_requests(user_text: str) -> None:
    parser = RuleBasedCommandParser()

    command = parser.parse(user_text)

    assert command.type == CommandType.GET_TIME


def test_parser_does_not_confuse_timer_with_time() -> None:
    parser = RuleBasedCommandParser()

    command = parser.parse("timer 5")

    assert command.type == CommandType.UNKNOWN


def test_parser_returns_unknown_for_unsupported_request() -> None:
    parser = RuleBasedCommandParser()

    command = parser.parse("play some music")

    assert command.type == CommandType.UNKNOWN
