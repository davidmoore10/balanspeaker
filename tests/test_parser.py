"""Tests for the rule-based command parser."""

import pytest

from assistant.parser import RuleBasedCommandParser
from models.command import CommandType
from models.parser_error import ParserErrorCode


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
    assert command.error is None


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
    assert command.error is None


@pytest.mark.parametrize(
    ("user_text", "expected_seconds"),
    [
        ("timer 5", 5),
        ("timer 10 seconds", 10),
        ("timer 20 secs", 20),
        ("set a timer for 2 minutes", 120),
        ("set timer 3 mins", 180),
        ("countdown 1 hour", 3600),
        ("timer 2 hrs", 7200),
        ("timer 1.5 minutes", 90),
        ("TIMER 30 SECONDS!", 30),
    ],
)
def test_parser_converts_timer_duration_to_seconds(
    user_text: str,
    expected_seconds: int,
) -> None:
    parser = RuleBasedCommandParser()

    command = parser.parse(user_text)

    assert command.type == CommandType.START_TIMER
    assert command.parameters == {
        "duration_seconds": expected_seconds,
    }
    assert command.error is None


def test_parser_defaults_timer_unit_to_seconds() -> None:
    parser = RuleBasedCommandParser()

    command = parser.parse("timer 15")

    assert command.parameters["duration_seconds"] == 15


@pytest.mark.parametrize(
    "user_text",
    [
        "timer",
        "set a timer",
        "start countdown",
    ],
)
def test_parser_reports_missing_timer_duration(
    user_text: str,
) -> None:
    parser = RuleBasedCommandParser()

    command = parser.parse(user_text)

    assert command.type == CommandType.START_TIMER
    assert command.error is not None
    assert command.error.code == ParserErrorCode.MISSING_TIMER_DURATION


@pytest.mark.parametrize(
    "user_text",
    [
        "timer 0",
        "timer 0 seconds",
    ],
)
def test_parser_rejects_zero_timer_duration(
    user_text: str,
) -> None:
    parser = RuleBasedCommandParser()

    command = parser.parse(user_text)

    assert command.type == CommandType.START_TIMER
    assert command.error is not None
    assert command.error.code == ParserErrorCode.INVALID_TIMER_DURATION


def test_parser_does_not_confuse_timer_with_time() -> None:
    parser = RuleBasedCommandParser()

    command = parser.parse("timer 5")

    assert command.type == CommandType.START_TIMER
    assert command.type != CommandType.GET_TIME


def test_parser_returns_unknown_for_unsupported_request() -> None:
    parser = RuleBasedCommandParser()

    command = parser.parse("play some music")

    assert command.type == CommandType.UNKNOWN
