"""Tests for alarm-related command parsing."""

import pytest

from assistant.parser import RuleBasedCommandParser
from models.command import CommandType


@pytest.mark.parametrize(
    "user_text",
    [
        "stop alarm",
        "stop the alarm",
        "silence alarm",
        "silence the alarm",
        "turn off alarm",
        "turn off the alarm",
        "dismiss alarm",
        "stop ringing",
    ],
)
def test_parser_recognises_stop_alarm_requests(
    user_text: str,
) -> None:
    """Alarm dismissal phrases should produce a stop command."""

    parser = RuleBasedCommandParser()

    command = parser.parse(user_text)

    assert command.type == CommandType.STOP_ALARM
    assert command.parameters == {}
    assert command.error is None


def test_cancel_timer_remains_timer_cancellation() -> None:
    """Timer cancellation should remain separate from alarm dismissal."""

    parser = RuleBasedCommandParser()

    command = parser.parse("cancel timer")

    assert command.type == CommandType.CANCEL_TIMER
