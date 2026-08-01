"""Tests for media command parsing."""

import pytest

from assistant.parser import RuleBasedCommandParser
from models.command import CommandType


@pytest.mark.parametrize(
    "user_text",
    [
        "play music",
        "play media",
        "start music",
        "start playing music",
        "put some music on",
    ],
)
def test_parser_recognises_play_media_requests(
    user_text: str,
) -> None:
    """Play phrases should produce play-media commands."""

    parser = RuleBasedCommandParser()

    command = parser.parse(user_text)

    assert command.type == CommandType.PLAY_MEDIA
    assert command.parameters == {}
    assert command.error is None


@pytest.mark.parametrize(
    "user_text",
    [
        "pause music",
        "pause media",
        "pause the music",
        "pause playback",
    ],
)
def test_parser_recognises_pause_media_requests(
    user_text: str,
) -> None:
    """Pause phrases should produce pause-media commands."""

    parser = RuleBasedCommandParser()

    command = parser.parse(user_text)

    assert command.type == CommandType.PAUSE_MEDIA
    assert command.parameters == {}
    assert command.error is None


@pytest.mark.parametrize(
    "user_text",
    [
        "resume music",
        "resume media",
        "resume the music",
        "continue music",
        "continue playing",
    ],
)
def test_parser_recognises_resume_media_requests(
    user_text: str,
) -> None:
    """Resume phrases should produce resume-media commands."""

    parser = RuleBasedCommandParser()

    command = parser.parse(user_text)

    assert command.type == CommandType.RESUME_MEDIA
    assert command.parameters == {}
    assert command.error is None


@pytest.mark.parametrize(
    "user_text",
    [
        "stop music",
        "stop media",
        "stop the music",
        "stop playback",
        "turn off music",
        "turn off the music",
    ],
)
def test_parser_recognises_stop_media_requests(
    user_text: str,
) -> None:
    """Stop phrases should produce stop-media commands."""

    parser = RuleBasedCommandParser()

    command = parser.parse(user_text)

    assert command.type == CommandType.STOP_MEDIA
    assert command.parameters == {}
    assert command.error is None


def test_stop_alarm_remains_alarm_command() -> None:
    """Stopping an alarm must not be interpreted as media control."""

    parser = RuleBasedCommandParser()

    command = parser.parse("stop alarm")

    assert command.type == CommandType.STOP_ALARM


def test_cancel_timer_remains_timer_command() -> None:
    """Cancelling a timer must not be interpreted as media control."""

    parser = RuleBasedCommandParser()

    command = parser.parse("cancel timer")

    assert command.type == CommandType.CANCEL_TIMER
