"""Tests for natural-language duration parsing."""

import pytest

from assistant.duration_parser import (
    parse_duration_seconds,
    parse_number_phrase,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("five", 5),
        ("twenty", 20),
        ("twenty one", 21),
        ("twenty-one", 21),
        ("ninety nine", 99),
        ("one hundred", 100),
        ("one hundred and five", 105),
        ("two hundred and thirty", 230),
        ("one thousand", 1000),
        ("1.5", 1.5),
        ("a", 1),
        ("an", 1),
    ],
)
def test_parse_number_phrase(
    text: str,
    expected: float,
) -> None:
    """Spoken number phrases should become numbers."""

    assert parse_number_phrase(text) == expected


@pytest.mark.parametrize(
    ("text", "expected_seconds"),
    [
        ("timer five seconds", 5),
        ("set a timer for twenty seconds", 20),
        ("timer twenty-one seconds", 21),
        ("set a timer for ninety minutes", 5400),
        ("set a timer for one hundred seconds", 100),
        ("timer 1.5 minutes", 90),
        ("set a timer for half an hour", 1800),
        ("set a timer for quarter of an hour", 900),
        ("set a timer for a quarter hour", 900),
        ("timer one and a half hours", 5400),
        ("timer two and a half hours", 9000),
        ("set a timer for a minute", 60),
    ],
)
def test_parse_duration_seconds(
    text: str,
    expected_seconds: int,
) -> None:
    """Natural durations should be converted to seconds."""

    assert parse_duration_seconds(text) == expected_seconds


@pytest.mark.parametrize(
    "text",
    [
        "set a timer",
        "timer pasta",
        "five",
        "",
    ],
)
def test_missing_duration_returns_none(
    text: str,
) -> None:
    """Text without a duration should return None."""

    assert parse_duration_seconds(text) is None
