"""Errors produced while interpreting user commands."""

from dataclasses import dataclass
from enum import StrEnum


class ParserErrorCode(StrEnum):
    """Known parser error categories."""

    MISSING_TIMER_DURATION = "missing_timer_duration"
    INVALID_TIMER_DURATION = "invalid_timer_duration"
    UNSUPPORTED_TIMER_UNIT = "unsupported_timer_unit"


@dataclass(frozen=True, slots=True)
class ParserError:
    """A structured error produced by command parsing."""

    code: ParserErrorCode
    message: str
