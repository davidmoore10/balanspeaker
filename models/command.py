"""Structured commands produced from user input."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from models.parser_error import ParserError


class CommandType(StrEnum):
    """Actions understood by the assistant."""

    GREET = "greet"
    GET_TIME = "get_time"
    START_TIMER = "start_timer"
    LIST_TIMERS = "list_timers"
    CANCEL_TIMER = "cancel_timer"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Command:
    """A structured request to an assistant service."""

    type: CommandType
    parameters: dict[str, Any] = field(default_factory=dict)
    original_text: str = ""
    error: ParserError | None = None
