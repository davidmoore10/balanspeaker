"""Convert user text into structured commands."""

import re
from abc import ABC, abstractmethod

from models.command import Command, CommandType
from models.parser_error import ParserError, ParserErrorCode


class CommandParser(ABC):
    """Interface implemented by command parsers."""

    @abstractmethod
    def parse(self, user_text: str) -> Command:
        """Convert user text into a structured command."""


class RuleBasedCommandParser(CommandParser):
    """Parse a small set of commands using deterministic rules."""

    _GREETINGS = {
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    }

    _TIME_REQUESTS = {
        "what time is it",
        "what's the time",
        "tell me the time",
        "current time",
        "time",
    }

    _TIMER_KEYWORDS = {
        "timer",
        "countdown",
    }

    _TIMER_PATTERN = re.compile(
        r"\b(?P<duration>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)?\b"
    )

    def parse(self, user_text: str) -> Command:
        """Convert user text into a structured command."""

        original_text = user_text
        normalized_text = self._normalize(user_text)

        if normalized_text in self._GREETINGS:
            return Command(
                type=CommandType.GREET,
                original_text=original_text,
            )

        if normalized_text in self._TIME_REQUESTS:
            return Command(
                type=CommandType.GET_TIME,
                original_text=original_text,
            )

        if self._looks_like_timer_request(normalized_text):
            return self._parse_timer(
                normalized_text=normalized_text,
                original_text=original_text,
            )

        return Command(
            type=CommandType.UNKNOWN,
            original_text=original_text,
        )

    def _parse_timer(
        self,
        normalized_text: str,
        original_text: str,
    ) -> Command:
        """Parse a timer request into seconds."""

        match = self._TIMER_PATTERN.search(normalized_text)

        if match is None:
            return Command(
                type=CommandType.START_TIMER,
                original_text=original_text,
                error=ParserError(
                    code=ParserErrorCode.MISSING_TIMER_DURATION,
                    message="Please specify how long the timer should run.",
                ),
            )

        raw_duration = match.group("duration")
        raw_unit = match.group("unit") or "seconds"

        try:
            duration = float(raw_duration)
        except ValueError:
            return Command(
                type=CommandType.START_TIMER,
                original_text=original_text,
                error=ParserError(
                    code=ParserErrorCode.INVALID_TIMER_DURATION,
                    message="The timer duration is invalid.",
                ),
            )

        if duration <= 0:
            return Command(
                type=CommandType.START_TIMER,
                original_text=original_text,
                error=ParserError(
                    code=ParserErrorCode.INVALID_TIMER_DURATION,
                    message="The timer duration must be greater than zero.",
                ),
            )

        multiplier = self._unit_multiplier(raw_unit)

        if multiplier is None:
            return Command(
                type=CommandType.START_TIMER,
                original_text=original_text,
                error=ParserError(
                    code=ParserErrorCode.UNSUPPORTED_TIMER_UNIT,
                    message="That timer unit is not supported.",
                ),
            )

        duration_seconds = round(duration * multiplier)

        if duration_seconds <= 0:
            return Command(
                type=CommandType.START_TIMER,
                original_text=original_text,
                error=ParserError(
                    code=ParserErrorCode.INVALID_TIMER_DURATION,
                    message="The timer duration is too short.",
                ),
            )

        return Command(
            type=CommandType.START_TIMER,
            parameters={
                "duration_seconds": duration_seconds,
            },
            original_text=original_text,
        )

    def _looks_like_timer_request(self, normalized_text: str) -> bool:
        """Return whether the text appears to be a timer request."""

        words = set(normalized_text.split())
        return bool(words & self._TIMER_KEYWORDS)

    @staticmethod
    def _unit_multiplier(unit: str) -> int | None:
        """Return the number of seconds represented by a unit."""

        normalized_unit = unit.lower()

        if normalized_unit in {"second", "seconds", "sec", "secs"}:
            return 1

        if normalized_unit in {"minute", "minutes", "min", "mins"}:
            return 60

        if normalized_unit in {"hour", "hours", "hr", "hrs"}:
            return 3600

        return None

    @staticmethod
    def _normalize(user_text: str) -> str:
        """Normalize text for deterministic matching."""

        return user_text.strip().lower().rstrip("?.!")
