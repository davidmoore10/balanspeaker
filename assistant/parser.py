"""Convert user text into structured commands."""

from abc import ABC, abstractmethod

from models.command import Command, CommandType


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

        return Command(
            type=CommandType.UNKNOWN,
            original_text=original_text,
        )

    @staticmethod
    def _normalize(user_text: str) -> str:
        """Normalize text for deterministic matching."""

        return user_text.strip().lower().rstrip("?.!")
