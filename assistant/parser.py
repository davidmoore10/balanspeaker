"""Convert user text into structured commands."""

import re
from abc import ABC, abstractmethod

from assistant.duration_parser import parse_duration_seconds
from models.command import Command, CommandType
from models.parser_error import ParserError, ParserErrorCode


class CommandParser(ABC):
    """Interface implemented by command parsers."""

    @abstractmethod
    def parse(self, user_text: str) -> Command:
        """Convert user text into a structured command."""


class RuleBasedCommandParser(CommandParser):
    """Parse deterministic commands and conversational input."""

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
        "what time it is",
        "what's the time",
        "tell me the time",
        "current time",
        "time",
    }

    _ENABLE_AI_REQUESTS = {
        "engage ai",
        "enable ai",
        "enable ai mode",
        "start ai mode",
        "enter ai mode",
        "let's talk",
        "lets talk",
        "i have a question",
        "can i ask you something",
    }

    _DISABLE_AI_REQUESTS = {
        "disengage ai",
        "disable ai",
        "disable ai mode",
        "exit ai mode",
        "leave ai mode",
        "end conversation",
        "finish conversation",
        "back to command mode",
        "return to command mode",
    }

    _STOP_SPEECH_REQUESTS = {
        "stop speaking",
        "stop talking",
        "be quiet",
        "quiet",
        "that's enough",
        "thats enough",
        "enough",
        "shut up",
    }

    _TIMER_LIST_REQUESTS = {
        "list timers",
        "list my timers",
        "show timers",
        "show my timers",
        "what timers are running",
        "which timers are running",
        "what timers do i have",
        "how long is left",
        "how much time is left",
        "timer status",
        "timers",
    }

    _STOP_ALARM_REQUESTS = {
        "stop alarm",
        "stop the alarm",
        "silence alarm",
        "silence the alarm",
        "turn off alarm",
        "turn off the alarm",
        "dismiss alarm",
        "dismiss the alarm",
        "stop ringing",
        "stop the ringing",
    }

    _PLAY_MEDIA_REQUESTS = {
        "play music",
        "play media",
        "start music",
        "start playing music",
        "put some music on",
    }

    _PAUSE_MEDIA_REQUESTS = {
        "pause music",
        "pause media",
        "pause the music",
        "pause playback",
    }

    _RESUME_MEDIA_REQUESTS = {
        "resume music",
        "resume media",
        "resume the music",
        "continue music",
        "continue playing",
    }

    _STOP_MEDIA_REQUESTS = {
        "stop music",
        "stop media",
        "stop the music",
        "stop playback",
        "turn off music",
        "turn off the music",
    }

    _TIMER_KEYWORDS = {
        "timer",
        "timers",
        "countdown",
    }

    _NAMED_TIMER_PATTERN = re.compile(
        r"\b(?:set|start)\s+(?:a|the|my)\s+"
        r"(?P<name>[a-z][a-z0-9 _-]*?)\s+timer\b"
    )

    _CANCEL_TIMER_PATTERN = re.compile(
        r"\b(?:cancel|delete|remove)\s+"
        r"(?:(?:the|my)\s+)?"
        r"(?:(?P<name>[a-z][a-z0-9 _-]*?)\s+)?"
        r"timers?\b"
    )

    _DURATION_UNIT_WORDS = {
        "second",
        "seconds",
        "sec",
        "secs",
        "minute",
        "minutes",
        "min",
        "mins",
        "hour",
        "hours",
        "hr",
        "hrs",
    }

    def parse(self, user_text: str) -> Command:
        """Convert user text into a structured command."""

        original_text = user_text
        normalized_text = self._normalize(user_text)

        if normalized_text in self._STOP_SPEECH_REQUESTS:
            return Command(
                type=CommandType.STOP_SPEECH,
                original_text=original_text,
            )

        if normalized_text in self._ENABLE_AI_REQUESTS:
            return Command(
                type=CommandType.ENABLE_AI_MODE,
                original_text=original_text,
            )

        if normalized_text in self._DISABLE_AI_REQUESTS:
            return Command(
                type=CommandType.DISABLE_AI_MODE,
                original_text=original_text,
            )

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

        if normalized_text in self._STOP_ALARM_REQUESTS:
            return Command(
                type=CommandType.STOP_ALARM,
                original_text=original_text,
            )

        media_command = self._parse_media_command(
            normalized_text=normalized_text,
            original_text=original_text,
        )

        if media_command is not None:
            return media_command

        if self._is_cancel_timer_request(normalized_text):
            return self._parse_cancel_timer(
                normalized_text=normalized_text,
                original_text=original_text,
            )

        if normalized_text in self._TIMER_LIST_REQUESTS:
            return Command(
                type=CommandType.LIST_TIMERS,
                original_text=original_text,
            )

        if self._looks_like_timer_request(normalized_text):
            return self._parse_start_timer(
                normalized_text=normalized_text,
                original_text=original_text,
            )

        return Command(
            type=CommandType.CHAT,
            parameters={
                "message": original_text.strip(),
            },
            original_text=original_text,
        )

    def _parse_media_command(
        self,
        *,
        normalized_text: str,
        original_text: str,
    ) -> Command | None:
        """Parse a media playback command."""

        if normalized_text in self._PLAY_MEDIA_REQUESTS:
            return Command(
                type=CommandType.PLAY_MEDIA,
                original_text=original_text,
            )

        if normalized_text in self._PAUSE_MEDIA_REQUESTS:
            return Command(
                type=CommandType.PAUSE_MEDIA,
                original_text=original_text,
            )

        if normalized_text in self._RESUME_MEDIA_REQUESTS:
            return Command(
                type=CommandType.RESUME_MEDIA,
                original_text=original_text,
            )

        if normalized_text in self._STOP_MEDIA_REQUESTS:
            return Command(
                type=CommandType.STOP_MEDIA,
                original_text=original_text,
            )

        return None

    def _parse_start_timer(
        self,
        normalized_text: str,
        original_text: str,
    ) -> Command:
        """Parse a timer creation request."""

        duration_seconds = parse_duration_seconds(normalized_text)

        if duration_seconds is None:
            return Command(
                type=CommandType.START_TIMER,
                original_text=original_text,
                error=ParserError(
                    code=(ParserErrorCode.MISSING_TIMER_DURATION),
                    message=("Please specify how long the timer should run."),
                ),
            )

        if duration_seconds <= 0:
            return Command(
                type=CommandType.START_TIMER,
                original_text=original_text,
                error=ParserError(
                    code=(ParserErrorCode.INVALID_TIMER_DURATION),
                    message=("The timer duration must be greater than zero."),
                ),
            )

        parameters: dict[str, object] = {
            "duration_seconds": duration_seconds,
        }

        timer_name = self._extract_timer_name(normalized_text)

        if timer_name is not None:
            parameters["name"] = timer_name

        return Command(
            type=CommandType.START_TIMER,
            parameters=parameters,
            original_text=original_text,
        )

    def _parse_cancel_timer(
        self,
        normalized_text: str,
        original_text: str,
    ) -> Command:
        """Parse a timer cancellation request."""

        match = self._CANCEL_TIMER_PATTERN.search(normalized_text)

        parameters: dict[str, object] = {}

        if match is not None:
            raw_name = match.group("name")

            if raw_name is not None:
                cleaned_name = raw_name.strip()

                if cleaned_name:
                    parameters["name"] = cleaned_name

        return Command(
            type=CommandType.CANCEL_TIMER,
            parameters=parameters,
            original_text=original_text,
        )

    def _extract_timer_name(
        self,
        normalized_text: str,
    ) -> str | None:
        """Extract an optional timer name."""

        match = self._NAMED_TIMER_PATTERN.search(normalized_text)

        if match is None:
            return None

        name = match.group("name").strip()

        if not name:
            return None

        # Avoid interpreting phrases such as
        # "start a twenty second timer" as a named timer.
        name_words = set(name.split())

        if name_words & self._DURATION_UNIT_WORDS:
            return None

        return name

    def _looks_like_timer_request(
        self,
        normalized_text: str,
    ) -> bool:
        """Return whether text appears to request a timer."""

        words = set(normalized_text.split())

        return bool(words & self._TIMER_KEYWORDS)

    def _is_cancel_timer_request(
        self,
        normalized_text: str,
    ) -> bool:
        """Return whether text requests timer cancellation."""

        return self._CANCEL_TIMER_PATTERN.search(normalized_text) is not None

    @staticmethod
    def _normalize(user_text: str) -> str:
        """Normalize text for deterministic matching."""

        normalized_text = user_text.strip().lower().rstrip("?.!")

        return re.sub(r"\s+", " ", normalized_text)
