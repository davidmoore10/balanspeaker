"""Service for creating countdown timers."""

from assistant.context import ApplicationContext
from models.command import Command, CommandType
from models.response import AssistantResponse
from services.base import Service


class TimerService(Service):
    """Handle commands that create countdown timers."""

    @property
    def name(self) -> str:
        """Return the service name."""

        return "timer"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        """Return supported command types."""

        return frozenset({CommandType.START_TIMER})

    async def execute(
        self,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Create a timer from a structured command."""

        if command.type not in self.supported_commands:
            raise ValueError(f"{self.name} cannot handle command '{command.type}'.")

        duration_seconds = command.parameters.get("duration_seconds")

        if (
            not isinstance(duration_seconds, int)
            or isinstance(duration_seconds, bool)
            or duration_seconds <= 0
        ):
            raise ValueError(
                "Timer command requires a positive integer "
                "'duration_seconds' parameter."
            )

        timer_name = command.parameters.get("name")

        if timer_name is not None and not isinstance(timer_name, str):
            raise ValueError("Timer name must be a string.")

        await context.timer_manager.create_timer(
            duration_seconds=duration_seconds,
            name=timer_name,
        )

        formatted_duration = self._format_duration(duration_seconds)

        return AssistantResponse(text=f"Timer set for {formatted_duration}.")

    @staticmethod
    def _format_duration(duration_seconds: int) -> str:
        """Return a human-readable timer duration."""

        if duration_seconds % 3600 == 0:
            hours = duration_seconds // 3600
            unit = "hour" if hours == 1 else "hours"
            return f"{hours} {unit}"

        if duration_seconds % 60 == 0:
            minutes = duration_seconds // 60
            unit = "minute" if minutes == 1 else "minutes"
            return f"{minutes} {unit}"

        unit = "second" if duration_seconds == 1 else "seconds"
        return f"{duration_seconds} {unit}"
