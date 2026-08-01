"""Service for reporting the current local time."""

from assistant.context import ApplicationContext
from models.command import Command, CommandType
from models.response import AssistantResponse
from services.base import Service


class ClockService(Service):
    """Handle current-time commands."""

    @property
    def name(self) -> str:
        """Return the service name."""

        return "clock"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        """Return supported command types."""

        return frozenset({CommandType.GET_TIME})

    async def execute(
        self,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Return the current local time."""

        if command.type not in self.supported_commands:
            raise ValueError(f"{self.name} cannot handle command '{command.type}'.")

        current_time = context.clock.now()

        return AssistantResponse(text=f"The current time is {current_time:%H:%M}.")
