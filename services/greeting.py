"""Service for greeting the user."""

from models.command import Command, CommandType
from models.response import AssistantResponse
from services.base import Service


class GreetingService(Service):
    """Handle greeting commands."""

    @property
    def name(self) -> str:
        """Return the service name."""

        return "greeting"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        """Return supported command types."""

        return frozenset({CommandType.GREET})

    async def execute(self, command: Command) -> AssistantResponse:
        """Return a greeting response."""

        if command.type not in self.supported_commands:
            raise ValueError(f"{self.name} cannot handle command '{command.type}'.")

        return AssistantResponse(text="Hello! How can I help?")
