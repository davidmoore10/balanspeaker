"""Service for interrupting assistant speech."""

from assistant.context import ApplicationContext
from models.command import Command, CommandType
from models.response import AssistantResponse
from services.base import Service


class SpeechControlService(Service):
    """Handle speech interruption commands."""

    @property
    def name(self) -> str:
        """Return the service name."""

        return "speech-control"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        """Return supported speech commands."""

        return frozenset({CommandType.STOP_SPEECH})

    async def execute(
        self,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Stop current assistant speech."""

        if command.type not in self.supported_commands:
            raise ValueError(f"{self.name} cannot handle command '{command.type}'.")

        interrupted = await context.speech_manager.interrupt()

        if interrupted:
            return AssistantResponse(text="Speech stopped.")

        return AssistantResponse(text="I am not currently speaking.")
