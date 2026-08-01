"""Service for controlling AI conversation mode."""

from assistant.context import ApplicationContext
from models.command import Command, CommandType
from models.response import AssistantResponse
from services.base import Service


class InteractionService(Service):
    """Enable and disable conversational AI mode."""

    @property
    def name(self) -> str:
        """Return the service name."""

        return "interaction"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        """Return supported interaction commands."""

        return frozenset(
            {
                CommandType.ENABLE_AI_MODE,
                CommandType.DISABLE_AI_MODE,
            }
        )

    async def execute(
        self,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Change the current interaction mode."""

        if command.type not in self.supported_commands:
            raise ValueError(f"{self.name} cannot handle command '{command.type}'.")

        if command.type == CommandType.ENABLE_AI_MODE:
            return self._enable_ai(context)

        return self._disable_ai(context)

    @staticmethod
    def _enable_ai(
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Enable conversational AI."""

        changed = context.interaction_manager.enable_ai_mode()

        if not changed:
            return AssistantResponse(text="AI mode is already enabled.")

        context.conversation_manager.clear()

        return AssistantResponse(
            text=("AI mode enabled. What would you like to discuss?")
        )

    @staticmethod
    def _disable_ai(
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Disable conversational AI."""

        changed = context.interaction_manager.disable_ai_mode()

        context.conversation_manager.clear()

        if not changed:
            return AssistantResponse(text="AI mode is already disabled.")

        return AssistantResponse(text="AI mode disabled.")
