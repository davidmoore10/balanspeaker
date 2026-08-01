"""Core assistant application."""

import traceback

from assistant.context import ApplicationContext
from assistant.parser import CommandParser
from assistant.registry import ServiceRegistry
from models.command import CommandType
from models.response import AssistantResponse


class Assistant:
    """Parse requests and dispatch commands to services."""

    def __init__(
        self,
        registry: ServiceRegistry,
        parser: CommandParser,
        context: ApplicationContext,
        name: str = "Balanspeaker",
    ) -> None:
        cleaned_name = name.strip()

        if not cleaned_name:
            raise ValueError("Assistant name cannot be empty.")

        self._name = cleaned_name
        self._registry = registry
        self._parser = parser
        self._context = context

    @property
    def name(self) -> str:
        """Return the assistant's display name."""

        return self._name

    def start_message(self) -> AssistantResponse:
        """Return the application start message."""

        return AssistantResponse(text=(f"{self._name} is ready. Type 'exit' to quit."))

    async def handle_text(
        self,
        user_text: str,
    ) -> AssistantResponse:
        """Parse and execute one user request."""

        cleaned_text = user_text.strip()

        if not cleaned_text:
            return AssistantResponse(text="Please enter a command.")

        command = self._parser.parse(cleaned_text)

        if command.error is not None:
            return AssistantResponse(text=command.error.message)

        if (
            command.type == CommandType.CHAT
            and not self._context.interaction_manager.ai_mode_enabled
        ):
            return AssistantResponse(
                text=(
                    "I didn't recognise that command. "
                    "Say 'engage AI' to start a conversation."
                )
            )

        if command.type == CommandType.UNKNOWN:
            return AssistantResponse(text="I don't know how to handle that yet.")

        service = self._registry.find_handler(command.type)

        if service is None:
            return AssistantResponse(
                text=("That capability is not currently available.")
            )

        try:
            return await service.execute(
                command=command,
                context=self._context,
            )
        except Exception:
            print("\n[ERROR] Request failed:")
            traceback.print_exc()

            return AssistantResponse(
                text=("Something went wrong while handling that request.")
            )
