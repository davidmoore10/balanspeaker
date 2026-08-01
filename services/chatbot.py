"""Service for multi-turn chatbot conversations."""

from assistant.context import ApplicationContext
from models.command import Command, CommandType
from models.response import AssistantResponse
from services.base import Service


class ChatbotService(Service):
    """Handle conversational requests."""

    @property
    def name(self) -> str:
        """Return the service name."""

        return "chatbot"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        """Return supported command types."""

        return frozenset({CommandType.CHAT})

    async def execute(
        self,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Generate and retain a conversational response."""

        if command.type not in self.supported_commands:
            raise ValueError(f"{self.name} cannot handle command '{command.type}'.")

        raw_message = command.parameters.get("message")

        if not isinstance(raw_message, str) or not raw_message.strip():
            raise ValueError("Chat command requires a non-empty message.")

        context.conversation_manager.add_user_message(raw_message)

        response_text = await context.chatbot_provider.generate_response(
            history=context.conversation_manager.get_history(),
        )

        cleaned_response = response_text.strip()

        if not cleaned_response:
            raise ValueError("Chatbot provider returned an empty response.")

        context.conversation_manager.add_assistant_message(cleaned_response)

        return AssistantResponse(text=cleaned_response)
