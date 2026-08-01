"""Core assistant application."""

from assistant.registry import ServiceRegistry
from models.response import AssistantResponse


class Assistant:
    """Coordinates user requests and registered services."""

    def __init__(
        self,
        registry: ServiceRegistry,
        name: str = "Balanspeaker",
    ) -> None:
        cleaned_name = name.strip()

        if not cleaned_name:
            raise ValueError("Assistant name cannot be empty.")

        self._name = cleaned_name
        self._registry = registry

    @property
    def name(self) -> str:
        """Return the assistant's display name."""

        return self._name

    def start_message(self) -> AssistantResponse:
        """Return the message shown when the assistant starts."""

        return AssistantResponse(text=f"{self._name} is ready. Type 'exit' to quit.")

    async def handle_text(self, user_text: str) -> AssistantResponse:
        """Route user text to an appropriate registered service."""

        cleaned_text = user_text.strip()

        if not cleaned_text:
            return AssistantResponse(text="Please enter a command.")

        service = self._registry.find_handler(cleaned_text)

        if service is None:
            return AssistantResponse(text="I don't know how to handle that yet.")

        try:
            return await service.execute(cleaned_text)
        except Exception:
            return AssistantResponse(
                text="Something went wrong while handling that request."
            )
