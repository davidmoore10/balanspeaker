"""A simple greeting service."""

from models.response import AssistantResponse
from services.base import Service


class GreetingService(Service):
    """Handles basic greetings."""

    _GREETINGS = {
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    }

    @property
    def name(self) -> str:
        """Return the service name."""

        return "greeting"

    def can_handle(self, user_text: str) -> bool:
        """Return whether the input is a recognised greeting."""

        normalized_text = user_text.strip().lower()
        return normalized_text in self._GREETINGS

    async def execute(self, user_text: str) -> AssistantResponse:
        """Return a greeting response."""

        return AssistantResponse(text="Hello! How can I help?")
