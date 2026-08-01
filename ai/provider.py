"""Interfaces implemented by chatbot model providers."""

from abc import ABC, abstractmethod

from domains.conversation.message import ConversationMessage


class ChatbotProvider(ABC):
    """Interface for local or cloud chatbot providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""

    @abstractmethod
    async def generate_response(
        self,
        *,
        history: tuple[ConversationMessage, ...],
    ) -> str:
        """Generate a response using conversation history."""
