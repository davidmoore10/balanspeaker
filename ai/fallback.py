"""Fallback coordination between chatbot providers."""

from ai.errors import ChatbotProviderUnavailableError
from ai.provider import ChatbotProvider
from domains.conversation.message import ConversationMessage


class FallbackChatbotProvider(ChatbotProvider):
    """Use a secondary provider when the primary is unavailable."""

    def __init__(
        self,
        *,
        primary: ChatbotProvider,
        fallback: ChatbotProvider,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._last_provider_name: str | None = None
        self._last_used_fallback = False

    @property
    def name(self) -> str:
        """Return a description of the provider chain."""

        return f"{self._primary.name} -> {self._fallback.name}"

    @property
    def last_provider_name(self) -> str | None:
        """Return the provider used for the latest request."""

        return self._last_provider_name

    @property
    def last_used_fallback(self) -> bool:
        """Return whether the latest request used the fallback."""

        return self._last_used_fallback

    async def generate_response(
        self,
        *,
        history: tuple[ConversationMessage, ...],
    ) -> str:
        """Use the primary provider, falling back if unavailable."""

        try:
            response = await self._primary.generate_response(history=history)
        except ChatbotProviderUnavailableError:
            self._last_provider_name = self._fallback.name
            self._last_used_fallback = True

            return await self._fallback.generate_response(history=history)

        self._last_provider_name = self._primary.name
        self._last_used_fallback = False

        return response
