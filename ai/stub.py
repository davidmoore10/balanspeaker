"""Zero-cost development chatbot provider."""

from ai.provider import ChatbotProvider
from domains.conversation.message import ConversationMessage
from domains.conversation.role import ConversationRole


class StubChatbotProvider(ChatbotProvider):
    """Deterministic provider used before connecting a real model."""

    @property
    def name(self) -> str:
        """Return the provider name."""

        return "stub"

    async def generate_response(
        self,
        *,
        history: tuple[ConversationMessage, ...],
    ) -> str:
        """Return a deterministic development response."""

        user_messages = tuple(
            message for message in history if message.role == ConversationRole.USER
        )

        if not user_messages:
            return "I did not receive a question."

        latest_message = user_messages[-1].content

        if len(user_messages) == 1:
            return (
                "I am currently using the local development chatbot. "
                f"I received your question: {latest_message}"
            )

        previous_message = user_messages[-2].content

        return (
            "I received your follow-up question: "
            f"{latest_message} "
            "The previous user message was: "
            f"{previous_message}"
        )
