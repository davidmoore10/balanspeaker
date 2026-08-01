"""Short-term conversation history management."""

from core.clock import Clock
from domains.conversation.message import ConversationMessage
from domains.conversation.role import ConversationRole


class ConversationManager:
    """Store and limit short-term conversation history."""

    def __init__(
        self,
        *,
        clock: Clock,
        maximum_messages: int = 20,
    ) -> None:
        if maximum_messages <= 0:
            raise ValueError("Maximum conversation messages must be greater than zero.")

        self._clock = clock
        self._maximum_messages = maximum_messages
        self._messages: list[ConversationMessage] = []

    @property
    def maximum_messages(self) -> int:
        """Return the maximum retained message count."""

        return self._maximum_messages

    def add_message(
        self,
        *,
        role: ConversationRole,
        content: str,
    ) -> ConversationMessage:
        """Add one message and trim old history if necessary."""

        message = ConversationMessage.create(
            role=role,
            content=content,
            created_at=self._clock.now(),
        )

        self._messages.append(message)
        self._trim_history()

        return message

    def add_user_message(
        self,
        content: str,
    ) -> ConversationMessage:
        """Add a user message."""

        return self.add_message(
            role=ConversationRole.USER,
            content=content,
        )

    def add_assistant_message(
        self,
        content: str,
    ) -> ConversationMessage:
        """Add an assistant message."""

        return self.add_message(
            role=ConversationRole.ASSISTANT,
            content=content,
        )

    def get_history(self) -> tuple[ConversationMessage, ...]:
        """Return conversation history in chronological order."""

        return tuple(self._messages)

    def clear(self) -> None:
        """Remove all retained conversation history."""

        self._messages.clear()

    def _trim_history(self) -> None:
        """Remove oldest messages beyond the configured limit."""

        excess_count = len(self._messages) - self._maximum_messages

        if excess_count > 0:
            del self._messages[:excess_count]
