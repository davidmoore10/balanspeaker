"""Conversation message model."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from domains.conversation.role import ConversationRole


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """One message in the assistant conversation."""

    role: ConversationRole
    content: str
    created_at: datetime
    id: UUID

    @classmethod
    def create(
        cls,
        *,
        role: ConversationRole,
        content: str,
        created_at: datetime,
    ) -> "ConversationMessage":
        """Create a validated conversation message."""

        cleaned_content = content.strip()

        if not cleaned_content:
            raise ValueError("Conversation message cannot be empty.")

        return cls(
            role=role,
            content=cleaned_content,
            created_at=created_at,
            id=uuid4(),
        )
