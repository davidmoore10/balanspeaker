"""Conversation participant roles."""

from enum import StrEnum


class ConversationRole(StrEnum):
    """Roles that may appear in conversation history."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
