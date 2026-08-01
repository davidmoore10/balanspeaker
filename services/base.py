"""Interfaces shared by assistant services."""

from abc import ABC, abstractmethod

from models.response import AssistantResponse


class Service(ABC):
    """Base interface implemented by every assistant service."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique service name."""

    @abstractmethod
    def can_handle(self, user_text: str) -> bool:
        """Return whether this service can handle the user's request."""

    @abstractmethod
    async def execute(self, user_text: str) -> AssistantResponse:
        """Execute a request and return the assistant's response."""
