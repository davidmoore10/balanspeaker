"""Interfaces shared by assistant services."""

from abc import ABC, abstractmethod

from assistant.context import ApplicationContext
from models.command import Command, CommandType
from models.response import AssistantResponse


class Service(ABC):
    """Base interface implemented by every assistant service."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique service name."""

    @property
    @abstractmethod
    def supported_commands(self) -> frozenset[CommandType]:
        """Return the command types handled by this service."""

    @abstractmethod
    async def execute(
        self,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Execute a structured command."""
