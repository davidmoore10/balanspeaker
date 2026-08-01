"""Service for reporting the current local time."""

from collections.abc import Callable
from datetime import datetime

from models.response import AssistantResponse
from services.base import Service


class ClockService(Service):
    """Handles requests for the current time."""

    _SUPPORTED_PHRASES = {
        "what time is it",
        "what's the time",
        "tell me the time",
        "current time",
        "time",
    }

    def __init__(
        self,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._now_provider = now_provider or datetime.now

    @property
    def name(self) -> str:
        """Return the service name."""

        return "clock"

    def can_handle(self, user_text: str) -> bool:
        """Return whether the user is asking for the current time."""

        normalized_text = user_text.strip().lower().rstrip("?.!")
        return normalized_text in self._SUPPORTED_PHRASES

    async def execute(self, user_text: str) -> AssistantResponse:
        """Return the current local time."""

        current_time = self._now_provider()

        return AssistantResponse(text=f"The current time is {current_time:%H:%M}.")
