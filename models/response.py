"""Response models returned by the assistant and its services."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    """A response produced by the assistant."""

    text: str
