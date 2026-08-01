"""Available assistant interaction modes."""

from enum import StrEnum


class InteractionMode(StrEnum):
    """High-level modes controlling assistant behaviour."""

    COMMAND = "command"
    AI = "ai"
    SLEEP = "sleep"
