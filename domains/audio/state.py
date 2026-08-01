"""Audio playback state models."""

from enum import StrEnum


class MediaPlaybackState(StrEnum):
    """Possible media playback states."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class AlarmPlaybackState(StrEnum):
    """Possible alarm playback states."""

    INACTIVE = "inactive"
    ACTIVE = "active"
