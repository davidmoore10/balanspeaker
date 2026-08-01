"""Audio backend interfaces and test implementations."""

from abc import ABC, abstractmethod


class AudioBackend(ABC):
    """Interface implemented by physical or simulated audio systems."""

    @abstractmethod
    async def play_media(self) -> None:
        """Begin or resume media playback."""

    @abstractmethod
    async def pause_media(self) -> None:
        """Pause media playback."""

    @abstractmethod
    async def stop_media(self) -> None:
        """Stop media playback."""

    @abstractmethod
    async def start_alarm(self) -> None:
        """Begin alarm playback."""

    @abstractmethod
    async def stop_alarm(self) -> None:
        """Stop alarm playback."""


class SimulatedAudioBackend(AudioBackend):
    """Audio backend that records operations without producing sound."""

    def __init__(self) -> None:
        self._operations: list[str] = []

    @property
    def operations(self) -> tuple[str, ...]:
        """Return recorded audio operations."""

        return tuple(self._operations)

    def clear_operations(self) -> None:
        """Remove all recorded operations."""

        self._operations.clear()

    async def play_media(self) -> None:
        """Record a media-play operation."""

        self._operations.append("play_media")

    async def pause_media(self) -> None:
        """Record a media-pause operation."""

        self._operations.append("pause_media")

    async def stop_media(self) -> None:
        """Record a media-stop operation."""

        self._operations.append("stop_media")

    async def start_alarm(self) -> None:
        """Record an alarm-start operation."""

        self._operations.append("start_alarm")

    async def stop_alarm(self) -> None:
        """Record an alarm-stop operation."""

        self._operations.append("stop_alarm")
