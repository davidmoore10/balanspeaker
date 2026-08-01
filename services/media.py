"""Service for controlling media playback."""

from assistant.context import ApplicationContext
from domains.audio.state import MediaPlaybackState
from models.command import Command, CommandType
from models.response import AssistantResponse
from services.base import Service


class MediaService(Service):
    """Handle basic media playback commands."""

    @property
    def name(self) -> str:
        """Return the service name."""

        return "media"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        """Return supported command types."""

        return frozenset(
            {
                CommandType.PLAY_MEDIA,
                CommandType.PAUSE_MEDIA,
                CommandType.RESUME_MEDIA,
                CommandType.STOP_MEDIA,
            }
        )

    async def execute(
        self,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Execute a media playback command."""

        if command.type not in self.supported_commands:
            raise ValueError(f"{self.name} cannot handle command '{command.type}'.")

        if command.type == CommandType.PLAY_MEDIA:
            return await self._play_media(context)

        if command.type == CommandType.PAUSE_MEDIA:
            return await self._pause_media(context)

        if command.type == CommandType.RESUME_MEDIA:
            return await self._resume_media(context)

        if command.type == CommandType.STOP_MEDIA:
            return await self._stop_media(context)

        raise ValueError(f"{self.name} cannot handle command '{command.type}'.")

    @staticmethod
    async def _play_media(
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Start media playback."""

        if context.audio_manager.media_state == MediaPlaybackState.PLAYING:
            return AssistantResponse(text="Music is already playing.")

        await context.audio_manager.play_media()

        if context.audio_manager.alarm_is_active:
            return AssistantResponse(text="Music will start when the alarm stops.")

        return AssistantResponse(text="Playing music.")

    @staticmethod
    async def _pause_media(
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Pause active media playback."""

        if context.audio_manager.media_state != MediaPlaybackState.PLAYING:
            return AssistantResponse(text="No music is currently playing.")

        await context.audio_manager.pause_media()

        return AssistantResponse(text="Music paused.")

    @staticmethod
    async def _resume_media(
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Resume paused media playback."""

        if context.audio_manager.media_state == MediaPlaybackState.PLAYING:
            return AssistantResponse(text="Music is already playing.")

        if context.audio_manager.media_state == MediaPlaybackState.STOPPED:
            return AssistantResponse(text="There is no paused music to resume.")

        await context.audio_manager.play_media()

        if context.audio_manager.alarm_is_active:
            return AssistantResponse(text="Music will resume when the alarm stops.")

        return AssistantResponse(text="Music resumed.")

    @staticmethod
    async def _stop_media(
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Stop media playback."""

        if context.audio_manager.media_state == MediaPlaybackState.STOPPED:
            return AssistantResponse(text="Music is already stopped.")

        await context.audio_manager.stop_media()

        return AssistantResponse(text="Music stopped.")
