"""Coordination of media and alarm audio."""

from domains.audio.backend import AudioBackend
from domains.audio.state import AlarmPlaybackState, MediaPlaybackState


class AudioManager:
    """Coordinate media playback and higher-priority alarm audio."""

    def __init__(
        self,
        *,
        backend: AudioBackend,
    ) -> None:
        self._backend = backend
        self._media_state = MediaPlaybackState.STOPPED
        self._alarm_state = AlarmPlaybackState.INACTIVE
        self._resume_media_after_alarm = False

    @property
    def backend(self) -> AudioBackend:
        """Return the configured audio backend."""

        return self._backend

    @property
    def media_state(self) -> MediaPlaybackState:
        """Return the current media playback state."""

        return self._media_state

    @property
    def alarm_state(self) -> AlarmPlaybackState:
        """Return the current alarm playback state."""

        return self._alarm_state

    @property
    def alarm_is_active(self) -> bool:
        """Return whether alarm audio is currently active."""

        return self._alarm_state == AlarmPlaybackState.ACTIVE

    async def play_media(self) -> None:
        """Start or resume media playback."""

        if self.alarm_is_active:
            self._resume_media_after_alarm = True
            self._media_state = MediaPlaybackState.PAUSED
            return

        await self._backend.play_media()
        self._media_state = MediaPlaybackState.PLAYING

    async def pause_media(self) -> None:
        """Pause media playback."""

        if self._media_state != MediaPlaybackState.PLAYING:
            self._resume_media_after_alarm = False
            return

        await self._backend.pause_media()
        self._media_state = MediaPlaybackState.PAUSED
        self._resume_media_after_alarm = False

    async def stop_media(self) -> None:
        """Stop media playback completely."""

        if self._media_state == MediaPlaybackState.STOPPED:
            self._resume_media_after_alarm = False
            return

        await self._backend.stop_media()
        self._media_state = MediaPlaybackState.STOPPED
        self._resume_media_after_alarm = False

    async def start_alarm(self) -> None:
        """Start alarm audio and interrupt media if necessary."""

        if self.alarm_is_active:
            return

        if self._media_state == MediaPlaybackState.PLAYING:
            await self._backend.pause_media()
            self._media_state = MediaPlaybackState.PAUSED
            self._resume_media_after_alarm = True
        else:
            self._resume_media_after_alarm = False

        await self._backend.start_alarm()
        self._alarm_state = AlarmPlaybackState.ACTIVE

    async def stop_alarm(self) -> None:
        """Stop alarm audio and restore interrupted media."""

        if not self.alarm_is_active:
            return

        await self._backend.stop_alarm()
        self._alarm_state = AlarmPlaybackState.INACTIVE

        if self._resume_media_after_alarm:
            await self._backend.play_media()
            self._media_state = MediaPlaybackState.PLAYING

        self._resume_media_after_alarm = False
