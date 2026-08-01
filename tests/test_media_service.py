"""Tests for media playback commands."""

import pytest

from domains.audio.backend import SimulatedAudioBackend
from domains.audio.state import MediaPlaybackState
from models.command import Command, CommandType
from services.media import MediaService
from tests.helpers import build_test_context


def get_backend() -> tuple[
    object,
    SimulatedAudioBackend,
]:
    """Create a context and expose its simulated backend."""

    context = build_test_context()
    backend = context.audio_manager.backend

    assert isinstance(backend, SimulatedAudioBackend)

    return context, backend


def test_media_service_declares_supported_commands() -> None:
    """The service should declare all media commands."""

    service = MediaService()

    assert service.supported_commands == frozenset(
        {
            CommandType.PLAY_MEDIA,
            CommandType.PAUSE_MEDIA,
            CommandType.RESUME_MEDIA,
            CommandType.STOP_MEDIA,
        }
    )


@pytest.mark.asyncio
async def test_play_music() -> None:
    """Play should start simulated media."""

    context, backend = get_backend()
    service = MediaService()

    response = await service.execute(
        command=Command(type=CommandType.PLAY_MEDIA),
        context=context,
    )

    assert response.text == "Playing music."
    assert context.audio_manager.media_state == MediaPlaybackState.PLAYING
    assert backend.operations == ("play_media",)


@pytest.mark.asyncio
async def test_play_when_already_playing() -> None:
    """Repeated play commands should be idempotent."""

    context, backend = get_backend()
    service = MediaService()

    await service.execute(
        command=Command(type=CommandType.PLAY_MEDIA),
        context=context,
    )

    response = await service.execute(
        command=Command(type=CommandType.PLAY_MEDIA),
        context=context,
    )

    assert response.text == "Music is already playing."
    assert backend.operations == ("play_media",)


@pytest.mark.asyncio
async def test_pause_music() -> None:
    """Pause should pause active media."""

    context, backend = get_backend()
    service = MediaService()

    await context.audio_manager.play_media()
    backend.clear_operations()

    response = await service.execute(
        command=Command(type=CommandType.PAUSE_MEDIA),
        context=context,
    )

    assert response.text == "Music paused."
    assert context.audio_manager.media_state == MediaPlaybackState.PAUSED
    assert backend.operations == ("pause_media",)


@pytest.mark.asyncio
async def test_pause_without_playing_music() -> None:
    """Pause should report when nothing is playing."""

    context, backend = get_backend()
    service = MediaService()

    response = await service.execute(
        command=Command(type=CommandType.PAUSE_MEDIA),
        context=context,
    )

    assert response.text == "No music is currently playing."
    assert backend.operations == ()


@pytest.mark.asyncio
async def test_resume_paused_music() -> None:
    """Resume should continue paused media."""

    context, backend = get_backend()
    service = MediaService()

    await context.audio_manager.play_media()
    await context.audio_manager.pause_media()
    backend.clear_operations()

    response = await service.execute(
        command=Command(type=CommandType.RESUME_MEDIA),
        context=context,
    )

    assert response.text == "Music resumed."
    assert context.audio_manager.media_state == MediaPlaybackState.PLAYING
    assert backend.operations == ("play_media",)


@pytest.mark.asyncio
async def test_resume_stopped_music() -> None:
    """Resume should not start media that was fully stopped."""

    context, backend = get_backend()
    service = MediaService()

    response = await service.execute(
        command=Command(type=CommandType.RESUME_MEDIA),
        context=context,
    )

    assert response.text == "There is no paused music to resume."
    assert backend.operations == ()


@pytest.mark.asyncio
async def test_stop_music() -> None:
    """Stop should fully stop active media."""

    context, backend = get_backend()
    service = MediaService()

    await context.audio_manager.play_media()
    backend.clear_operations()

    response = await service.execute(
        command=Command(type=CommandType.STOP_MEDIA),
        context=context,
    )

    assert response.text == "Music stopped."
    assert context.audio_manager.media_state == MediaPlaybackState.STOPPED
    assert backend.operations == ("stop_media",)


@pytest.mark.asyncio
async def test_stop_when_already_stopped() -> None:
    """Repeated stop commands should be idempotent."""

    context, backend = get_backend()
    service = MediaService()

    response = await service.execute(
        command=Command(type=CommandType.STOP_MEDIA),
        context=context,
    )

    assert response.text == "Music is already stopped."
    assert backend.operations == ()


@pytest.mark.asyncio
async def test_play_during_alarm_is_deferred() -> None:
    """A play request during an alarm should wait."""

    context, backend = get_backend()
    service = MediaService()

    await context.audio_manager.start_alarm()
    backend.clear_operations()

    response = await service.execute(
        command=Command(type=CommandType.PLAY_MEDIA),
        context=context,
    )

    assert response.text == "Music will start when the alarm stops."
    assert context.audio_manager.media_state == MediaPlaybackState.PAUSED
    assert backend.operations == ()


@pytest.mark.asyncio
async def test_media_service_rejects_wrong_command() -> None:
    """The service should reject unsupported command types."""

    context, _ = get_backend()
    service = MediaService()

    with pytest.raises(ValueError, match="cannot handle"):
        await service.execute(
            command=Command(type=CommandType.GET_TIME),
            context=context,
        )
