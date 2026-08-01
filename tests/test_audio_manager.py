"""Tests for audio interruption and restoration."""

import pytest

from domains.audio.backend import SimulatedAudioBackend
from domains.audio.manager import AudioManager
from domains.audio.state import AlarmPlaybackState, MediaPlaybackState


def build_audio_manager() -> tuple[
    AudioManager,
    SimulatedAudioBackend,
]:
    """Create an audio manager with a simulated backend."""

    backend = SimulatedAudioBackend()

    manager = AudioManager(
        backend=backend,
    )

    return manager, backend


def test_initial_audio_state() -> None:
    """Audio should begin stopped with no alarm."""

    manager, backend = build_audio_manager()

    assert manager.media_state == MediaPlaybackState.STOPPED
    assert manager.alarm_state == AlarmPlaybackState.INACTIVE
    assert not manager.alarm_is_active
    assert backend.operations == ()


@pytest.mark.asyncio
async def test_play_media() -> None:
    """Playing media should invoke the backend."""

    manager, backend = build_audio_manager()

    await manager.play_media()

    assert manager.media_state == MediaPlaybackState.PLAYING
    assert backend.operations == ("play_media",)


@pytest.mark.asyncio
async def test_pause_media() -> None:
    """Playing media should be pausable."""

    manager, backend = build_audio_manager()

    await manager.play_media()
    await manager.pause_media()

    assert manager.media_state == MediaPlaybackState.PAUSED
    assert backend.operations == (
        "play_media",
        "pause_media",
    )


@pytest.mark.asyncio
async def test_stop_media() -> None:
    """Media should be stoppable."""

    manager, backend = build_audio_manager()

    await manager.play_media()
    await manager.stop_media()

    assert manager.media_state == MediaPlaybackState.STOPPED
    assert backend.operations == (
        "play_media",
        "stop_media",
    )


@pytest.mark.asyncio
async def test_alarm_pauses_playing_media() -> None:
    """Starting an alarm should interrupt active media."""

    manager, backend = build_audio_manager()

    await manager.play_media()
    backend.clear_operations()

    await manager.start_alarm()

    assert manager.media_state == MediaPlaybackState.PAUSED
    assert manager.alarm_state == AlarmPlaybackState.ACTIVE
    assert backend.operations == (
        "pause_media",
        "start_alarm",
    )


@pytest.mark.asyncio
async def test_stopping_alarm_resumes_interrupted_media() -> None:
    """Media interrupted by an alarm should resume afterwards."""

    manager, backend = build_audio_manager()

    await manager.play_media()
    await manager.start_alarm()
    backend.clear_operations()

    await manager.stop_alarm()

    assert manager.alarm_state == AlarmPlaybackState.INACTIVE
    assert manager.media_state == MediaPlaybackState.PLAYING
    assert backend.operations == (
        "stop_alarm",
        "play_media",
    )


@pytest.mark.asyncio
async def test_alarm_does_not_resume_previously_paused_media() -> None:
    """Media paused before an alarm should remain paused afterwards."""

    manager, backend = build_audio_manager()

    await manager.play_media()
    await manager.pause_media()
    backend.clear_operations()

    await manager.start_alarm()
    await manager.stop_alarm()

    assert manager.media_state == MediaPlaybackState.PAUSED
    assert backend.operations == (
        "start_alarm",
        "stop_alarm",
    )


@pytest.mark.asyncio
async def test_alarm_does_not_resume_stopped_media() -> None:
    """Stopped media should remain stopped after an alarm."""

    manager, backend = build_audio_manager()

    await manager.start_alarm()
    await manager.stop_alarm()

    assert manager.media_state == MediaPlaybackState.STOPPED
    assert backend.operations == (
        "start_alarm",
        "stop_alarm",
    )


@pytest.mark.asyncio
async def test_start_alarm_is_idempotent() -> None:
    """Starting an already-active alarm should have no effect."""

    manager, backend = build_audio_manager()

    await manager.start_alarm()
    await manager.start_alarm()

    assert backend.operations == ("start_alarm",)


@pytest.mark.asyncio
async def test_stop_alarm_is_idempotent() -> None:
    """Stopping an inactive alarm should have no effect."""

    manager, backend = build_audio_manager()

    await manager.stop_alarm()

    assert backend.operations == ()


@pytest.mark.asyncio
async def test_play_request_during_alarm_is_deferred() -> None:
    """Media requested during an alarm should begin afterwards."""

    manager, backend = build_audio_manager()

    await manager.start_alarm()
    backend.clear_operations()

    await manager.play_media()

    assert manager.media_state == MediaPlaybackState.PAUSED
    assert backend.operations == ()

    await manager.stop_alarm()

    assert manager.media_state == MediaPlaybackState.PLAYING
    assert backend.operations == (
        "stop_alarm",
        "play_media",
    )


@pytest.mark.asyncio
async def test_manual_pause_cancels_alarm_resume() -> None:
    """A manual pause should prevent restoration after the alarm."""

    manager, backend = build_audio_manager()

    await manager.play_media()
    await manager.start_alarm()
    await manager.pause_media()
    backend.clear_operations()

    await manager.stop_alarm()

    assert manager.media_state == MediaPlaybackState.PAUSED
    assert backend.operations == ("stop_alarm",)
