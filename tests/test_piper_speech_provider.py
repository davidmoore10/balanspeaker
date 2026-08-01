"""Tests for the local Piper speech provider."""

import wave
from pathlib import Path

import pytest

from speech.errors import SpeechProviderUnavailableError
from speech.piper_provider import PiperSpeechProvider


class FakePiperVoice:
    """Voice that writes a small valid WAV file."""

    def __init__(self) -> None:
        self.utterances: list[str] = []

    def synthesize_wav(
        self,
        text: str,
        wav_file: wave.Wave_write,
    ) -> None:
        """Write silent PCM audio to the WAV file."""

        self.utterances.append(text)

        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)


@pytest.mark.asyncio
async def test_provider_loads_and_plays_voice(
    tmp_path: Path,
) -> None:
    """A valid model should be loaded and played."""

    model_path = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"

    model_path.write_bytes(b"model")
    config_path.write_text("{}", encoding="utf-8")

    fake_voice = FakePiperVoice()
    loaded_paths: list[Path] = []
    played_paths: list[Path] = []

    def voice_loader(path: Path) -> FakePiperVoice:
        loaded_paths.append(path)
        return fake_voice

    def wav_player(path: Path) -> None:
        assert path.exists()
        played_paths.append(path)

    provider = PiperSpeechProvider(
        model_path=model_path,
        voice_loader=voice_loader,
        wav_player=wav_player,
        playback_stopper=lambda: None,
    )

    await provider.speak("Hello from Balanspeaker.")

    assert loaded_paths == [model_path]
    assert fake_voice.utterances == ["Hello from Balanspeaker."]
    assert len(played_paths) == 1
    assert provider.is_loaded


@pytest.mark.asyncio
async def test_provider_reuses_loaded_voice(
    tmp_path: Path,
) -> None:
    """The model should only be loaded once."""

    model_path = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"

    model_path.write_bytes(b"model")
    config_path.write_text("{}", encoding="utf-8")

    fake_voice = FakePiperVoice()
    load_count = 0

    def voice_loader(path: Path) -> FakePiperVoice:
        nonlocal load_count
        load_count += 1
        return fake_voice

    provider = PiperSpeechProvider(
        model_path=model_path,
        voice_loader=voice_loader,
        wav_player=lambda path: None,
        playback_stopper=lambda: None,
    )

    await provider.speak("First")
    await provider.speak("Second")

    assert load_count == 1
    assert fake_voice.utterances == [
        "First",
        "Second",
    ]


@pytest.mark.asyncio
async def test_provider_ignores_empty_text(
    tmp_path: Path,
) -> None:
    """Blank utterances should not load the voice."""

    provider = PiperSpeechProvider(
        model_path=tmp_path / "missing.onnx",
    )

    await provider.speak("   ")

    assert not provider.is_loaded


@pytest.mark.asyncio
async def test_missing_model_raises_unavailable(
    tmp_path: Path,
) -> None:
    """A missing model should produce an availability error."""

    provider = PiperSpeechProvider(
        model_path=tmp_path / "missing.onnx",
    )

    with pytest.raises(
        SpeechProviderUnavailableError,
        match="voice model was not found",
    ):
        await provider.speak("Hello")


@pytest.mark.asyncio
async def test_missing_configuration_raises_unavailable(
    tmp_path: Path,
) -> None:
    """A model without its JSON configuration is unusable."""

    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"model")

    provider = PiperSpeechProvider(
        model_path=model_path,
    )

    with pytest.raises(
        SpeechProviderUnavailableError,
        match="configuration file was not found",
    ):
        await provider.speak("Hello")


@pytest.mark.asyncio
async def test_stop_uses_playback_stopper(
    tmp_path: Path,
) -> None:
    """Stopping speech should invoke the configured stopper."""

    stop_calls = 0

    def playback_stopper() -> None:
        nonlocal stop_calls
        stop_calls += 1

    provider = PiperSpeechProvider(
        model_path=tmp_path / "voice.onnx",
        playback_stopper=playback_stopper,
    )

    await provider.stop()

    assert stop_calls == 1
