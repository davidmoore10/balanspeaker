"""Tests for speech-related application settings."""

from pathlib import Path

import pytest

from config.settings import load_settings


def test_default_speech_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Piper should be the default local speech provider."""

    monkeypatch.delenv(
        "BALANSPEAKER_SPEECH_PROVIDER",
        raising=False,
    )
    monkeypatch.delenv(
        "BALANSPEAKER_PIPER_VOICE_PATH",
        raising=False,
    )

    settings = load_settings()

    assert settings.speech_provider == "piper"
    assert settings.piper_voice_path == Path("models_data/piper/en_GB-alan-medium.onnx")


def test_speech_settings_read_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speech settings should support environment overrides."""

    monkeypatch.setenv(
        "BALANSPEAKER_SPEECH_PROVIDER",
        "silent",
    )
    monkeypatch.setenv(
        "BALANSPEAKER_PIPER_VOICE_PATH",
        "voices/custom.onnx",
    )

    settings = load_settings()

    assert settings.speech_provider == "silent"
    assert settings.piper_voice_path == Path("voices/custom.onnx")


def test_invalid_speech_provider_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown speech providers should be rejected."""

    monkeypatch.setenv(
        "BALANSPEAKER_SPEECH_PROVIDER",
        "cloud",
    )

    with pytest.raises(
        ValueError,
        match="must be 'piper' or 'silent'",
    ):
        load_settings()


def test_empty_voice_path_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Piper must have a configured voice path."""

    monkeypatch.setenv(
        "BALANSPEAKER_PIPER_VOICE_PATH",
        "   ",
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        load_settings()
