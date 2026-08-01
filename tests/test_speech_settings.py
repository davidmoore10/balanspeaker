"""Tests for speech-related application settings."""

from pathlib import Path

import pytest

from config.settings import load_settings


def configure_non_openai_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid requiring an OpenAI key in speech-only tests."""

    monkeypatch.setenv(
        "BALANSPEAKER_CHATBOT_PROVIDER",
        "stub",
    )


def test_default_speech_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Piper should be the default local speech provider."""

    configure_non_openai_mode(monkeypatch)

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
    assert settings.piper_voice_path == Path(
        "models_data/piper/en_GB-jenny_dioco-medium.onnx"
    )


def test_speech_settings_read_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speech settings should support environment overrides."""

    configure_non_openai_mode(monkeypatch)

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

    configure_non_openai_mode(monkeypatch)

    monkeypatch.setenv(
        "BALANSPEAKER_SPEECH_PROVIDER",
        "cloud",
    )

    with pytest.raises(
        ValueError,
        match=("BALANSPEAKER_SPEECH_PROVIDER must be one of: piper, silent"),
    ):
        load_settings()


def test_empty_voice_path_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Piper must have a configured voice path."""

    configure_non_openai_mode(monkeypatch)

    monkeypatch.setenv(
        "BALANSPEAKER_PIPER_VOICE_PATH",
        "   ",
    )

    with pytest.raises(
        ValueError,
        match=("BALANSPEAKER_PIPER_VOICE_PATH cannot be empty"),
    ):
        load_settings()
