"""Tests for wake-word settings."""

import pytest

from config.settings import load_settings


def configure_stub_chatbot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid requiring an OpenAI key."""

    monkeypatch.setenv(
        "BALANSPEAKER_CHATBOT_PROVIDER",
        "stub",
    )


def test_default_wake_word_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wake-word listening should default to Hey Jarvis."""

    configure_stub_chatbot(monkeypatch)

    variable_names = (
        "BALANSPEAKER_WAKE_WORD_ENABLED",
        "BALANSPEAKER_WAKE_WORD_MICROPHONE_DEVICE",
        "BALANSPEAKER_WAKE_WORD_PROVIDER",
        "BALANSPEAKER_WAKE_WORD_MODEL",
        "BALANSPEAKER_WAKE_WORD_THRESHOLD",
        "BALANSPEAKER_WAKE_WORD_VAD_THRESHOLD",
        "BALANSPEAKER_WAKE_WORD_FRAME_SECONDS",
        "BALANSPEAKER_WAKE_WORD_COOLDOWN_SECONDS",
    )

    for variable_name in variable_names:
        monkeypatch.delenv(
            variable_name,
            raising=False,
        )

    settings = load_settings()

    assert settings.wake_word_enabled
    assert settings.wake_word_microphone_device is None
    assert settings.wake_word_provider == "hybrid"
    assert settings.wake_word_model == "hey_jarvis"
    assert settings.wake_word_threshold == 0.5
    assert settings.wake_word_vad_threshold == 0.0
    assert settings.wake_word_frame_seconds == 0.08
    assert settings.wake_word_cooldown_seconds == 2.0


def test_wake_word_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Always-on listening should be configurable."""

    configure_stub_chatbot(monkeypatch)

    monkeypatch.setenv(
        "BALANSPEAKER_WAKE_WORD_ENABLED",
        "false",
    )

    settings = load_settings()

    assert not settings.wake_word_enabled


def test_wake_word_microphone_can_differ_from_command_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wake detection should support a separately selected input endpoint."""

    configure_stub_chatbot(monkeypatch)
    monkeypatch.setenv("BALANSPEAKER_MICROPHONE_DEVICE", "1")
    monkeypatch.setenv("BALANSPEAKER_WAKE_WORD_MICROPHONE_DEVICE", "10")

    settings = load_settings()

    assert settings.microphone_device == 1
    assert settings.wake_word_microphone_device == 10


def test_wake_word_threshold_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Threshold should support environment overrides."""

    configure_stub_chatbot(monkeypatch)

    monkeypatch.setenv(
        "BALANSPEAKER_WAKE_WORD_THRESHOLD",
        "0.65",
    )

    settings = load_settings()

    assert settings.wake_word_threshold == 0.65


def test_invalid_wake_word_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thresholds above one should be rejected."""

    configure_stub_chatbot(monkeypatch)

    monkeypatch.setenv(
        "BALANSPEAKER_WAKE_WORD_THRESHOLD",
        "1.2",
    )

    with pytest.raises(
        ValueError,
        match="no greater than one",
    ):
        load_settings()
