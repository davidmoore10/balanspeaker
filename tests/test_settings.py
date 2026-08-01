"""Tests for environment-based application settings."""

import pytest

from config.settings import load_settings


def test_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default settings should select local Ollama."""

    monkeypatch.delenv(
        "BALANSPEAKER_CHATBOT_PROVIDER",
        raising=False,
    )
    monkeypatch.delenv(
        "BALANSPEAKER_OLLAMA_HOST",
        raising=False,
    )
    monkeypatch.delenv(
        "BALANSPEAKER_OLLAMA_MODEL",
        raising=False,
    )
    monkeypatch.delenv(
        "BALANSPEAKER_OLLAMA_TEMPERATURE",
        raising=False,
    )
    monkeypatch.delenv(
        "BALANSPEAKER_OLLAMA_KEEP_ALIVE",
        raising=False,
    )

    settings = load_settings()

    assert settings.chatbot_provider == "ollama"
    assert settings.ollama_host == "http://localhost:11434"
    assert settings.ollama_model == "llama3.2:3b"
    assert settings.ollama_temperature == 0.3
    assert settings.ollama_keep_alive == "5m"


def test_settings_read_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment variables should override defaults."""

    monkeypatch.setenv(
        "BALANSPEAKER_CHATBOT_PROVIDER",
        "stub",
    )
    monkeypatch.setenv(
        "BALANSPEAKER_OLLAMA_HOST",
        "http://localhost:12345",
    )
    monkeypatch.setenv(
        "BALANSPEAKER_OLLAMA_MODEL",
        "llama3.2:1b",
    )
    monkeypatch.setenv(
        "BALANSPEAKER_OLLAMA_TEMPERATURE",
        "0.5",
    )
    monkeypatch.setenv(
        "BALANSPEAKER_OLLAMA_KEEP_ALIVE",
        "2m",
    )

    settings = load_settings()

    assert settings.chatbot_provider == "stub"
    assert settings.ollama_host == "http://localhost:12345"
    assert settings.ollama_model == "llama3.2:1b"
    assert settings.ollama_temperature == 0.5
    assert settings.ollama_keep_alive == "2m"


def test_invalid_provider_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only configured provider types should be accepted."""

    monkeypatch.setenv(
        "BALANSPEAKER_CHATBOT_PROVIDER",
        "invalid",
    )

    with pytest.raises(
        ValueError,
        match="must be 'ollama' or 'stub'",
    ):
        load_settings()


def test_invalid_temperature_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Temperature must be numeric."""

    monkeypatch.setenv(
        "BALANSPEAKER_OLLAMA_TEMPERATURE",
        "warm",
    )

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        load_settings()
