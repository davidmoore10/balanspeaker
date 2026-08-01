"""Tests for environment-based application settings."""

import pytest

from config.settings import load_settings


def clear_chatbot_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clear chatbot configuration variables."""

    variable_names = (
        "BALANSPEAKER_CHATBOT_PROVIDER",
        "OPENAI_API_KEY",
        "BALANSPEAKER_OPENAI_MODEL",
        "BALANSPEAKER_OPENAI_MAX_OUTPUT_TOKENS",
        "BALANSPEAKER_OPENAI_TIMEOUT_SECONDS",
        "BALANSPEAKER_OLLAMA_HOST",
        "BALANSPEAKER_OLLAMA_MODEL",
        "BALANSPEAKER_OLLAMA_TEMPERATURE",
        "BALANSPEAKER_OLLAMA_KEEP_ALIVE",
    )

    for variable_name in variable_names:
        monkeypatch.delenv(
            variable_name,
            raising=False,
        )


def test_default_openai_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI should be the default chatbot provider."""

    clear_chatbot_environment(monkeypatch)

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )

    settings = load_settings()

    assert settings.chatbot_provider == "openai"
    assert settings.openai_api_key == "test-key"
    assert settings.openai_model == "gpt-5-mini"
    assert settings.openai_max_output_tokens == 500
    assert settings.openai_timeout_seconds == 30
    assert settings.ollama_model == "llama3.2:1b"
    assert settings.ollama_keep_alive == "30m"


def test_openai_settings_read_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI settings should support environment overrides."""

    clear_chatbot_environment(monkeypatch)

    monkeypatch.setenv(
        "BALANSPEAKER_CHATBOT_PROVIDER",
        "openai",
    )
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "secret-test-key",
    )
    monkeypatch.setenv(
        "BALANSPEAKER_OPENAI_MODEL",
        "gpt-5-mini",
    )
    monkeypatch.setenv(
        "BALANSPEAKER_OPENAI_MAX_OUTPUT_TOKENS",
        "120",
    )
    monkeypatch.setenv(
        "BALANSPEAKER_OPENAI_TIMEOUT_SECONDS",
        "15",
    )

    settings = load_settings()

    assert settings.openai_api_key == ("secret-test-key")
    assert settings.openai_max_output_tokens == 120
    assert settings.openai_timeout_seconds == 15


def test_stub_does_not_require_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub mode should work without OpenAI credentials."""

    clear_chatbot_environment(monkeypatch)

    monkeypatch.setenv(
        "BALANSPEAKER_CHATBOT_PROVIDER",
        "stub",
    )

    settings = load_settings()

    assert settings.chatbot_provider == "stub"
    assert settings.openai_api_key == ""


def test_openai_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI mode should require an API key."""

    clear_chatbot_environment(monkeypatch)

    monkeypatch.setenv(
        "BALANSPEAKER_CHATBOT_PROVIDER",
        "openai",
    )

    with pytest.raises(
        ValueError,
        match="OPENAI_API_KEY must be configured",
    ):
        load_settings()


def test_invalid_provider_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported chatbot providers should be rejected."""

    clear_chatbot_environment(monkeypatch)

    monkeypatch.setenv(
        "BALANSPEAKER_CHATBOT_PROVIDER",
        "invalid",
    )

    with pytest.raises(
        ValueError,
        match=("BALANSPEAKER_CHATBOT_PROVIDER must be one of: ollama, openai, stub"),
    ):
        load_settings()
