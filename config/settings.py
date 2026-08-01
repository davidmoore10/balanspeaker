"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration for Balanspeaker."""

    chatbot_provider: str = "ollama"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_temperature: float = 0.3
    ollama_keep_alive: str = "5m"

    speech_provider: str = "piper"
    piper_voice_path: Path = Path("models_data/piper/en_GB-alan-medium.onnx")


def load_settings() -> Settings:
    """Load and validate application settings."""

    chatbot_provider = (
        os.getenv(
            "BALANSPEAKER_CHATBOT_PROVIDER",
            "ollama",
        )
        .strip()
        .lower()
    )

    if chatbot_provider not in {"ollama", "stub"}:
        raise ValueError("BALANSPEAKER_CHATBOT_PROVIDER must be 'ollama' or 'stub'.")

    ollama_host = os.getenv(
        "BALANSPEAKER_OLLAMA_HOST",
        "http://localhost:11434",
    ).strip()

    ollama_model = os.getenv(
        "BALANSPEAKER_OLLAMA_MODEL",
        "llama3.2:3b",
    ).strip()

    temperature_text = os.getenv(
        "BALANSPEAKER_OLLAMA_TEMPERATURE",
        "0.3",
    ).strip()

    ollama_keep_alive = os.getenv(
        "BALANSPEAKER_OLLAMA_KEEP_ALIVE",
        "5m",
    ).strip()

    speech_provider = (
        os.getenv(
            "BALANSPEAKER_SPEECH_PROVIDER",
            "piper",
        )
        .strip()
        .lower()
    )

    piper_voice_path_text = os.getenv(
        "BALANSPEAKER_PIPER_VOICE_PATH",
        "models_data/piper/en_GB-alan-medium.onnx",
    ).strip()

    if not ollama_host:
        raise ValueError("BALANSPEAKER_OLLAMA_HOST cannot be empty.")

    if not ollama_model:
        raise ValueError("BALANSPEAKER_OLLAMA_MODEL cannot be empty.")

    if not ollama_keep_alive:
        raise ValueError("BALANSPEAKER_OLLAMA_KEEP_ALIVE cannot be empty.")

    if speech_provider not in {"piper", "silent"}:
        raise ValueError("BALANSPEAKER_SPEECH_PROVIDER must be 'piper' or 'silent'.")

    if not piper_voice_path_text:
        raise ValueError("BALANSPEAKER_PIPER_VOICE_PATH cannot be empty.")

    try:
        temperature = float(temperature_text)
    except ValueError as error:
        raise ValueError("BALANSPEAKER_OLLAMA_TEMPERATURE must be numeric.") from error

    if not 0 <= temperature <= 2:
        raise ValueError(
            "BALANSPEAKER_OLLAMA_TEMPERATURE must be between zero and two."
        )

    return Settings(
        chatbot_provider=chatbot_provider,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        ollama_temperature=temperature,
        ollama_keep_alive=ollama_keep_alive,
        speech_provider=speech_provider,
        piper_voice_path=Path(piper_voice_path_text),
    )
