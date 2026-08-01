"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration for Balanspeaker."""

    chatbot_provider: str = "openai"

    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    openai_max_output_tokens: int = 180
    openai_timeout_seconds: float = 30.0

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_temperature: float = 0.3
    ollama_keep_alive: str = "30m"

    speech_provider: str = "piper"
    piper_voice_path: Path = Path("models_data/piper/en_GB-jenny_dioco-medium.onnx")

    speech_to_text_provider: str = "faster-whisper"
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = "en"
    whisper_beam_size: int = 5

    microphone_sample_rate: int = 16000
    microphone_device: int | str | None = None


def load_settings() -> Settings:
    """Load and validate application settings."""

    chatbot_provider = (
        os.getenv(
            "BALANSPEAKER_CHATBOT_PROVIDER",
            "openai",
        )
        .strip()
        .lower()
    )

    supported_chatbot_providers = {
        "openai",
        "ollama",
        "stub",
    }

    if chatbot_provider not in supported_chatbot_providers:
        raise ValueError(
            "BALANSPEAKER_CHATBOT_PROVIDER must be 'openai', 'ollama' or 'stub'."
        )

    openai_api_key = os.getenv(
        "OPENAI_API_KEY",
        "",
    ).strip()

    openai_model = os.getenv(
        "BALANSPEAKER_OPENAI_MODEL",
        "gpt-5-mini",
    ).strip()

    openai_max_output_tokens_text = os.getenv(
        "BALANSPEAKER_OPENAI_MAX_OUTPUT_TOKENS",
        "180",
    ).strip()

    openai_timeout_text = os.getenv(
        "BALANSPEAKER_OPENAI_TIMEOUT_SECONDS",
        "30",
    ).strip()

    ollama_host = os.getenv(
        "BALANSPEAKER_OLLAMA_HOST",
        "http://localhost:11434",
    ).strip()

    ollama_model = os.getenv(
        "BALANSPEAKER_OLLAMA_MODEL",
        "llama3.2:1b",
    ).strip()

    ollama_temperature_text = os.getenv(
        "BALANSPEAKER_OLLAMA_TEMPERATURE",
        "0.3",
    ).strip()

    ollama_keep_alive = os.getenv(
        "BALANSPEAKER_OLLAMA_KEEP_ALIVE",
        "30m",
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
        ("models_data/piper/en_GB-jenny_dioco-medium.onnx"),
    ).strip()

    speech_to_text_provider = (
        os.getenv(
            "BALANSPEAKER_STT_PROVIDER",
            "faster-whisper",
        )
        .strip()
        .lower()
    )

    whisper_model = os.getenv(
        "BALANSPEAKER_WHISPER_MODEL",
        "base.en",
    ).strip()

    whisper_device = (
        os.getenv(
            "BALANSPEAKER_WHISPER_DEVICE",
            "cpu",
        )
        .strip()
        .lower()
    )

    whisper_compute_type = (
        os.getenv(
            "BALANSPEAKER_WHISPER_COMPUTE_TYPE",
            "int8",
        )
        .strip()
        .lower()
    )

    whisper_language = (
        os.getenv(
            "BALANSPEAKER_WHISPER_LANGUAGE",
            "en",
        )
        .strip()
        .lower()
    )

    whisper_beam_size_text = os.getenv(
        "BALANSPEAKER_WHISPER_BEAM_SIZE",
        "5",
    ).strip()

    microphone_sample_rate_text = os.getenv(
        "BALANSPEAKER_MICROPHONE_SAMPLE_RATE",
        "16000",
    ).strip()

    microphone_device_text = os.getenv(
        "BALANSPEAKER_MICROPHONE_DEVICE",
        "",
    ).strip()

    if chatbot_provider == "openai" and not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY must be configured when using the OpenAI chatbot provider."
        )

    if not openai_model:
        raise ValueError("BALANSPEAKER_OPENAI_MODEL cannot be empty.")

    try:
        openai_max_output_tokens = int(openai_max_output_tokens_text)
    except ValueError as error:
        raise ValueError(
            "BALANSPEAKER_OPENAI_MAX_OUTPUT_TOKENS must be an integer."
        ) from error

    if openai_max_output_tokens <= 0:
        raise ValueError(
            "BALANSPEAKER_OPENAI_MAX_OUTPUT_TOKENS must be greater than zero."
        )

    try:
        openai_timeout_seconds = float(openai_timeout_text)
    except ValueError as error:
        raise ValueError(
            "BALANSPEAKER_OPENAI_TIMEOUT_SECONDS must be numeric."
        ) from error

    if openai_timeout_seconds <= 0:
        raise ValueError(
            "BALANSPEAKER_OPENAI_TIMEOUT_SECONDS must be greater than zero."
        )

    if not ollama_host:
        raise ValueError("BALANSPEAKER_OLLAMA_HOST cannot be empty.")

    if not ollama_model:
        raise ValueError("BALANSPEAKER_OLLAMA_MODEL cannot be empty.")

    try:
        ollama_temperature = float(ollama_temperature_text)
    except ValueError as error:
        raise ValueError("BALANSPEAKER_OLLAMA_TEMPERATURE must be numeric.") from error

    if not 0 <= ollama_temperature <= 2:
        raise ValueError(
            "BALANSPEAKER_OLLAMA_TEMPERATURE must be between zero and two."
        )

    if not ollama_keep_alive:
        raise ValueError("BALANSPEAKER_OLLAMA_KEEP_ALIVE cannot be empty.")

    if speech_provider not in {"piper", "silent"}:
        raise ValueError("BALANSPEAKER_SPEECH_PROVIDER must be 'piper' or 'silent'.")

    if not piper_voice_path_text:
        raise ValueError("BALANSPEAKER_PIPER_VOICE_PATH cannot be empty.")

    if speech_to_text_provider != "faster-whisper":
        raise ValueError("BALANSPEAKER_STT_PROVIDER must be 'faster-whisper'.")

    if not whisper_model:
        raise ValueError("BALANSPEAKER_WHISPER_MODEL cannot be empty.")

    if not whisper_device:
        raise ValueError("BALANSPEAKER_WHISPER_DEVICE cannot be empty.")

    if not whisper_compute_type:
        raise ValueError("BALANSPEAKER_WHISPER_COMPUTE_TYPE cannot be empty.")

    if not whisper_language:
        raise ValueError("BALANSPEAKER_WHISPER_LANGUAGE cannot be empty.")

    try:
        whisper_beam_size = int(whisper_beam_size_text)
    except ValueError as error:
        raise ValueError(
            "BALANSPEAKER_WHISPER_BEAM_SIZE must be an integer."
        ) from error

    if whisper_beam_size <= 0:
        raise ValueError("BALANSPEAKER_WHISPER_BEAM_SIZE must be greater than zero.")

    try:
        microphone_sample_rate = int(microphone_sample_rate_text)
    except ValueError as error:
        raise ValueError(
            "BALANSPEAKER_MICROPHONE_SAMPLE_RATE must be an integer."
        ) from error

    if microphone_sample_rate <= 0:
        raise ValueError(
            "BALANSPEAKER_MICROPHONE_SAMPLE_RATE must be greater than zero."
        )

    microphone_device: int | str | None

    if not microphone_device_text:
        microphone_device = None
    else:
        try:
            microphone_device = int(microphone_device_text)
        except ValueError:
            microphone_device = microphone_device_text

    return Settings(
        chatbot_provider=chatbot_provider,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_max_output_tokens=(openai_max_output_tokens),
        openai_timeout_seconds=openai_timeout_seconds,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        ollama_temperature=ollama_temperature,
        ollama_keep_alive=ollama_keep_alive,
        speech_provider=speech_provider,
        piper_voice_path=Path(piper_voice_path_text),
        speech_to_text_provider=speech_to_text_provider,
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
        whisper_language=whisper_language,
        whisper_beam_size=whisper_beam_size,
        microphone_sample_rate=microphone_sample_rate,
        microphone_device=microphone_device,
    )
