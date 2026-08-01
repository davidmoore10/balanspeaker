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
    openai_max_output_tokens: int = 500
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

    microphone_block_seconds: float = 0.1
    microphone_speech_threshold: float = 0.015
    microphone_silence_seconds: float = 1.0
    microphone_start_timeout_seconds: float = 5.0
    microphone_maximum_recording_seconds: float = 20.0
    microphone_minimum_speech_seconds: float = 0.2
    microphone_pre_roll_seconds: float = 0.3


def load_settings() -> Settings:
    """Load and validate application settings."""

    chatbot_provider = _read_choice(
        name="BALANSPEAKER_CHATBOT_PROVIDER",
        default="openai",
        choices={"openai", "ollama", "stub"},
    )

    openai_api_key = os.getenv(
        "OPENAI_API_KEY",
        "",
    ).strip()

    openai_model = _read_non_empty(
        name="BALANSPEAKER_OPENAI_MODEL",
        default="gpt-5-mini",
    )

    openai_max_output_tokens = _read_positive_int(
        name=("BALANSPEAKER_OPENAI_MAX_OUTPUT_TOKENS"),
        default=500,
    )

    openai_timeout_seconds = _read_positive_float(
        name="BALANSPEAKER_OPENAI_TIMEOUT_SECONDS",
        default=30.0,
    )

    if chatbot_provider == "openai" and not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY must be configured when using the OpenAI chatbot provider."
        )

    ollama_host = _read_non_empty(
        name="BALANSPEAKER_OLLAMA_HOST",
        default="http://localhost:11434",
    )

    ollama_model = _read_non_empty(
        name="BALANSPEAKER_OLLAMA_MODEL",
        default="llama3.2:1b",
    )

    ollama_temperature = _read_float(
        name="BALANSPEAKER_OLLAMA_TEMPERATURE",
        default=0.3,
    )

    if not 0 <= ollama_temperature <= 2:
        raise ValueError(
            "BALANSPEAKER_OLLAMA_TEMPERATURE must be between zero and two."
        )

    ollama_keep_alive = _read_non_empty(
        name="BALANSPEAKER_OLLAMA_KEEP_ALIVE",
        default="30m",
    )

    speech_provider = _read_choice(
        name="BALANSPEAKER_SPEECH_PROVIDER",
        default="piper",
        choices={"piper", "silent"},
    )

    piper_voice_path = Path(
        _read_non_empty(
            name="BALANSPEAKER_PIPER_VOICE_PATH",
            default=("models_data/piper/en_GB-jenny_dioco-medium.onnx"),
        )
    )

    speech_to_text_provider = _read_choice(
        name="BALANSPEAKER_STT_PROVIDER",
        default="faster-whisper",
        choices={"faster-whisper"},
    )

    whisper_model = _read_non_empty(
        name="BALANSPEAKER_WHISPER_MODEL",
        default="base.en",
    )

    whisper_device = _read_non_empty(
        name="BALANSPEAKER_WHISPER_DEVICE",
        default="cpu",
    ).lower()

    whisper_compute_type = _read_non_empty(
        name="BALANSPEAKER_WHISPER_COMPUTE_TYPE",
        default="int8",
    ).lower()

    whisper_language = _read_non_empty(
        name="BALANSPEAKER_WHISPER_LANGUAGE",
        default="en",
    ).lower()

    whisper_beam_size = _read_positive_int(
        name="BALANSPEAKER_WHISPER_BEAM_SIZE",
        default=5,
    )

    microphone_sample_rate = _read_positive_int(
        name="BALANSPEAKER_MICROPHONE_SAMPLE_RATE",
        default=16000,
    )

    microphone_device_text = os.getenv(
        "BALANSPEAKER_MICROPHONE_DEVICE",
        "",
    ).strip()

    microphone_device: int | str | None

    if not microphone_device_text:
        microphone_device = None
    else:
        try:
            microphone_device = int(microphone_device_text)
        except ValueError:
            microphone_device = microphone_device_text

    microphone_block_seconds = _read_positive_float(
        name="BALANSPEAKER_MICROPHONE_BLOCK_SECONDS",
        default=0.1,
    )

    microphone_speech_threshold = _read_positive_float(
        name=("BALANSPEAKER_MICROPHONE_SPEECH_THRESHOLD"),
        default=0.015,
    )

    microphone_silence_seconds = _read_positive_float(
        name=("BALANSPEAKER_MICROPHONE_SILENCE_SECONDS"),
        default=1.0,
    )

    microphone_start_timeout_seconds = _read_positive_float(
        name=("BALANSPEAKER_MICROPHONE_START_TIMEOUT_SECONDS"),
        default=5.0,
    )

    microphone_maximum_recording_seconds = _read_positive_float(
        name=("BALANSPEAKER_MICROPHONE_MAXIMUM_SECONDS"),
        default=20.0,
    )

    microphone_minimum_speech_seconds = _read_non_negative_float(
        name=("BALANSPEAKER_MICROPHONE_MINIMUM_SPEECH_SECONDS"),
        default=0.2,
    )

    microphone_pre_roll_seconds = _read_non_negative_float(
        name=("BALANSPEAKER_MICROPHONE_PRE_ROLL_SECONDS"),
        default=0.3,
    )

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
        piper_voice_path=piper_voice_path,
        speech_to_text_provider=speech_to_text_provider,
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
        whisper_language=whisper_language,
        whisper_beam_size=whisper_beam_size,
        microphone_sample_rate=microphone_sample_rate,
        microphone_device=microphone_device,
        microphone_block_seconds=(microphone_block_seconds),
        microphone_speech_threshold=(microphone_speech_threshold),
        microphone_silence_seconds=(microphone_silence_seconds),
        microphone_start_timeout_seconds=(microphone_start_timeout_seconds),
        microphone_maximum_recording_seconds=(microphone_maximum_recording_seconds),
        microphone_minimum_speech_seconds=(microphone_minimum_speech_seconds),
        microphone_pre_roll_seconds=(microphone_pre_roll_seconds),
    )


def _read_choice(
    *,
    name: str,
    default: str,
    choices: set[str],
) -> str:
    """Read an environment value from an allowed set."""

    value = os.getenv(name, default).strip().lower()

    if value not in choices:
        formatted_choices = ", ".join(sorted(choices))

        raise ValueError(f"{name} must be one of: {formatted_choices}.")

    return value


def _read_non_empty(
    *,
    name: str,
    default: str,
) -> str:
    """Read a required non-empty string."""

    value = os.getenv(name, default).strip()

    if not value:
        raise ValueError(f"{name} cannot be empty.")

    return value


def _read_float(
    *,
    name: str,
    default: float,
) -> float:
    """Read a floating-point environment value."""

    value = os.getenv(name, str(default)).strip()

    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric.") from error


def _read_positive_float(
    *,
    name: str,
    default: float,
) -> float:
    """Read a positive floating-point value."""

    value = _read_float(
        name=name,
        default=default,
    )

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value


def _read_non_negative_float(
    *,
    name: str,
    default: float,
) -> float:
    """Read a non-negative floating-point value."""

    value = _read_float(
        name=name,
        default=default,
    )

    if value < 0:
        raise ValueError(f"{name} cannot be negative.")

    return value


def _read_positive_int(
    *,
    name: str,
    default: int,
) -> int:
    """Read a positive integer value."""

    raw_value = os.getenv(
        name,
        str(default),
    ).strip()

    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value
