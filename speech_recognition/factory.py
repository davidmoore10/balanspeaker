"""Construction of configured speech-recognition components."""

from config.settings import Settings
from speech_recognition.faster_whisper_provider import (
    FasterWhisperProvider,
)
from speech_recognition.microphone import MicrophoneRecorder
from speech_recognition.provider import SpeechToTextProvider


def build_speech_to_text_provider(
    settings: Settings,
) -> SpeechToTextProvider:
    """Build the configured transcription provider."""

    if settings.speech_to_text_provider == "faster-whisper":
        return FasterWhisperProvider(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            language=settings.whisper_language,
            beam_size=settings.whisper_beam_size,
        )

    raise ValueError(
        f"Unsupported speech-to-text provider: {settings.speech_to_text_provider}"
    )


def build_microphone_recorder(
    settings: Settings,
) -> MicrophoneRecorder:
    """Build the configured microphone recorder."""

    return MicrophoneRecorder(
        sample_rate=settings.microphone_sample_rate,
        channels=1,
        device=settings.microphone_device,
    )
