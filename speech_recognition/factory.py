"""Construction of speech-recognition components."""

from config.settings import Settings
from speech_recognition.faster_whisper_provider import (
    FasterWhisperProvider,
)
from speech_recognition.microphone import MicrophoneRecorder
from speech_recognition.provider import SpeechToTextProvider
from speech_recognition.silence import (
    SilenceDetectionSettings,
)


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

    silence_settings = SilenceDetectionSettings(
        speech_threshold=(settings.microphone_speech_threshold),
        silence_duration_seconds=(settings.microphone_silence_seconds),
        speech_start_timeout_seconds=(settings.microphone_start_timeout_seconds),
        maximum_recording_seconds=(settings.microphone_maximum_recording_seconds),
        minimum_speech_seconds=(settings.microphone_minimum_speech_seconds),
        pre_roll_seconds=(settings.microphone_pre_roll_seconds),
    )

    return MicrophoneRecorder(
        sample_rate=settings.microphone_sample_rate,
        channels=1,
        device=settings.microphone_device,
        block_duration_seconds=(settings.microphone_block_seconds),
        silence_settings=silence_settings,
    )
