"""Construction of wake-word components."""

from config.settings import Settings
from speech_recognition.faster_whisper_provider import FasterWhisperProvider
from wake_word.hybrid_provider import HybridWakeWordProvider
from wake_word.listener import WakeWordListener
from wake_word.openwakeword_provider import (
    OpenWakeWordProvider,
)
from wake_word.provider import WakeWordProvider


def build_wake_word_provider(
    settings: Settings,
) -> WakeWordProvider:
    """Build the configured wake-word provider."""

    if settings.wake_word_provider in {"openwakeword", "hybrid"}:
        primary = OpenWakeWordProvider(
            model_name=settings.wake_word_model,
            threshold=settings.wake_word_threshold,
            vad_threshold=settings.wake_word_vad_threshold,
        )

        if settings.wake_word_provider == "openwakeword":
            return primary

        return HybridWakeWordProvider(
            primary=primary,
            transcriber=FasterWhisperProvider(
                model_name=settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                language=settings.whisper_language,
                beam_size=1,
            ),
        )

    raise ValueError(f"Unsupported wake-word provider: {settings.wake_word_provider}")


def build_wake_word_listener(
    settings: Settings,
) -> WakeWordListener:
    """Build the continuous microphone listener."""

    provider = build_wake_word_provider(settings)

    return WakeWordListener(
        provider=provider,
        device=(
            settings.wake_word_microphone_device
            if settings.wake_word_microphone_device is not None
            else settings.microphone_device
        ),
        frame_duration_seconds=(settings.wake_word_frame_seconds),
        cooldown_seconds=(settings.wake_word_cooldown_seconds),
    )
