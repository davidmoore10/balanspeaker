"""Construction of configured speech providers."""

from config.settings import Settings
from speech.fallback import FallbackSpeechProvider
from speech.piper_provider import PiperSpeechProvider
from speech.provider import SpeechProvider
from speech.silent import SilentSpeechProvider


def build_speech_provider(
    settings: Settings,
) -> SpeechProvider:
    """Build the speech provider selected by configuration."""

    silent_provider = SilentSpeechProvider()

    if settings.speech_provider == "silent":
        return silent_provider

    if settings.speech_provider == "piper":
        piper_provider = PiperSpeechProvider(
            model_path=settings.piper_voice_path,
        )

        return FallbackSpeechProvider(
            primary=piper_provider,
            fallback=silent_provider,
        )

    raise ValueError(f"Unsupported speech provider: {settings.speech_provider}")
