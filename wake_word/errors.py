"""Errors raised by wake-word components."""


class WakeWordError(RuntimeError):
    """Base exception raised by wake-word components."""


class WakeWordProviderUnavailableError(WakeWordError):
    """Raised when the wake-word model cannot be used."""


class WakeWordMicrophoneError(WakeWordError):
    """Raised when the wake-word microphone cannot be used."""
