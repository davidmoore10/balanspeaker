"""Errors raised by speech providers."""


class SpeechProviderError(RuntimeError):
    """Base exception raised by speech providers."""


class SpeechProviderUnavailableError(SpeechProviderError):
    """Raised when a speech provider cannot be used."""


class SpeechSynthesisError(SpeechProviderError):
    """Raised when speech cannot be synthesised or played."""
