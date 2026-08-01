"""Errors raised by local speech-recognition components."""


class SpeechRecognitionError(RuntimeError):
    """Base exception raised during speech recognition."""


class MicrophoneUnavailableError(SpeechRecognitionError):
    """Raised when microphone audio cannot be captured."""


class TranscriptionUnavailableError(SpeechRecognitionError):
    """Raised when the transcription model cannot be used."""


class EmptyTranscriptionError(SpeechRecognitionError):
    """Raised when no meaningful speech was detected."""
