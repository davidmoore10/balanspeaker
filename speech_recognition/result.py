"""Speech-transcription result models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Text and metadata produced from recorded speech."""

    text: str
    language: str | None = None
    language_probability: float | None = None
