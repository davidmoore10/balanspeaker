"""Wake-word detection result models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WakeWordDetection:
    """One successful wake-word activation."""

    model_name: str
    score: float

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("Wake-word model name cannot be empty.")

        if not 0 <= self.score <= 1:
            raise ValueError("Wake-word score must be between zero and one.")
