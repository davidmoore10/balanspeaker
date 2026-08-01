"""Speech activity and silence detection."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class RecordingCompletionReason(StrEnum):
    """Reasons an automatic recording may finish."""

    SILENCE = "silence"
    NO_SPEECH = "no_speech"
    MAXIMUM_DURATION = "maximum_duration"


@dataclass(frozen=True, slots=True)
class SilenceDetectionSettings:
    """Configuration for amplitude-based speech detection."""

    speech_threshold: float = 0.015
    silence_duration_seconds: float = 1.0
    speech_start_timeout_seconds: float = 5.0
    maximum_recording_seconds: float = 20.0
    minimum_speech_seconds: float = 0.2
    pre_roll_seconds: float = 0.3

    def __post_init__(self) -> None:
        if self.speech_threshold <= 0:
            raise ValueError("Speech threshold must be greater than zero.")

        durations = (
            self.silence_duration_seconds,
            self.speech_start_timeout_seconds,
            self.maximum_recording_seconds,
            self.minimum_speech_seconds,
            self.pre_roll_seconds,
        )

        if any(value < 0 for value in durations):
            raise ValueError("Silence-detection durations cannot be negative.")

        if self.silence_duration_seconds == 0:
            raise ValueError("Silence duration must be greater than zero.")

        if self.speech_start_timeout_seconds == 0:
            raise ValueError("Speech-start timeout must be greater than zero.")

        if self.maximum_recording_seconds == 0:
            raise ValueError("Maximum recording duration must be greater than zero.")

        if self.maximum_recording_seconds <= self.minimum_speech_seconds:
            raise ValueError(
                "Maximum recording duration must exceed the minimum speech duration."
            )


class SpeechActivityDetector:
    """Detect speech onset and trailing silence from audio blocks."""

    def __init__(
        self,
        *,
        sample_rate: int,
        settings: SilenceDetectionSettings,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("Sample rate must be greater than zero.")

        self._sample_rate = sample_rate
        self._settings = settings

        self._total_samples = 0
        self._speech_samples = 0
        self._trailing_silence_samples = 0

        self._speech_started = False
        self._complete = False
        self._completion_reason: RecordingCompletionReason | None = None

    @property
    def speech_started(self) -> bool:
        """Return whether speech has been detected."""

        return self._speech_started

    @property
    def complete(self) -> bool:
        """Return whether recording should stop."""

        return self._complete

    @property
    def completion_reason(
        self,
    ) -> RecordingCompletionReason | None:
        """Return why recording was completed."""

        return self._completion_reason

    @property
    def total_duration_seconds(self) -> float:
        """Return total processed audio duration."""

        return self._total_samples / self._sample_rate

    @property
    def speech_duration_seconds(self) -> float:
        """Return detected non-silent duration."""

        return self._speech_samples / self._sample_rate

    def process(
        self,
        block: NDArray[np.float32],
    ) -> None:
        """Process one mono audio block."""

        if self._complete:
            return

        samples = np.asarray(
            block,
            dtype=np.float32,
        ).reshape(-1)

        if samples.size == 0:
            return

        block_sample_count = int(samples.size)
        self._total_samples += block_sample_count

        rms = float(np.sqrt(np.mean(np.square(samples))))

        speech_present = rms >= self._settings.speech_threshold

        if not self._speech_started:
            if speech_present:
                self._speech_started = True
                self._speech_samples += block_sample_count
                self._trailing_silence_samples = 0
            elif (
                self.total_duration_seconds
                >= self._settings.speech_start_timeout_seconds
            ):
                self._finish(RecordingCompletionReason.NO_SPEECH)

        else:
            if speech_present:
                self._speech_samples += block_sample_count
                self._trailing_silence_samples = 0
            else:
                self._trailing_silence_samples += block_sample_count

            trailing_silence_seconds = (
                self._trailing_silence_samples / self._sample_rate
            )

            if (
                self.speech_duration_seconds >= self._settings.minimum_speech_seconds
                and trailing_silence_seconds >= self._settings.silence_duration_seconds
            ):
                self._finish(RecordingCompletionReason.SILENCE)

        if (
            not self._complete
            and self.total_duration_seconds >= self._settings.maximum_recording_seconds
        ):
            self._finish(RecordingCompletionReason.MAXIMUM_DURATION)

    def _finish(
        self,
        reason: RecordingCompletionReason,
    ) -> None:
        """Mark recording as complete."""

        self._complete = True
        self._completion_reason = reason
