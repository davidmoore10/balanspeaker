"""Tests for automatic speech and silence detection."""

import numpy as np
import pytest

from speech_recognition.silence import (
    RecordingCompletionReason,
    SilenceDetectionSettings,
    SpeechActivityDetector,
)

SAMPLE_RATE = 1000
BLOCK_SIZE = 100


def build_block(
    amplitude: float,
) -> np.ndarray:
    """Build a 100 ms constant-amplitude block."""

    return np.full(
        BLOCK_SIZE,
        amplitude,
        dtype=np.float32,
    )


def build_detector(
    **overrides: float,
) -> SpeechActivityDetector:
    """Build a detector with test-friendly timings."""

    settings = SilenceDetectionSettings(
        speech_threshold=overrides.get(
            "speech_threshold",
            0.1,
        ),
        silence_duration_seconds=overrides.get(
            "silence_duration_seconds",
            0.3,
        ),
        speech_start_timeout_seconds=overrides.get(
            "speech_start_timeout_seconds",
            0.5,
        ),
        maximum_recording_seconds=overrides.get(
            "maximum_recording_seconds",
            2.0,
        ),
        minimum_speech_seconds=overrides.get(
            "minimum_speech_seconds",
            0.2,
        ),
        pre_roll_seconds=0.1,
    )

    return SpeechActivityDetector(
        sample_rate=SAMPLE_RATE,
        settings=settings,
    )


def test_speech_above_threshold_starts_recording() -> None:
    """A sufficiently loud block should start speech."""

    detector = build_detector()

    detector.process(build_block(0.2))

    assert detector.speech_started
    assert not detector.complete


def test_quiet_block_does_not_start_speech() -> None:
    """Background silence should not start speech."""

    detector = build_detector()

    detector.process(build_block(0.01))

    assert not detector.speech_started
    assert not detector.complete


def test_trailing_silence_finishes_recording() -> None:
    """Silence after valid speech should finish recording."""

    detector = build_detector()

    detector.process(build_block(0.2))
    detector.process(build_block(0.2))

    detector.process(build_block(0.0))
    detector.process(build_block(0.0))
    detector.process(build_block(0.0))

    assert detector.complete
    assert detector.completion_reason == (RecordingCompletionReason.SILENCE)


def test_short_noise_does_not_satisfy_minimum_speech() -> None:
    """A short impulse should not count as a full utterance."""

    detector = build_detector(
        minimum_speech_seconds=0.3,
    )

    detector.process(build_block(0.2))

    for _ in range(3):
        detector.process(build_block(0.0))

    assert not detector.complete


def test_no_speech_timeout() -> None:
    """Listening should stop if speech never starts."""

    detector = build_detector()

    for _ in range(5):
        detector.process(build_block(0.0))

    assert detector.complete
    assert detector.completion_reason == (RecordingCompletionReason.NO_SPEECH)


def test_maximum_duration_stops_long_utterance() -> None:
    """Continuous audio should respect the maximum duration."""

    detector = build_detector(
        maximum_recording_seconds=0.5,
    )

    for _ in range(5):
        detector.process(build_block(0.2))

    assert detector.complete
    assert detector.completion_reason == (RecordingCompletionReason.MAXIMUM_DURATION)


def test_invalid_threshold_is_rejected() -> None:
    """Speech threshold must be positive."""

    with pytest.raises(
        ValueError,
        match="threshold",
    ):
        SilenceDetectionSettings(
            speech_threshold=0,
        )
