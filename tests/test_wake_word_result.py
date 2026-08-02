"""Tests for wake-word result models."""

import pytest

from wake_word.result import WakeWordDetection


def test_valid_detection() -> None:
    """Valid detections should retain their values."""

    detection = WakeWordDetection(
        model_name="hey_jarvis",
        score=0.75,
    )

    assert detection.model_name == "hey_jarvis"
    assert detection.score == 0.75


@pytest.mark.parametrize(
    "score",
    [-0.1, 1.1],
)
def test_invalid_score_is_rejected(
    score: float,
) -> None:
    """Scores must be probabilities."""

    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        WakeWordDetection(
            model_name="hey_jarvis",
            score=score,
        )
