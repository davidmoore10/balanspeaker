"""Tests for the openWakeWord provider."""

from collections.abc import Mapping

import numpy as np
import pytest

from wake_word.openwakeword_provider import (
    OpenWakeWordProvider,
)


class FakeOpenWakeWordModel:
    """Model returning configured predictions."""

    def __init__(
        self,
        predictions: Mapping[str, float],
    ) -> None:
        self._predictions = predictions
        self.frames: list[np.ndarray] = []

    def predict(
        self,
        audio: np.ndarray,
    ) -> Mapping[str, float]:
        """Record audio and return predictions."""

        self.frames.append(audio.copy())

        return self._predictions


def test_provider_name() -> None:
    """The provider should expose its model."""

    provider = OpenWakeWordProvider(
        model_name="hey_jarvis",
        model=FakeOpenWakeWordModel({}),
    )

    assert provider.name == ("openwakeword:hey_jarvis")
    assert provider.sample_rate == 16000


@pytest.mark.asyncio
async def test_detection_above_threshold() -> None:
    """A sufficiently high score should activate."""

    model = FakeOpenWakeWordModel(
        {
            "hey_jarvis_v0.1": 0.82,
        }
    )

    provider = OpenWakeWordProvider(
        model_name="hey_jarvis",
        threshold=0.5,
        model=model,
    )

    detection = await provider.process(np.zeros(1280, dtype=np.int16))

    assert detection is not None
    assert detection.model_name == "hey_jarvis"
    assert detection.score == pytest.approx(0.82)


@pytest.mark.asyncio
async def test_score_below_threshold_is_ignored() -> None:
    """Low confidence should not activate."""

    provider = OpenWakeWordProvider(
        model_name="hey_jarvis",
        threshold=0.5,
        model=FakeOpenWakeWordModel(
            {
                "hey_jarvis_v0.1": 0.49,
            }
        ),
    )

    detection = await provider.process(np.zeros(1280, dtype=np.int16))

    assert detection is None


@pytest.mark.asyncio
async def test_unrelated_model_is_ignored() -> None:
    """Other model predictions should not activate."""

    provider = OpenWakeWordProvider(
        model_name="hey_jarvis",
        threshold=0.5,
        model=FakeOpenWakeWordModel(
            {
                "alexa_v0.1": 0.99,
            }
        ),
    )

    detection = await provider.process(np.zeros(1280, dtype=np.int16))

    assert detection is None


@pytest.mark.asyncio
async def test_empty_frame_is_ignored() -> None:
    """An empty frame should not call the model."""

    model = FakeOpenWakeWordModel(
        {
            "hey_jarvis": 1.0,
        }
    )

    provider = OpenWakeWordProvider(
        model_name="hey_jarvis",
        model=model,
    )

    detection = await provider.process(np.array([], dtype=np.int16))

    assert detection is None
    assert model.frames == []


@pytest.mark.parametrize(
    "threshold",
    [0, -0.1, 1.1],
)
def test_invalid_threshold_is_rejected(
    threshold: float,
) -> None:
    """Activation threshold must be valid."""

    with pytest.raises(
        ValueError,
        match="threshold",
    ):
        OpenWakeWordProvider(
            threshold=threshold,
            model=FakeOpenWakeWordModel({}),
        )
