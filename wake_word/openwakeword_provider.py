"""Wake-word provider backed by openWakeWord."""

import asyncio
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from openwakeword import MODELS
from openwakeword.model import Model

from wake_word.errors import (
    WakeWordProviderUnavailableError,
)
from wake_word.provider import WakeWordProvider
from wake_word.result import WakeWordDetection


class OpenWakeWordModelProtocol(Protocol):
    """Subset of the openWakeWord model used here."""

    def predict(
        self,
        audio: NDArray[np.int16],
    ) -> Mapping[str, float]:
        """Return wake-word confidence scores."""


class OpenWakeWordProvider(WakeWordProvider):
    """Detect one pretrained openWakeWord phrase."""

    def __init__(
        self,
        *,
        model_name: str = "hey_jarvis",
        threshold: float = 0.5,
        vad_threshold: float = 0.0,
        model: OpenWakeWordModelProtocol | None = None,
    ) -> None:
        cleaned_model_name = model_name.strip().lower()

        if not cleaned_model_name:
            raise ValueError("Wake-word model name cannot be empty.")

        if not 0 < threshold <= 1:
            raise ValueError(
                "Wake-word threshold must be greater than zero and no greater than one."
            )

        if not 0 <= vad_threshold <= 1:
            raise ValueError("Wake-word VAD threshold must be between zero and one.")

        self._model_name = cleaned_model_name
        self._threshold = threshold
        self._vad_threshold = vad_threshold
        self._model = model
        self._model_lock = asyncio.Lock()
        self._debug_enabled = os.getenv(
            "BALANSPEAKER_WAKE_WORD_DEBUG",
            "false",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._debug_max_score = 0.0
        self._debug_peak_pcm = 0
        self._debug_rms_pcm = 0.0
        self._debug_frame_count = 0
        self._debug_started_at = time.monotonic()

    @property
    def name(self) -> str:
        """Return the provider name."""

        return f"openwakeword:{self._model_name}"

    @property
    def sample_rate(self) -> int:
        """Return the required PCM sample rate."""

        return 16000

    @property
    def model_name(self) -> str:
        """Return the configured model key."""

        return self._model_name

    @property
    def threshold(self) -> float:
        """Return the activation threshold."""

        return self._threshold

    @property
    def is_loaded(self) -> bool:
        """Return whether the model has been loaded."""

        return self._model is not None

    async def process(
        self,
        audio: NDArray[np.int16],
    ) -> WakeWordDetection | None:
        """Process one 16-bit PCM frame."""

        normalized_audio = np.asarray(
            audio,
            dtype=np.int16,
        ).reshape(-1)

        if normalized_audio.size == 0:
            return None

        model = await self._get_or_load_model()

        try:
            predictions = await asyncio.to_thread(
                model.predict,
                normalized_audio,
            )
        except Exception as error:
            raise WakeWordProviderUnavailableError(
                "openWakeWord prediction failed."
            ) from error

        selected_score = self._find_model_score(predictions)
        self._report_debug_frame(
            score=selected_score,
            audio=normalized_audio,
        )

        if selected_score < self._threshold:
            return None

        return WakeWordDetection(
            model_name=self._model_name,
            score=selected_score,
        )

    def _report_debug_frame(
        self,
        *,
        score: float,
        audio: NDArray[np.int16],
    ) -> None:
        """Periodically report whether live frames reach the model."""

        if not self._debug_enabled:
            return

        self._debug_frame_count += 1
        self._debug_max_score = max(self._debug_max_score, score)
        pcm_float = audio.astype(np.float32)
        self._debug_peak_pcm = max(
            self._debug_peak_pcm,
            int(np.max(np.abs(pcm_float))),
        )
        self._debug_rms_pcm = max(
            self._debug_rms_pcm,
            float(np.sqrt(np.mean(np.square(pcm_float)))),
        )
        now = time.monotonic()

        if now - self._debug_started_at < 5.0:
            return

        print(
            "\n[WAKE DEBUG] Processed "
            f"{self._debug_frame_count} frames; highest score "
            f"{self._debug_max_score:.4f}; peak PCM "
            f"{self._debug_peak_pcm}; maximum RMS "
            f"{self._debug_rms_pcm:.1f}."
        )
        self._debug_frame_count = 0
        self._debug_max_score = 0.0
        self._debug_peak_pcm = 0
        self._debug_rms_pcm = 0.0
        self._debug_started_at = now

    async def _get_or_load_model(
        self,
    ) -> OpenWakeWordModelProtocol:
        """Load the selected pretrained model once."""

        if self._model is not None:
            return self._model

        async with self._model_lock:
            if self._model is not None:
                return self._model

            model_path = self._resolve_model_path()

            try:
                self._model = await asyncio.to_thread(
                    Model,
                    wakeword_models=[str(model_path)],
                    inference_framework="onnx",
                    vad_threshold=self._vad_threshold,
                )
            except Exception as error:
                raise WakeWordProviderUnavailableError(
                    "The openWakeWord model could not be "
                    "loaded. Run the model download command "
                    "before starting Balanspeaker."
                ) from error

        return self._model

    def _resolve_model_path(self) -> Path:
        """Resolve the configured pretrained ONNX model."""

        model_details: dict[str, Any] | None = MODELS.get(self._model_name)

        if model_details is None:
            raise WakeWordProviderUnavailableError(
                f"Unknown openWakeWord model: {self._model_name}."
            )

        raw_path = model_details.get("model_path")

        if not isinstance(raw_path, str):
            raise WakeWordProviderUnavailableError(
                "The openWakeWord model path is invalid."
            )

        model_path = Path(raw_path)

        if model_path.suffix.lower() == ".tflite":
            model_path = model_path.with_suffix(".onnx")

        if not model_path.is_file():
            raise WakeWordProviderUnavailableError(
                "The openWakeWord ONNX model was not found. "
                "Run openwakeword.utils.download_models()."
            )

        return model_path

    def _find_model_score(
        self,
        predictions: Mapping[str, float],
    ) -> float:
        """Find the configured model's score."""

        target = _normalize_model_name(self._model_name)

        matching_scores: list[float] = []

        for prediction_name, raw_score in predictions.items():
            normalized_name = _normalize_model_name(prediction_name)

            if (
                normalized_name == target
                or target in normalized_name
                or normalized_name in target
            ):
                matching_scores.append(_safe_score(raw_score))

        if not matching_scores:
            return 0.0

        return max(matching_scores)


def _normalize_model_name(value: str) -> str:
    """Normalize model names for comparison."""

    normalized = value.strip().lower()

    for character in {
        " ",
        "-",
        ".",
        "/",
        "\\",
    }:
        normalized = normalized.replace(
            character,
            "_",
        )

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    for suffix in {
        "_v0_1_onnx",
        "_v0_1_tflite",
        "_onnx",
        "_tflite",
    }:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]

    return normalized.strip("_")


def _safe_score(value: Any) -> float:
    """Convert provider output into a valid score."""

    if isinstance(value, bool):
        return 0.0

    if not isinstance(value, int | float):
        return 0.0

    return min(1.0, max(0.0, float(value)))
