"""Local speech recognition backed by faster-whisper."""

import asyncio
from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np
from faster_whisper import WhisperModel
from numpy.typing import NDArray

from speech_recognition.errors import (
    EmptyTranscriptionError,
    TranscriptionUnavailableError,
)
from speech_recognition.provider import SpeechToTextProvider
from speech_recognition.result import TranscriptionResult


class WhisperSegmentProtocol(Protocol):
    """Subset of a faster-whisper segment used here."""

    text: str


class WhisperInfoProtocol(Protocol):
    """Subset of transcription metadata used here."""

    language: str
    language_probability: float


class WhisperModelProtocol(Protocol):
    """Subset of WhisperModel used by Balanspeaker."""

    def transcribe(
        self,
        audio: NDArray[np.float32],
        **kwargs: Any,
    ) -> tuple[
        Iterable[WhisperSegmentProtocol],
        WhisperInfoProtocol,
    ]:
        """Transcribe one audio array."""


class FasterWhisperProvider(SpeechToTextProvider):
    """Transcribe speech locally with faster-whisper."""

    def __init__(
        self,
        *,
        model_name: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 5,
        model: WhisperModelProtocol | None = None,
    ) -> None:
        cleaned_model_name = model_name.strip()
        cleaned_device = device.strip()
        cleaned_compute_type = compute_type.strip()
        cleaned_language = language.strip()

        if not cleaned_model_name:
            raise ValueError("Speech-recognition model cannot be empty.")

        if not cleaned_device:
            raise ValueError("Speech-recognition device cannot be empty.")

        if not cleaned_compute_type:
            raise ValueError("Speech-recognition compute type cannot be empty.")

        if not cleaned_language:
            raise ValueError("Speech-recognition language cannot be empty.")

        if beam_size <= 0:
            raise ValueError("Speech-recognition beam size must be greater than zero.")

        self._model_name = cleaned_model_name
        self._device = cleaned_device
        self._compute_type = cleaned_compute_type
        self._language = cleaned_language
        self._beam_size = beam_size
        self._model = model
        self._model_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Return the provider name."""

        return f"faster-whisper:{self._model_name}"

    @property
    def is_loaded(self) -> bool:
        """Return whether the Whisper model has been loaded."""

        return self._model is not None

    async def transcribe(
        self,
        *,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> TranscriptionResult:
        """Transcribe mono floating-point audio."""

        if sample_rate <= 0:
            raise ValueError("Audio sample rate must be greater than zero.")

        normalized_audio = np.asarray(
            audio,
            dtype=np.float32,
        ).reshape(-1)

        if normalized_audio.size == 0:
            raise EmptyTranscriptionError("No audio was supplied for transcription.")

        # Whisper expects 16 kHz audio. MicrophoneRecorder currently
        # records directly at this rate, avoiding a resampling step.
        if sample_rate != 16000:
            raise ValueError("FasterWhisperProvider currently requires 16 kHz audio.")

        model = await self._get_or_load_model()

        try:
            segments, info = await asyncio.to_thread(
                model.transcribe,
                normalized_audio,
                language=self._language,
                beam_size=self._beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
            )

            segment_list = await asyncio.to_thread(
                list,
                segments,
            )
        except Exception as error:
            raise TranscriptionUnavailableError(
                "Local speech transcription failed."
            ) from error

        text = " ".join(
            segment.text.strip() for segment in segment_list if segment.text.strip()
        ).strip()

        if not text:
            raise EmptyTranscriptionError("No speech was detected.")

        language = getattr(info, "language", None)
        probability = getattr(
            info,
            "language_probability",
            None,
        )

        return TranscriptionResult(
            text=text,
            language=(language if isinstance(language, str) else None),
            language_probability=(
                float(probability) if isinstance(probability, int | float) else None
            ),
        )

    async def _get_or_load_model(
        self,
    ) -> WhisperModelProtocol:
        """Load the model once and reuse it."""

        if self._model is not None:
            return self._model

        async with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                self._model = await asyncio.to_thread(
                    WhisperModel,
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                )
            except Exception as error:
                raise TranscriptionUnavailableError(
                    "The local speech-recognition model could not be loaded."
                ) from error

        return self._model
