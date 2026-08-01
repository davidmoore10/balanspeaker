"""Microphone recording for voice interaction."""

import asyncio
import queue
from collections import deque
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from speech_recognition.errors import (
    EmptyTranscriptionError,
    MicrophoneUnavailableError,
)
from speech_recognition.silence import (
    RecordingCompletionReason,
    SilenceDetectionSettings,
    SpeechActivityDetector,
)

InputFunction = Callable[[str], str]


class InputStreamFactory(Protocol):
    """Factory for microphone input streams."""

    def __call__(
        self,
        **kwargs: Any,
    ) -> AbstractContextManager[Any]:
        """Create an input stream context manager."""


@dataclass(frozen=True, slots=True)
class RecordedAudio:
    """Mono microphone audio and its sample rate."""

    samples: NDArray[np.float32]
    sample_rate: int


def create_input_stream(
    **kwargs: Any,
) -> AbstractContextManager[Any]:
    """Create the production sounddevice input stream."""

    return sd.InputStream(**kwargs)


class MicrophoneRecorder:
    """Capture manual or silence-terminated microphone audio."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
        block_duration_seconds: float = 0.1,
        silence_settings: (SilenceDetectionSettings | None) = None,
        input_function: InputFunction = input,
        stream_factory: InputStreamFactory = (create_input_stream),
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("Microphone sample rate must be greater than zero.")

        if channels != 1:
            raise ValueError("Balanspeaker currently requires mono microphone audio.")

        if block_duration_seconds <= 0:
            raise ValueError("Microphone block duration must be greater than zero.")

        self._sample_rate = sample_rate
        self._channels = channels
        self._device = device
        self._block_duration_seconds = block_duration_seconds
        self._silence_settings = silence_settings or SilenceDetectionSettings()
        self._input_function = input_function
        self._stream_factory = stream_factory

    @property
    def sample_rate(self) -> int:
        """Return the configured sample rate."""

        return self._sample_rate

    @property
    def device(self) -> int | str | None:
        """Return the configured input device."""

        return self._device

    @property
    def silence_settings(
        self,
    ) -> SilenceDetectionSettings:
        """Return automatic recording settings."""

        return self._silence_settings

    async def record_until_silence(
        self,
    ) -> RecordedAudio:
        """Record until speech is followed by silence."""

        return await asyncio.to_thread(self._record_until_silence_sync)

    async def record_push_to_talk(
        self,
    ) -> RecordedAudio:
        """Record between user-controlled Enter presses."""

        return await asyncio.to_thread(self._record_push_to_talk_sync)

    def _record_until_silence_sync(
        self,
    ) -> RecordedAudio:
        """Perform blocking silence-terminated recording."""

        block_size = max(
            1,
            round(self._sample_rate * self._block_duration_seconds),
        )

        detector = SpeechActivityDetector(
            sample_rate=self._sample_rate,
            settings=self._silence_settings,
        )

        audio_queue: queue.Queue[NDArray[np.float32]] = queue.Queue()

        captured_blocks: list[NDArray[np.float32]] = []

        pre_roll_block_count = max(
            1,
            round(
                self._silence_settings.pre_roll_seconds / self._block_duration_seconds
            ),
        )

        pre_roll: deque[NDArray[np.float32]] = deque(maxlen=pre_roll_block_count)

        def callback(
            input_data: NDArray[np.float32],
            frames: int,
            time_info: Any,
            status: sd.CallbackFlags,
        ) -> None:
            del frames
            del time_info
            del status

            mono_block = np.asarray(
                input_data[:, 0],
                dtype=np.float32,
            ).copy()

            audio_queue.put(mono_block)

        print("Listening...")

        try:
            with self._stream_factory(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                device=self._device,
                blocksize=block_size,
                callback=callback,
            ):
                while not detector.complete:
                    try:
                        block = audio_queue.get(
                            timeout=max(
                                1.0,
                                self._block_duration_seconds * 4,
                            )
                        )
                    except queue.Empty as error:
                        raise MicrophoneUnavailableError(
                            "The microphone stopped providing audio."
                        ) from error

                    was_started = detector.speech_started

                    if not was_started:
                        pre_roll.append(block)

                    detector.process(block)

                    if not was_started and detector.speech_started:
                        captured_blocks.extend(pre_roll)
                        pre_roll.clear()
                    elif was_started:
                        captured_blocks.append(block)

        except (
            sd.PortAudioError,
            OSError,
        ) as error:
            raise MicrophoneUnavailableError(
                "The microphone could not be opened."
            ) from error

        if detector.completion_reason == RecordingCompletionReason.NO_SPEECH:
            raise EmptyTranscriptionError(
                "No speech was detected before the listening timeout."
            )

        if not detector.speech_started:
            raise EmptyTranscriptionError("No speech was detected.")

        if not captured_blocks:
            raise MicrophoneUnavailableError("No microphone audio was captured.")

        samples = np.concatenate(captured_blocks).astype(
            np.float32,
            copy=False,
        )

        return RecordedAudio(
            samples=samples,
            sample_rate=self._sample_rate,
        )

    def _record_push_to_talk_sync(
        self,
    ) -> RecordedAudio:
        """Perform manual push-to-talk recording."""

        self._input_function("Press Enter to start recording.")

        captured_blocks: list[NDArray[np.float32]] = []

        def callback(
            input_data: NDArray[np.float32],
            frames: int,
            time_info: Any,
            status: sd.CallbackFlags,
        ) -> None:
            del frames
            del time_info
            del status

            captured_blocks.append(
                np.asarray(
                    input_data[:, 0],
                    dtype=np.float32,
                ).copy()
            )

        try:
            with self._stream_factory(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                device=self._device,
                callback=callback,
            ):
                self._input_function("Recording... Press Enter to stop.")
        except (
            sd.PortAudioError,
            OSError,
        ) as error:
            raise MicrophoneUnavailableError(
                "The microphone could not be opened."
            ) from error

        if not captured_blocks:
            raise MicrophoneUnavailableError("No microphone audio was captured.")

        samples = np.concatenate(captured_blocks).astype(
            np.float32,
            copy=False,
        )

        if samples.size == 0:
            raise MicrophoneUnavailableError("No microphone audio was captured.")

        return RecordedAudio(
            samples=samples,
            sample_rate=self._sample_rate,
        )


def list_input_devices() -> tuple[str, ...]:
    """Return descriptions of microphone devices."""

    try:
        devices = sd.query_devices()
    except (
        sd.PortAudioError,
        OSError,
    ) as error:
        raise MicrophoneUnavailableError(
            "Audio devices could not be queried."
        ) from error

    descriptions: list[str] = []

    for index, device in enumerate(devices):
        maximum_input_channels = int(device["max_input_channels"])

        if maximum_input_channels <= 0:
            continue

        descriptions.append(
            f"{index}: {device['name']} ({maximum_input_channels} input channel(s))"
        )

    return tuple(descriptions)
