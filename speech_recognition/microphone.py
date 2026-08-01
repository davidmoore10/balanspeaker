"""Microphone recording for push-to-talk interaction."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from speech_recognition.errors import MicrophoneUnavailableError

InputFunction = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class RecordedAudio:
    """Mono microphone audio and its sample rate."""

    samples: NDArray[np.float32]
    sample_rate: int


class MicrophoneRecorder:
    """Capture microphone audio between two Enter presses."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
        input_function: InputFunction = input,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("Microphone sample rate must be greater than zero.")

        if channels != 1:
            raise ValueError("Balanspeaker currently requires mono microphone audio.")

        self._sample_rate = sample_rate
        self._channels = channels
        self._device = device
        self._input_function = input_function

    @property
    def sample_rate(self) -> int:
        """Return the configured recording sample rate."""

        return self._sample_rate

    @property
    def device(self) -> int | str | None:
        """Return the configured input device."""

        return self._device

    async def record_push_to_talk(self) -> RecordedAudio:
        """Record between user-controlled start and stop prompts."""

        return await asyncio.to_thread(self._record_push_to_talk_sync)

    def _record_push_to_talk_sync(self) -> RecordedAudio:
        """Perform blocking push-to-talk recording."""

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

            if status:
                # PortAudio status is not necessarily fatal. Captured
                # audio is retained, and transcription may still work.
                pass

            captured_blocks.append(
                np.asarray(
                    input_data[:, 0],
                    dtype=np.float32,
                ).copy()
            )

        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                device=self._device,
                callback=callback,
            ):
                self._input_function("Recording... Press Enter to stop.")
        except (sd.PortAudioError, OSError) as error:
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
    """Return descriptions of available microphone devices."""

    try:
        devices = sd.query_devices()
    except (sd.PortAudioError, OSError) as error:
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
