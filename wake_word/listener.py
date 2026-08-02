"""Continuous microphone listening for a wake phrase."""

import asyncio
import os
import queue
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Protocol

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from wake_word.errors import (
    WakeWordMicrophoneError,
)
from wake_word.provider import WakeWordProvider
from wake_word.result import WakeWordDetection


class InputStreamFactory(Protocol):
    """Factory for sounddevice-compatible streams."""

    def __call__(
        self,
        **kwargs: Any,
    ) -> AbstractContextManager[Any]:
        """Create an input stream."""


def create_input_stream(
    **kwargs: Any,
) -> AbstractContextManager[Any]:
    """Create a microphone stream, falling back from a broken default device."""

    try:
        return sd.InputStream(**kwargs)
    except sd.PortAudioError as default_error:
        if kwargs.get("device") is not None:
            device_index = _device_index(kwargs["device"])

            if device_index is None:
                raise

            native_stream = _open_at_native_rate(
                kwargs,
                device_index=device_index,
            )

            if native_stream is not None:
                details = sd.query_devices()[device_index]
                print(
                    "[AUDIO] Using explicit input device "
                    f"{device_index}: {details.get('name', 'unknown')} at "
                    f"{int(details['default_samplerate'])} Hz; resampling to "
                    f"{int(kwargs['samplerate'])} Hz."
                )
                return native_stream

            raise

        # On Windows, PortAudio's default MME input can be advertised as
        # compatible by check_input_settings() and still fail when opened.
        # Prefer modern host APIs, then try the remaining input devices.
        devices = sd.query_devices()
        host_apis = sd.query_hostapis()
        candidates: list[tuple[int, int]] = []

        for index, device in enumerate(devices):
            if int(device.get("max_input_channels", 0)) < 1:
                continue

            host_api_index = int(device.get("hostapi", -1))
            host_api_name = ""

            if 0 <= host_api_index < len(host_apis):
                host_api_name = str(host_apis[host_api_index].get("name", ""))

            normalized_name = host_api_name.lower()
            priority = 0

            if "wasapi" in normalized_name:
                priority = 2
            elif "directsound" in normalized_name:
                priority = 1

            candidates.append((priority, index))

        for _, device_index in sorted(
            candidates,
            key=lambda candidate: (-candidate[0], candidate[1]),
        ):
            fallback_kwargs = {**kwargs, "device": device_index}
            device = devices[device_index]

            try:
                stream = sd.InputStream(**fallback_kwargs)
                print(
                    "[AUDIO] Using input device "
                    f"{device_index}: {device.get('name', 'unknown')}."
                )
                return stream
            except (sd.PortAudioError, OSError):
                stream = _open_at_native_rate(
                    fallback_kwargs,
                    device_index=device_index,
                )

                if stream is not None:
                    print(
                        "[AUDIO] Using input device "
                        f"{device_index}: {device.get('name', 'unknown')} at "
                        f"{int(device['default_samplerate'])} Hz; resampling to "
                        f"{int(kwargs['samplerate'])} Hz."
                    )
                    return stream

        raise default_error


def _device_index(device: object) -> int | None:
    """Resolve an explicit sounddevice input selector to its index."""

    if isinstance(device, int):
        return device

    if not isinstance(device, str):
        return None

    normalized = device.strip().lower()

    for index, details in enumerate(sd.query_devices()):
        if (
            int(details.get("max_input_channels", 0)) > 0
            and normalized in str(details.get("name", "")).lower()
        ):
            return index

    return None


def _open_at_native_rate(
    kwargs: dict[str, Any],
    *,
    device_index: int,
) -> AbstractContextManager[Any] | None:
    """Open one input at its native rate and resample its callbacks."""

    devices = sd.query_devices()

    if not 0 <= device_index < len(devices):
        return None

    native_rate = int(devices[device_index].get("default_samplerate", 0))
    requested_rate = int(kwargs.get("samplerate", 0))

    if (
        native_rate <= 0
        or requested_rate <= 0
        or native_rate == requested_rate
        or "callback" not in kwargs
    ):
        return None

    native_kwargs = _resampling_stream_kwargs(
        {**kwargs, "device": device_index},
        input_rate=native_rate,
        output_rate=requested_rate,
    )

    try:
        return sd.InputStream(**native_kwargs)
    except (sd.PortAudioError, OSError):
        return None


def _resampling_stream_kwargs(
    kwargs: dict[str, Any],
    *,
    input_rate: int,
    output_rate: int,
) -> dict[str, Any]:
    """Adapt a native-rate stream to the rate required by openWakeWord."""

    original_callback = kwargs["callback"]
    output_blocksize = int(kwargs.get("blocksize", 0))
    input_blocksize = max(
        1,
        round(output_blocksize * input_rate / output_rate),
    )

    def resampling_callback(
        input_data: NDArray[np.float32],
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        del frames

        output_length = max(
            1,
            round(len(input_data) * output_rate / input_rate),
        )
        source_positions = np.arange(len(input_data), dtype=np.float64)
        target_positions = np.linspace(
            0,
            len(input_data) - 1,
            output_length,
        )
        resampled = np.empty(
            (output_length, input_data.shape[1]),
            dtype=np.float32,
        )

        for channel in range(input_data.shape[1]):
            resampled[:, channel] = np.interp(
                target_positions,
                source_positions,
                input_data[:, channel],
            )

        original_callback(
            resampled,
            output_length,
            time_info,
            status,
        )

    return {
        **kwargs,
        "samplerate": input_rate,
        "blocksize": input_blocksize,
        "callback": resampling_callback,
    }



class WakeWordListener:
    """Continuously capture audio until a wake word is found."""

    def __init__(
        self,
        *,
        provider: WakeWordProvider,
        device: int | str | None = None,
        frame_duration_seconds: float = 0.08,
        cooldown_seconds: float = 2.0,
        stream_factory: InputStreamFactory = (create_input_stream),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if frame_duration_seconds <= 0:
            raise ValueError("Wake-word frame duration must be greater than zero.")

        if cooldown_seconds < 0:
            raise ValueError("Wake-word cooldown cannot be negative.")

        self._provider = provider
        self._device = device
        self._frame_duration_seconds = frame_duration_seconds
        self._cooldown_seconds = cooldown_seconds
        self._stream_factory = stream_factory
        self._monotonic = monotonic

        self._shutdown_event = threading.Event()
        self._pause_event = threading.Event()
        self._inactive_event = threading.Event()
        self._inactive_event.set()

        self._last_activation_time = float("-inf")
        self._debug_enabled = os.getenv(
            "BALANSPEAKER_WAKE_WORD_DEBUG",
            "false",
        ).strip().lower() in {"1", "true", "yes", "on"}

    @property
    def provider(self) -> WakeWordProvider:
        """Return the configured provider."""

        return self._provider

    @property
    def is_paused(self) -> bool:
        """Return whether listening is paused."""

        return self._pause_event.is_set()

    @property
    def is_stopped(self) -> bool:
        """Return whether shutdown was requested."""

        return self._shutdown_event.is_set()

    async def wait_for_activation(
        self,
    ) -> WakeWordDetection | None:
        """Listen until detection, pause or shutdown."""

        if self.is_stopped or self.is_paused:
            return None

        return await self._wait_for_activation_async()

    async def _wait_for_activation_async(
        self,
    ) -> WakeWordDetection | None:
        """Capture frames without blocking the event loop."""

        frame_queue: queue.Queue[NDArray[np.int16]] = queue.Queue(maxsize=20)

        frame_size = max(
            1,
            round(self._provider.sample_rate * self._frame_duration_seconds),
        )

        loop = asyncio.get_running_loop()

        detection_future: asyncio.Future[WakeWordDetection | None] = (
            loop.create_future()
        )
        callback_count = 0

        def callback(
            input_data: NDArray[np.float32],
            frames: int,
            time_info: Any,
            status: sd.CallbackFlags,
        ) -> None:
            nonlocal callback_count
            del frames
            del time_info
            del status

            if self._shutdown_event.is_set() or self._pause_event.is_set():
                return

            callback_count += 1

            mono = np.asarray(
                input_data[:, 0],
                dtype=np.float32,
            )

            pcm = np.clip(
                mono * 32767,
                -32768,
                32767,
            ).astype(np.int16)

            try:
                frame_queue.put_nowait(pcm.copy())
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    frame_queue.put_nowait(pcm.copy())
                except queue.Full:
                    pass

        async def process_frames() -> None:
            """Process microphone frames until completion."""

            while not self._shutdown_event.is_set() and not self._pause_event.is_set():
                try:
                    frame = await asyncio.to_thread(
                        frame_queue.get,
                        True,
                        0.2,
                    )
                except queue.Empty:
                    continue

                detection = await self._provider.process(frame)

                if detection is None:
                    continue

                now = self._monotonic()

                if now - self._last_activation_time < self._cooldown_seconds:
                    continue

                self._last_activation_time = now

                if not detection_future.done():
                    detection_future.set_result(detection)

                return

            if not detection_future.done():
                detection_future.set_result(None)

        self._inactive_event.clear()

        try:
            with self._stream_factory(
                samplerate=self._provider.sample_rate,
                channels=1,
                dtype="float32",
                device=self._device,
                blocksize=frame_size,
                callback=callback,
            ):
                processor_task = asyncio.create_task(
                    process_frames(),
                    name="wake-word-frame-processor",
                )

                try:
                    next_debug_time = self._monotonic() + 5.0

                    while not detection_future.done():
                        if processor_task.done():
                            # Propagate provider/model failures instead of
                            # leaving the listener waiting forever on the
                            # separate detection future.
                            await processor_task

                            if not detection_future.done():
                                detection_future.set_result(None)

                            break

                        if self._shutdown_event.is_set() or self._pause_event.is_set():
                            detection_future.set_result(None)
                            break

                        now = self._monotonic()

                        if self._debug_enabled and now >= next_debug_time:
                            print(
                                "\n[WAKE STREAM DEBUG] Callbacks "
                                f"{callback_count}; queued frames "
                                f"{frame_queue.qsize()}; processor "
                                f"{'done' if processor_task.done() else 'running'}."
                            )
                            next_debug_time = now + 5.0

                        await asyncio.sleep(0.05)

                    return await detection_future
                finally:
                    processor_task.cancel()

                    await asyncio.gather(
                        processor_task,
                        return_exceptions=True,
                    )
        except (
            sd.PortAudioError,
            OSError,
        ) as error:
            raise WakeWordMicrophoneError(
                "The wake-word microphone stream could not be opened."
            ) from error
        finally:
            self._inactive_event.set()

    async def pause(self) -> None:
        """Pause and wait for the microphone stream to close."""

        self._pause_event.set()

        await asyncio.to_thread(
            self._inactive_event.wait,
            2.0,
        )

    def resume(self) -> None:
        """Allow wake-word listening to continue."""

        if not self._shutdown_event.is_set():
            self._pause_event.clear()

    async def stop(self) -> None:
        """Stop wake-word listening permanently."""

        self._shutdown_event.set()
        self._pause_event.set()

        await asyncio.to_thread(
            self._inactive_event.wait,
            2.0,
        )
