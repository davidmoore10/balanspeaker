"""Tests for wake-word microphone handling."""

import asyncio
from contextlib import AbstractContextManager
from typing import Any

import numpy as np
import pytest
import sounddevice as sd

from wake_word import listener
from wake_word.listener import WakeWordListener
from wake_word.provider import WakeWordProvider
from wake_word.result import WakeWordDetection


class FailingProvider(WakeWordProvider):
    """Provider that exposes frame-processor failures."""

    @property
    def name(self) -> str:
        return "failing"

    @property
    def sample_rate(self) -> int:
        return 16000

    async def process(
        self,
        audio: np.ndarray,
    ) -> WakeWordDetection | None:
        del audio
        raise RuntimeError("model processing failed")


class CallbackStream(AbstractContextManager["CallbackStream"]):
    """Stream that submits one frame when entered."""

    def __init__(self, **kwargs: Any) -> None:
        self._callback = kwargs["callback"]
        self._blocksize = kwargs["blocksize"]

    def __enter__(self) -> "CallbackStream":
        self._callback(
            np.zeros((self._blocksize, 1), dtype=np.float32),
            self._blocksize,
            None,
            None,
        )
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_default_microphone_falls_back_to_working_wasapi_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken Windows default should not stop wake-word listening."""

    opened_devices: list[int | None] = []
    working_stream = object()

    def input_stream(**kwargs: object) -> object:
        device = kwargs.get("device")
        assert device is None or isinstance(device, int)
        opened_devices.append(device)

        if device == 2:
            return working_stream

        raise sd.PortAudioError("cannot open")

    monkeypatch.setattr(listener.sd, "InputStream", input_stream)
    monkeypatch.setattr(
        listener.sd,
        "query_devices",
        lambda: [
            {"max_input_channels": 1, "hostapi": 0},
            {"max_input_channels": 0, "hostapi": 1},
            {"max_input_channels": 1, "hostapi": 2},
        ],
    )
    monkeypatch.setattr(
        listener.sd,
        "query_hostapis",
        lambda: [
            {"name": "MME"},
            {"name": "DirectSound"},
            {"name": "Windows WASAPI"},
        ],
    )

    stream = listener.create_input_stream(device=None)

    assert stream is working_stream
    assert opened_devices == [None, 2]


def test_explicit_microphone_failure_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly selected device should remain authoritative."""

    monkeypatch.setattr(
        listener.sd,
        "InputStream",
        lambda **kwargs: (_ for _ in ()).throw(sd.PortAudioError("cannot open")),
    )

    with pytest.raises(sd.PortAudioError, match="cannot open"):
        listener.create_input_stream(device=7)


def test_fallback_uses_native_rate_and_resamples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device limited to its native rate should still feed 16 kHz frames."""

    received: list[np.ndarray] = []
    opened: list[dict[str, object]] = []

    def input_stream(**kwargs: object) -> object:
        opened.append(kwargs)

        if kwargs.get("device") == 0 and kwargs.get("samplerate") == 48000:
            return object()

        raise sd.PortAudioError("cannot open")

    monkeypatch.setattr(listener.sd, "InputStream", input_stream)
    monkeypatch.setattr(
        listener.sd,
        "query_devices",
        lambda: [{"max_input_channels": 1, "hostapi": 0, "default_samplerate": 48000}],
    )
    monkeypatch.setattr(
        listener.sd,
        "query_hostapis",
        lambda: [{"name": "Windows WDM-KS"}],
    )

    listener.create_input_stream(
        device=None,
        samplerate=16000,
        blocksize=1280,
        callback=lambda data, *_: received.append(data),
    )
    native_callback = opened[-1]["callback"]
    assert callable(native_callback)
    native_callback(np.zeros((3840, 1), dtype=np.float32), 3840, None, None)

    assert opened[-1]["blocksize"] == 3840
    assert received[0].shape == (1280, 1)


@pytest.mark.asyncio
async def test_frame_processor_failure_is_propagated() -> None:
    """A failed model task must not leave the listener waiting forever."""

    wake_listener = WakeWordListener(
        provider=FailingProvider(),
        stream_factory=CallbackStream,
    )

    with pytest.raises(RuntimeError, match="model processing failed"):
        await asyncio.wait_for(
            wake_listener.wait_for_activation(),
            timeout=1,
        )
