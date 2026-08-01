"""Local text-to-speech provider backed by Piper."""

import asyncio
import importlib
import sys
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from piper import PiperVoice

from speech.errors import (
    SpeechProviderUnavailableError,
    SpeechSynthesisError,
)
from speech.provider import SpeechProvider


class PiperVoiceProtocol(Protocol):
    """Subset of PiperVoice used by Balanspeaker."""

    def synthesize_wav(
        self,
        text: str,
        wav_file: wave.Wave_write,
    ) -> None:
        """Write synthesised speech to a WAV file."""


VoiceLoader = Callable[[Path], PiperVoiceProtocol]
WavPlayer = Callable[[Path], None]
PlaybackStopper = Callable[[], None]


def load_piper_voice(model_path: Path) -> PiperVoiceProtocol:
    """Load a Piper voice from disk."""

    return PiperVoice.load(str(model_path))


def play_wav_on_windows(wav_path: Path) -> None:
    """Play a WAV file through the Windows audio system."""

    if sys.platform != "win32":
        raise SpeechProviderUnavailableError(
            "The current WAV player only supports Windows."
        )

    winsound = importlib.import_module("winsound")

    try:
        winsound.PlaySound(
            str(wav_path),
            winsound.SND_FILENAME,
        )
    except RuntimeError as error:
        raise SpeechSynthesisError(
            "Windows could not play the generated speech."
        ) from error


def stop_wav_on_windows() -> None:
    """Stop WAV playback through the Windows audio system."""

    if sys.platform != "win32":
        return

    winsound = importlib.import_module("winsound")

    try:
        winsound.PlaySound(
            None,
            winsound.SND_PURGE,
        )
    except RuntimeError:
        return


class PiperSpeechProvider(SpeechProvider):
    """Generate and play speech using a local Piper voice."""

    def __init__(
        self,
        *,
        model_path: Path,
        voice_loader: VoiceLoader = load_piper_voice,
        wav_player: WavPlayer = play_wav_on_windows,
        playback_stopper: PlaybackStopper = stop_wav_on_windows,
    ) -> None:
        self._model_path = model_path
        self._voice_loader = voice_loader
        self._wav_player = wav_player
        self._playback_stopper = playback_stopper
        self._voice: PiperVoiceProtocol | None = None
        self._speech_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Return the provider name."""

        return f"piper:{self._model_path.stem}"

    @property
    def model_path(self) -> Path:
        """Return the configured voice-model path."""

        return self._model_path

    @property
    def is_loaded(self) -> bool:
        """Return whether the voice has been loaded."""

        return self._voice is not None

    async def speak(self, text: str) -> None:
        """Synthesise and play one utterance."""

        cleaned_text = text.strip()

        if not cleaned_text:
            return

        async with self._speech_lock:
            await asyncio.to_thread(
                self._speak_sync,
                cleaned_text,
            )

    async def stop(self) -> None:
        """Stop current playback when supported."""

        await asyncio.to_thread(self._playback_stopper)

    def _speak_sync(self, text: str) -> None:
        """Perform blocking synthesis and playback."""

        voice = self._get_or_load_voice()

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

            with wave.open(str(temporary_path), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)

            self._wav_player(temporary_path)
        except SpeechProviderUnavailableError:
            raise
        except Exception as error:
            raise SpeechSynthesisError(
                "Piper could not synthesise or play speech."
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _get_or_load_voice(self) -> PiperVoiceProtocol:
        """Load the configured voice once and reuse it."""

        if self._voice is not None:
            return self._voice

        if not self._model_path.is_file():
            raise SpeechProviderUnavailableError(
                "The configured Piper voice model was not found."
            )

        config_path = Path(f"{self._model_path}.json")

        if not config_path.is_file():
            raise SpeechProviderUnavailableError(
                "The Piper voice configuration file was not found."
            )

        try:
            self._voice = self._voice_loader(self._model_path)
        except Exception as error:
            raise SpeechProviderUnavailableError(
                "The configured Piper voice could not be loaded."
            ) from error

        return self._voice
