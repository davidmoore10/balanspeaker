"""Coordination of interruptible speech playback."""

import asyncio

from speech.provider import SpeechProvider


class SpeechManager:
    """Run speech asynchronously and allow immediate interruption."""

    def __init__(
        self,
        *,
        provider: SpeechProvider,
    ) -> None:
        self._provider = provider
        self._active_task: asyncio.Task[None] | None = None
        self._current_text: str | None = None
        self._last_error: Exception | None = None

    @property
    def provider(self) -> SpeechProvider:
        """Return the configured speech provider."""

        return self._provider

    @property
    def is_speaking(self) -> bool:
        """Return whether speech playback is currently active."""

        return self._active_task is not None and not self._active_task.done()

    @property
    def current_text(self) -> str | None:
        """Return the text currently being spoken."""

        if not self.is_speaking:
            return None

        return self._current_text

    @property
    def last_error(self) -> Exception | None:
        """Return the latest speech error, if any."""

        return self._last_error

    async def speak(self, text: str) -> bool:
        """Begin speaking without blocking command processing.

        Returns False for empty text and True when speech starts.
        """

        cleaned_text = text.strip()

        if not cleaned_text:
            return False

        await self.interrupt()

        self._current_text = cleaned_text
        self._last_error = None

        self._active_task = asyncio.create_task(
            self._run_speech(cleaned_text),
            name="assistant-speech",
        )

        return True

    async def interrupt(self) -> bool:
        """Stop current speech immediately.

        Returns True when active speech was interrupted.
        """

        task = self._active_task

        if task is None or task.done():
            self._active_task = None
            self._current_text = None
            return False

        await self._provider.stop()

        task.cancel()

        await asyncio.gather(
            task,
            return_exceptions=True,
        )

        self._active_task = None
        self._current_text = None

        return True

    async def wait_until_idle(self) -> None:
        """Wait for the active utterance to finish."""

        task = self._active_task

        if task is None:
            return

        await asyncio.gather(
            task,
            return_exceptions=True,
        )

    async def close(self) -> None:
        """Stop speech and release active playback."""

        await self.interrupt()
        await self._provider.stop()

    async def _run_speech(self, text: str) -> None:
        """Run one speech request and record provider errors."""

        try:
            await self._provider.speak(text)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._last_error = error
        finally:
            current_task = asyncio.current_task()

            if self._active_task is current_task:
                self._active_task = None
                self._current_text = None
