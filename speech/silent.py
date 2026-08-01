"""Silent speech provider used as a safe fallback."""

from speech.provider import SpeechProvider


class SilentSpeechProvider(SpeechProvider):
    """Accept speech requests without producing audio."""

    def __init__(self) -> None:
        self._utterances: list[str] = []

    @property
    def name(self) -> str:
        """Return the provider name."""

        return "silent"

    @property
    def utterances(self) -> tuple[str, ...]:
        """Return text submitted to the provider."""

        return tuple(self._utterances)

    async def speak(self, text: str) -> None:
        """Record a speech request without playing audio."""

        cleaned_text = text.strip()

        if not cleaned_text:
            return

        self._utterances.append(cleaned_text)

    async def stop(self) -> None:
        """Perform no action because no audio is playing."""
