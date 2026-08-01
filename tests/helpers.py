"""Shared test helpers."""

from datetime import datetime

import numpy as np

from ai.provider import ChatbotProvider
from ai.stub import StubChatbotProvider
from assistant.context import ApplicationContext
from core.clock import FakeClock
from core.event_bus import EventBus
from domains.alarm.manager import AlarmManager
from domains.audio.backend import SimulatedAudioBackend
from domains.audio.manager import AudioManager
from domains.conversation.manager import ConversationManager
from domains.interaction.manager import InteractionManager
from domains.timer.manager import TimerManager
from speech.manager import SpeechManager
from speech.provider import SpeechProvider
from speech.silent import SilentSpeechProvider
from speech_recognition.microphone import (
    MicrophoneRecorder,
    RecordedAudio,
)
from speech_recognition.provider import SpeechToTextProvider
from speech_recognition.result import TranscriptionResult


class StubSpeechToTextProvider(SpeechToTextProvider):
    """Deterministic speech recognition for tests."""

    def __init__(
        self,
        text: str = "test transcription",
    ) -> None:
        self._text = text

    @property
    def name(self) -> str:
        """Return the provider name."""

        return "stub-stt"

    async def transcribe(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
    ) -> TranscriptionResult:
        """Return the configured transcription."""

        return TranscriptionResult(
            text=self._text,
            language="en",
            language_probability=1.0,
        )


class StubMicrophoneRecorder(MicrophoneRecorder):
    """Return deterministic audio for tests."""

    def __init__(self) -> None:
        super().__init__(
            sample_rate=16000,
            input_function=lambda prompt: "",
        )

    async def record_push_to_talk(
        self,
    ) -> RecordedAudio:
        """Return one second of silence."""

        return RecordedAudio(
            samples=np.zeros(
                16000,
                dtype=np.float32,
            ),
            sample_rate=16000,
        )


def build_test_context(
    current_time: datetime | None = None,
    chatbot_provider: ChatbotProvider | None = None,
    speech_provider: SpeechProvider | None = None,
    speech_to_text_provider: SpeechToTextProvider | None = None,
    microphone_recorder: MicrophoneRecorder | None = None,
) -> ApplicationContext:
    """Create a complete application context for tests."""

    clock = FakeClock(current_time or datetime(2026, 8, 1, 20, 0))

    event_bus = EventBus()

    timer_manager = TimerManager(
        clock=clock,
        event_bus=event_bus,
    )

    alarm_manager = AlarmManager(
        clock=clock,
        event_bus=event_bus,
    )

    audio_manager = AudioManager(
        backend=SimulatedAudioBackend(),
    )

    conversation_manager = ConversationManager(
        clock=clock,
        maximum_messages=10,
    )

    interaction_manager = InteractionManager()

    configured_speech_provider = speech_provider or SilentSpeechProvider()

    speech_manager = SpeechManager(
        provider=configured_speech_provider,
    )

    return ApplicationContext(
        clock=clock,
        event_bus=event_bus,
        timer_manager=timer_manager,
        alarm_manager=alarm_manager,
        audio_manager=audio_manager,
        conversation_manager=conversation_manager,
        interaction_manager=interaction_manager,
        chatbot_provider=(chatbot_provider or StubChatbotProvider()),
        speech_provider=configured_speech_provider,
        speech_manager=speech_manager,
        speech_to_text_provider=(speech_to_text_provider or StubSpeechToTextProvider()),
        microphone_recorder=(microphone_recorder or StubMicrophoneRecorder()),
    )
