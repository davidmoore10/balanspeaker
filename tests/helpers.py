"""Shared test helpers."""

from datetime import datetime

from ai.provider import ChatbotProvider
from ai.stub import StubChatbotProvider
from assistant.context import ApplicationContext
from core.clock import FakeClock
from core.event_bus import EventBus
from domains.alarm.manager import AlarmManager
from domains.audio.backend import SimulatedAudioBackend
from domains.audio.manager import AudioManager
from domains.conversation.manager import ConversationManager
from domains.timer.manager import TimerManager
from speech.provider import SpeechProvider
from speech.silent import SilentSpeechProvider


def build_test_context(
    current_time: datetime | None = None,
    chatbot_provider: ChatbotProvider | None = None,
    speech_provider: SpeechProvider | None = None,
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

    audio_backend = SimulatedAudioBackend()

    audio_manager = AudioManager(
        backend=audio_backend,
    )

    conversation_manager = ConversationManager(
        clock=clock,
        maximum_messages=20,
    )

    return ApplicationContext(
        clock=clock,
        event_bus=event_bus,
        timer_manager=timer_manager,
        alarm_manager=alarm_manager,
        audio_manager=audio_manager,
        conversation_manager=conversation_manager,
        chatbot_provider=(chatbot_provider or StubChatbotProvider()),
        speech_provider=(speech_provider or SilentSpeechProvider()),
    )
