"""Shared application dependencies available to services."""

from dataclasses import dataclass

from ai.provider import ChatbotProvider
from core.clock import Clock
from core.event_bus import EventBus
from domains.alarm.manager import AlarmManager
from domains.audio.manager import AudioManager
from domains.conversation.manager import ConversationManager
from domains.interaction.manager import InteractionManager
from domains.timer.manager import TimerManager
from speech.manager import SpeechManager
from speech.provider import SpeechProvider
from speech_recognition.microphone import MicrophoneRecorder
from speech_recognition.provider import SpeechToTextProvider


@dataclass(frozen=True, slots=True)
class ApplicationContext:
    """Dependencies and managers shared across the application."""

    clock: Clock
    event_bus: EventBus
    timer_manager: TimerManager
    alarm_manager: AlarmManager
    audio_manager: AudioManager
    conversation_manager: ConversationManager
    interaction_manager: InteractionManager
    chatbot_provider: ChatbotProvider
    speech_provider: SpeechProvider
    speech_manager: SpeechManager
    speech_to_text_provider: SpeechToTextProvider
    microphone_recorder: MicrophoneRecorder
