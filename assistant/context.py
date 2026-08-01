"""Shared application dependencies available to assistant services."""

from dataclasses import dataclass

from ai.provider import ChatbotProvider
from core.clock import Clock
from core.event_bus import EventBus
from domains.alarm.manager import AlarmManager
from domains.audio.manager import AudioManager
from domains.conversation.manager import ConversationManager
from domains.timer.manager import TimerManager


@dataclass(frozen=True, slots=True)
class ApplicationContext:
    """Dependencies and managers shared across the application."""

    clock: Clock
    event_bus: EventBus
    timer_manager: TimerManager
    alarm_manager: AlarmManager
    audio_manager: AudioManager
    conversation_manager: ConversationManager
    chatbot_provider: ChatbotProvider
