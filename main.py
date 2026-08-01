"""Command-line entry point for the smart speaker."""

import asyncio

from ai.factory import build_chatbot_provider
from ai.provider import ChatbotProvider
from assistant.assistant import Assistant
from assistant.context import ApplicationContext
from assistant.event_handler import handle_event
from assistant.parser import RuleBasedCommandParser
from assistant.registry import ServiceRegistry
from config.settings import Settings, load_settings
from core.clock import SystemClock
from core.event_bus import EventBus
from domains.alarm.manager import AlarmManager
from domains.audio.backend import SimulatedAudioBackend
from domains.audio.manager import AudioManager
from domains.conversation.manager import ConversationManager
from domains.timer.manager import TimerManager
from domains.timer.scheduler import TimerScheduler
from services.alarm import AlarmService
from services.chatbot import ChatbotService
from services.clock import ClockService
from services.greeting import GreetingService
from services.media import MediaService
from services.timer import TimerService
from speech.factory import build_speech_provider
from speech.provider import SpeechProvider


def build_application(
    *,
    chatbot_provider: ChatbotProvider | None = None,
    speech_provider: SpeechProvider | None = None,
    settings: Settings | None = None,
) -> tuple[
    Assistant,
    TimerScheduler,
    ApplicationContext,
]:
    """Create the assistant and its background infrastructure."""

    application_settings = settings or load_settings()

    clock = SystemClock()
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

    configured_chatbot_provider = chatbot_provider or build_chatbot_provider(
        application_settings
    )

    configured_speech_provider = speech_provider or build_speech_provider(
        application_settings
    )

    timer_scheduler = TimerScheduler(
        timer_manager=timer_manager,
        poll_interval_seconds=0.25,
    )

    context = ApplicationContext(
        clock=clock,
        event_bus=event_bus,
        timer_manager=timer_manager,
        alarm_manager=alarm_manager,
        audio_manager=audio_manager,
        conversation_manager=conversation_manager,
        chatbot_provider=configured_chatbot_provider,
        speech_provider=configured_speech_provider,
    )

    registry = ServiceRegistry()
    registry.register(GreetingService())
    registry.register(ClockService())
    registry.register(TimerService())
    registry.register(AlarmService())
    registry.register(MediaService())
    registry.register(ChatbotService())

    assistant = Assistant(
        registry=registry,
        parser=RuleBasedCommandParser(),
        context=context,
        name="Balanspeaker",
    )

    return assistant, timer_scheduler, context


async def read_user_input(prompt: str) -> str:
    """Read terminal input without blocking the event loop."""

    return await asyncio.to_thread(input, prompt)


async def deliver_response(
    *,
    assistant_name: str,
    response_text: str,
    speech_provider: SpeechProvider,
) -> None:
    """Print and speak an assistant response."""

    print(f"{assistant_name}: {response_text}")

    await speech_provider.speak(response_text)


async def consume_events(
    *,
    context: ApplicationContext,
    assistant_name: str,
) -> None:
    """Continuously process application events."""

    while True:
        event = await context.event_bus.get()

        try:
            message = await handle_event(
                event=event,
                context=context,
            )

            if message is not None:
                print()
                await deliver_response(
                    assistant_name=assistant_name,
                    response_text=message,
                    speech_provider=context.speech_provider,
                )
        finally:
            context.event_bus.task_done()


async def run_application() -> None:
    """Run the text-based development interface."""

    assistant, timer_scheduler, context = build_application()

    scheduler_task = asyncio.create_task(
        timer_scheduler.run(),
        name="timer-scheduler",
    )

    event_consumer_task = asyncio.create_task(
        consume_events(
            context=context,
            assistant_name=assistant.name,
        ),
        name="event-consumer",
    )

    start_message = assistant.start_message().text

    print(start_message)
    print(f"Chatbot provider: {context.chatbot_provider.name}")
    print(f"Speech provider: {context.speech_provider.name}")

    await context.speech_provider.speak(start_message)

    try:
        while True:
            try:
                user_text = await read_user_input("You: ")
            except (EOFError, KeyboardInterrupt):
                print("\nAssistant stopped.")
                break

            if user_text.strip().lower() == "exit":
                print("Assistant stopped.")
                break

            response = await assistant.handle_text(user_text)

            await deliver_response(
                assistant_name=assistant.name,
                response_text=response.text,
                speech_provider=context.speech_provider,
            )
    finally:
        scheduler_task.cancel()
        event_consumer_task.cancel()

        await context.speech_provider.stop()

        await asyncio.gather(
            scheduler_task,
            event_consumer_task,
            return_exceptions=True,
        )


def main() -> None:
    """Start the application."""

    asyncio.run(run_application())


if __name__ == "__main__":
    main()
