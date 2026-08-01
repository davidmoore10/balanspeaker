"""Command-line entry point for the smart speaker."""

import asyncio

from ai.factory import build_chatbot_provider
from ai.inspection import find_openai_provider
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
from domains.interaction.manager import InteractionManager
from domains.timer.manager import TimerManager
from domains.timer.scheduler import TimerScheduler
from services.alarm import AlarmService
from services.chatbot import ChatbotService
from services.clock import ClockService
from services.greeting import GreetingService
from services.interaction import InteractionService
from services.media import MediaService
from services.speech_control import SpeechControlService
from services.timer import TimerService
from speech.factory import build_speech_provider
from speech.manager import SpeechManager
from speech.provider import SpeechProvider
from speech_recognition.errors import SpeechRecognitionError
from speech_recognition.factory import (
    build_microphone_recorder,
    build_speech_to_text_provider,
)
from speech_recognition.microphone import (
    MicrophoneRecorder,
    list_input_devices,
)
from speech_recognition.provider import SpeechToTextProvider


def build_application(
    *,
    chatbot_provider: ChatbotProvider | None = None,
    speech_provider: SpeechProvider | None = None,
    speech_to_text_provider: SpeechToTextProvider | None = None,
    microphone_recorder: MicrophoneRecorder | None = None,
    settings: Settings | None = None,
) -> tuple[
    Assistant,
    TimerScheduler,
    ApplicationContext,
]:
    """Create the assistant and background infrastructure."""

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

    audio_manager = AudioManager(
        backend=SimulatedAudioBackend(),
    )

    conversation_manager = ConversationManager(
        clock=clock,
        maximum_messages=10,
    )

    interaction_manager = InteractionManager()

    configured_chatbot_provider = chatbot_provider or build_chatbot_provider(
        application_settings
    )

    configured_speech_provider = speech_provider or build_speech_provider(
        application_settings
    )

    speech_manager = SpeechManager(
        provider=configured_speech_provider,
    )

    configured_stt_provider = speech_to_text_provider or build_speech_to_text_provider(
        application_settings
    )

    configured_microphone = microphone_recorder or build_microphone_recorder(
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
        interaction_manager=interaction_manager,
        chatbot_provider=configured_chatbot_provider,
        speech_provider=configured_speech_provider,
        speech_manager=speech_manager,
        speech_to_text_provider=configured_stt_provider,
        microphone_recorder=configured_microphone,
    )

    registry = ServiceRegistry()
    registry.register(GreetingService())
    registry.register(ClockService())
    registry.register(TimerService())
    registry.register(AlarmService())
    registry.register(MediaService())
    registry.register(ChatbotService())
    registry.register(InteractionService())
    registry.register(SpeechControlService())

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
    speech_manager: SpeechManager,
) -> None:
    """Print and asynchronously speak a response."""

    print(f"{assistant_name}: {response_text}")

    await speech_manager.speak(response_text)


async def capture_voice_command(
    context: ApplicationContext,
) -> str | None:
    """Record and transcribe one utterance."""

    await context.speech_manager.interrupt()

    try:
        recorded_audio = await context.microphone_recorder.record_push_to_talk()

        print("Transcribing...")

        result = await context.speech_to_text_provider.transcribe(
            audio=recorded_audio.samples,
            sample_rate=recorded_audio.sample_rate,
        )
    except SpeechRecognitionError as error:
        print(f"Voice input failed: {error}")
        return None

    print(f"You said: {result.text}")

    return result.text


def print_ai_usage(
    chatbot_provider: ChatbotProvider,
) -> None:
    """Print OpenAI usage for this process."""

    openai_provider = find_openai_provider(chatbot_provider)

    if openai_provider is None:
        print("OpenAI is not configured.")
        return

    usage = openai_provider.total_usage

    print("OpenAI usage for this application run:")
    print(f"  Requests: {usage.request_count}")
    print(f"  Input tokens: {usage.input_tokens}")
    print(f"  Cached input tokens: {usage.cached_input_tokens}")
    print(f"  Output tokens: {usage.output_tokens}")
    print(f"  Estimated cost: ${openai_provider.estimated_total_cost_usd:.6f}")


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
                    speech_manager=context.speech_manager,
                )
        finally:
            context.event_bus.task_done()


async def run_application() -> None:
    """Run the text and push-to-talk interface."""

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
    print(f"Speech recognition: {context.speech_to_text_provider.name}")
    print("Mode: command")
    print("Say or type 'engage AI' for conversation.")
    print("Type /voice for push-to-talk.")
    print("Type /microphones to list input devices.")
    print("Type /ai-usage to view token usage.")

    await context.speech_manager.speak(start_message)

    try:
        while True:
            try:
                user_text = await read_user_input("You: ")
            except (EOFError, KeyboardInterrupt):
                print("\nAssistant stopped.")
                break

            normalized_text = user_text.strip().lower()

            if normalized_text == "exit":
                print("Assistant stopped.")
                break

            if normalized_text == "/ai-usage":
                print_ai_usage(context.chatbot_provider)
                continue

            if normalized_text == "/microphones":
                try:
                    devices = list_input_devices()
                except SpeechRecognitionError as error:
                    print(f"Microphone query failed: {error}")
                    continue

                if not devices:
                    print("No microphone input devices were found.")
                    continue

                print("Available microphone devices:")

                for device in devices:
                    print(f"  {device}")

                continue

            if normalized_text == "/voice":
                voice_text = await capture_voice_command(context)

                if voice_text is None:
                    continue

                user_text = voice_text
            else:
                # Any new typed request interrupts active speech.
                await context.speech_manager.interrupt()

            response = await assistant.handle_text(user_text)

            await deliver_response(
                assistant_name=assistant.name,
                response_text=response.text,
                speech_manager=context.speech_manager,
            )
    finally:
        scheduler_task.cancel()
        event_consumer_task.cancel()

        await context.speech_manager.close()

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
