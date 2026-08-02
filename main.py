"""Command-line entry point for the smart speaker."""

import asyncio
import traceback

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
from wake_word.factory import build_wake_word_listener
from wake_word.listener import WakeWordListener


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

    configured_chatbot_provider = (
        chatbot_provider
        or build_chatbot_provider(application_settings)
    )

    configured_speech_provider = (
        speech_provider
        or build_speech_provider(application_settings)
    )

    speech_manager = SpeechManager(
        provider=configured_speech_provider,
    )

    configured_stt_provider = (
        speech_to_text_provider
        or build_speech_to_text_provider(
            application_settings
        )
    )

    configured_microphone = (
        microphone_recorder
        or build_microphone_recorder(
            application_settings
        )
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
    *,
    manual: bool = False,
) -> str | None:
    """Record and transcribe one utterance."""

    await context.speech_manager.interrupt()

    try:
        if manual:
            recorded_audio = (
                await context.microphone_recorder
                .record_push_to_talk()
            )
        else:
            recorded_audio = (
                await context.microphone_recorder
                .record_until_silence()
            )

        print("Transcribing...")

        result = (
            await context.speech_to_text_provider
            .transcribe(
                audio=recorded_audio.samples,
                sample_rate=recorded_audio.sample_rate,
            )
        )
    except SpeechRecognitionError as error:
        print(f"Voice input failed: {error}")
        return None
    except Exception as error:
        print(
            "\n[VOICE ERROR] "
            f"{type(error).__name__}: {error}"
        )
        traceback.print_exc()
        return None

    print(f"You said: {result.text}")

    return result.text


async def process_user_command(
    *,
    assistant: Assistant,
    context: ApplicationContext,
    user_text: str,
) -> None:
    """Process and deliver one text or voice command."""

    response = await assistant.handle_text(user_text)

    await deliver_response(
        assistant_name=assistant.name,
        response_text=response.text,
        speech_manager=context.speech_manager,
    )


async def wake_word_loop(
    *,
    assistant: Assistant,
    context: ApplicationContext,
    listener: WakeWordListener,
    interaction_lock: asyncio.Lock,
) -> None:
    """Wait for wake words and process spoken commands."""

    print(
        "[WAKE] Listener started using "
        f"{listener.provider.name}"
    )

    while not listener.is_stopped:
        if listener.is_paused:
            await asyncio.sleep(0.1)
            continue

        try:
            detection = await listener.wait_for_activation()
        except asyncio.CancelledError:
            print("[WAKE] Listener task cancelled.")
            raise
        except Exception as error:
            print(
                "\n[WAKE ERROR] "
                f"{type(error).__name__}: {error}"
            )
            traceback.print_exc()
            return

        if detection is None:
            await asyncio.sleep(0.05)
            continue

        print(
            "\n[WAKE] Wake word detected "
            f"({detection.score:.2f})."
        )

        async with interaction_lock:
            await context.speech_manager.interrupt()

            await listener.pause()

            try:
                voice_text = await capture_voice_command(
                    context
                )

                if voice_text is None:
                    print(
                        "[WAKE] No command was captured. "
                        "Returning to wake-word listening."
                    )
                    continue

                await process_user_command(
                    assistant=assistant,
                    context=context,
                    user_text=voice_text,
                )
                await context.speech_manager.wait_until_idle()
            finally:
                listener.resume()


def report_wake_task_result(
    task: asyncio.Task[None],
) -> None:
    """Report unexpected termination of the wake-word task."""

    if task.cancelled():
        return

    try:
        error = task.exception()
    except asyncio.CancelledError:
        return

    if error is not None:
        print(
            "\n[WAKE TASK FAILED] "
            f"{type(error).__name__}: {error}"
        )
        traceback.print_exception(
            type(error),
            error,
            error.__traceback__,
        )
        return

    print(
        "\n[WAKE TASK STOPPED] "
        "The wake-word task ended unexpectedly."
    )


def print_ai_usage(
    chatbot_provider: ChatbotProvider,
) -> None:
    """Print OpenAI usage for this process."""

    openai_provider = find_openai_provider(
        chatbot_provider
    )

    if openai_provider is None:
        print("OpenAI is not configured.")
        return

    usage = openai_provider.total_usage

    print("OpenAI usage for this application run:")
    print(f"  Requests: {usage.request_count}")
    print(f"  Input tokens: {usage.input_tokens}")
    print(
        "  Cached input tokens: "
        f"{usage.cached_input_tokens}"
    )
    print(f"  Output tokens: {usage.output_tokens}")
    print(
        "  Estimated cost: "
        f"${openai_provider.estimated_total_cost_usd:.6f}"
    )


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


async def handle_manual_voice_input(
    *,
    assistant: Assistant,
    context: ApplicationContext,
    wake_listener: WakeWordListener | None,
    interaction_lock: asyncio.Lock,
    manual: bool,
) -> None:
    """Handle a manual or silence-terminated voice request."""

    async with interaction_lock:
        if wake_listener is not None:
            await wake_listener.pause()

        try:
            voice_text = await capture_voice_command(
                context,
                manual=manual,
            )

            if voice_text is None:
                return

            await process_user_command(
                assistant=assistant,
                context=context,
                user_text=voice_text,
            )
            await context.speech_manager.wait_until_idle()
        finally:
            if wake_listener is not None:
                wake_listener.resume()


async def run_application() -> None:
    """Run the text, voice and wake-word interface."""

    settings = load_settings()

    assistant, timer_scheduler, context = (
        build_application(settings=settings)
    )

    interaction_lock = asyncio.Lock()

    wake_listener = (
        build_wake_word_listener(settings)
        if settings.wake_word_enabled
        else None
    )

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

    wake_word_task: asyncio.Task[None] | None = None

    start_message = assistant.start_message().text

    print(start_message)
    print(
        "Chatbot provider: "
        f"{context.chatbot_provider.name}"
    )
    print(
        "Speech provider: "
        f"{context.speech_provider.name}"
    )
    print(
        "Speech recognition: "
        f"{context.speech_to_text_provider.name}"
    )
    print("Mode: command")

    microphone_description = (
        str(settings.microphone_device)
        if settings.microphone_device is not None
        else "Windows default"
    )

    print(
        "Microphone device: "
        f"{microphone_description}"
    )

    if wake_listener is not None:
        wake_microphone_description = (
            str(settings.wake_word_microphone_device)
            if settings.wake_word_microphone_device is not None
            else microphone_description
        )
        print(
            "Wake word: Hey Jarvis "
            f"(threshold {settings.wake_word_threshold}, "
            f"microphone {wake_microphone_description})"
        )
    else:
        print("Wake-word listening is disabled.")

    print("Say or type 'engage AI' for conversation.")
    print("Type /voice for automatic recording.")
    print("Type /voice-manual for manual recording.")
    print("Type /microphones to list input devices.")
    print("Type /ai-usage to view token usage.")

    await context.speech_manager.speak(start_message)
    await context.speech_manager.wait_until_idle()

    if wake_listener is not None:
        wake_word_task = asyncio.create_task(
            wake_word_loop(
                assistant=assistant,
                context=context,
                listener=wake_listener,
                interaction_lock=interaction_lock,
            ),
            name="wake-word-listener",
        )

        wake_word_task.add_done_callback(
            report_wake_task_result
        )

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
                print_ai_usage(
                    context.chatbot_provider
                )
                continue

            if normalized_text == "/microphones":
                try:
                    devices = list_input_devices()
                except SpeechRecognitionError as error:
                    print(
                        "Microphone query failed: "
                        f"{error}"
                    )
                    continue

                if not devices:
                    print(
                        "No microphone input devices "
                        "were found."
                    )
                    continue

                print("Available microphone devices:")

                for device in devices:
                    print(f"  {device}")

                continue

            if normalized_text == "/voice":
                await handle_manual_voice_input(
                    assistant=assistant,
                    context=context,
                    wake_listener=wake_listener,
                    interaction_lock=interaction_lock,
                    manual=False,
                )
                continue

            if normalized_text == "/voice-manual":
                await handle_manual_voice_input(
                    assistant=assistant,
                    context=context,
                    wake_listener=wake_listener,
                    interaction_lock=interaction_lock,
                    manual=True,
                )
                continue

            async with interaction_lock:
                await context.speech_manager.interrupt()

                if wake_listener is not None:
                    await wake_listener.pause()

                try:
                    await process_user_command(
                        assistant=assistant,
                        context=context,
                        user_text=user_text,
                    )
                    await context.speech_manager.wait_until_idle()
                finally:
                    if wake_listener is not None:
                        wake_listener.resume()
    finally:
        if wake_listener is not None:
            await wake_listener.stop()

        scheduler_task.cancel()
        event_consumer_task.cancel()

        tasks: list[asyncio.Task[object]] = [
            scheduler_task,
            event_consumer_task,
        ]

        if wake_word_task is not None:
            wake_word_task.cancel()
            tasks.append(wake_word_task)

        await context.speech_manager.close()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )


def main() -> None:
    """Start the application."""

    asyncio.run(run_application())


if __name__ == "__main__":
    main()
