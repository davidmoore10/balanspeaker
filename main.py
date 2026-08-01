"""Command-line entry point for the smart speaker."""

import asyncio

from assistant.assistant import Assistant
from assistant.context import ApplicationContext
from assistant.event_handler import handle_event
from assistant.parser import RuleBasedCommandParser
from assistant.registry import ServiceRegistry
from core.clock import SystemClock
from core.event_bus import EventBus
from domains.alarm.manager import AlarmManager
from domains.audio.backend import SimulatedAudioBackend
from domains.audio.manager import AudioManager
from domains.timer.manager import TimerManager
from domains.timer.scheduler import TimerScheduler
from services.alarm import AlarmService
from services.clock import ClockService
from services.greeting import GreetingService
from services.media import MediaService
from services.timer import TimerService


def build_application() -> tuple[
    Assistant,
    TimerScheduler,
    ApplicationContext,
]:
    """Create the assistant and its background infrastructure."""

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
    )

    registry = ServiceRegistry()
    registry.register(GreetingService())
    registry.register(ClockService())
    registry.register(TimerService())
    registry.register(AlarmService())
    registry.register(MediaService())

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
                print(f"\n{assistant_name}: {message}")
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

    print(assistant.start_message().text)

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
            print(f"{assistant.name}: {response.text}")
    finally:
        scheduler_task.cancel()
        event_consumer_task.cancel()

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
