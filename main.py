"""Command-line entry point for the smart speaker."""

import asyncio

from assistant.assistant import Assistant
from assistant.context import ApplicationContext
from assistant.event_handler import format_event_message
from assistant.parser import RuleBasedCommandParser
from assistant.registry import ServiceRegistry
from core.clock import SystemClock
from core.event_bus import EventBus
from domains.timer.manager import TimerManager
from domains.timer.scheduler import TimerScheduler
from services.clock import ClockService
from services.greeting import GreetingService
from services.timer import TimerService


def build_application() -> tuple[Assistant, TimerScheduler, EventBus]:
    """Create the assistant and its background infrastructure."""

    clock = SystemClock()
    event_bus = EventBus()

    timer_manager = TimerManager(
        clock=clock,
        event_bus=event_bus,
    )

    timer_scheduler = TimerScheduler(
        timer_manager=timer_manager,
        poll_interval_seconds=0.25,
    )

    context = ApplicationContext(
        clock=clock,
        event_bus=event_bus,
        timer_manager=timer_manager,
    )

    registry = ServiceRegistry()
    registry.register(GreetingService())
    registry.register(ClockService())
    registry.register(TimerService())

    assistant = Assistant(
        registry=registry,
        parser=RuleBasedCommandParser(),
        context=context,
        name="Balanspeaker",
    )

    return assistant, timer_scheduler, event_bus


async def read_user_input(prompt: str) -> str:
    """Read terminal input without blocking the event loop."""

    return await asyncio.to_thread(input, prompt)


async def consume_events(
    *,
    event_bus: EventBus,
    assistant_name: str,
) -> None:
    """Continuously process application events."""

    while True:
        event = await event_bus.get()

        try:
            message = format_event_message(event)

            if message is not None:
                print(f"\n{assistant_name}: {message}")
        finally:
            event_bus.task_done()


async def run_application() -> None:
    """Run the text-based development interface."""

    assistant, timer_scheduler, event_bus = build_application()

    scheduler_task = asyncio.create_task(
        timer_scheduler.run(),
        name="timer-scheduler",
    )

    event_consumer_task = asyncio.create_task(
        consume_events(
            event_bus=event_bus,
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
