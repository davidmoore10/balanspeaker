"""Command-line entry point for the smart speaker."""

import asyncio

from assistant.assistant import Assistant
from assistant.context import ApplicationContext
from assistant.parser import RuleBasedCommandParser
from assistant.registry import ServiceRegistry
from core.clock import SystemClock
from core.event_bus import EventBus
from services.clock import ClockService
from services.greeting import GreetingService


def build_assistant() -> Assistant:
    """Create the assistant and register its services."""

    registry = ServiceRegistry()
    registry.register(GreetingService())
    registry.register(ClockService())

    context = ApplicationContext(
        clock=SystemClock(),
        event_bus=EventBus(),
    )

    return Assistant(
        registry=registry,
        parser=RuleBasedCommandParser(),
        context=context,
        name="Balanspeaker",
    )


async def read_user_input(prompt: str) -> str:
    """Read terminal input without blocking the event loop."""

    return await asyncio.to_thread(input, prompt)


async def run_application() -> None:
    """Run the text-based development interface."""

    assistant = build_assistant()

    print(assistant.start_message().text)

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


def main() -> None:
    """Start the application."""

    asyncio.run(run_application())


if __name__ == "__main__":
    main()
