"""Process application events and create user-facing messages."""

from uuid import UUID

from assistant.context import ApplicationContext
from models.event import Event, EventType


async def handle_event(
    *,
    event: Event,
    context: ApplicationContext,
) -> str | None:
    """Process an application event and return an optional message."""

    if event.type != EventType.TIMER_FINISHED:
        return None

    timer_id = _extract_timer_id(event)
    timer_name = _extract_timer_name(event)

    if timer_id is not None:
        await context.alarm_manager.start_alarm(
            timer_id=timer_id,
            name=timer_name,
        )

        await context.audio_manager.start_alarm()

    return _format_timer_finished_message(timer_name)


def format_event_message(event: Event) -> str | None:
    """Format an event without performing event side effects."""

    if event.type != EventType.TIMER_FINISHED:
        return None

    return _format_timer_finished_message(_extract_timer_name(event))


def _extract_timer_id(event: Event) -> UUID | None:
    """Extract a valid timer UUID from event data."""

    raw_timer_id = event.data.get("timer_id")

    if not isinstance(raw_timer_id, str):
        return None

    try:
        return UUID(raw_timer_id)
    except ValueError:
        return None


def _extract_timer_name(event: Event) -> str:
    """Extract a normalized timer name from event data."""

    raw_name = event.data.get("name")

    if not isinstance(raw_name, str):
        return "Timer"

    cleaned_name = raw_name.strip()

    if not cleaned_name:
        return "Timer"

    return cleaned_name


def _format_timer_finished_message(timer_name: str) -> str:
    """Return a user-facing timer completion message."""

    if timer_name.casefold() == "timer":
        return "Your timer has finished."

    return f"Your {timer_name} timer has finished."
