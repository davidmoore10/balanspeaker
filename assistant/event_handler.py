"""Convert application events into user-facing messages."""

from models.event import Event, EventType


def format_event_message(event: Event) -> str | None:
    """Return a user-facing message for an application event."""

    if event.type != EventType.TIMER_FINISHED:
        return None

    timer_name = event.data.get("name")

    if not isinstance(timer_name, str):
        return "Your timer has finished."

    cleaned_name = timer_name.strip()

    if not cleaned_name or cleaned_name.lower() == "timer":
        return "Your timer has finished."

    return f"Your {cleaned_name} timer has finished."
