"""Service for creating, listing, and cancelling countdown timers."""

from assistant.context import ApplicationContext
from domains.timer.model import Timer
from models.command import Command, CommandType
from models.response import AssistantResponse
from services.base import Service


class TimerService(Service):
    """Handle countdown timer commands."""

    @property
    def name(self) -> str:
        """Return the service name."""

        return "timer"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        """Return supported command types."""

        return frozenset(
            {
                CommandType.START_TIMER,
                CommandType.LIST_TIMERS,
                CommandType.CANCEL_TIMER,
            }
        )

    async def execute(
        self,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Execute a timer command."""

        if command.type not in self.supported_commands:
            raise ValueError(f"{self.name} cannot handle command '{command.type}'.")

        if command.type == CommandType.START_TIMER:
            return await self._start_timer(
                command=command,
                context=context,
            )

        if command.type == CommandType.LIST_TIMERS:
            return self._list_timers(context=context)

        if command.type == CommandType.CANCEL_TIMER:
            return await self._cancel_timer(
                command=command,
                context=context,
            )

        raise ValueError(f"{self.name} cannot handle command '{command.type}'.")

    async def _start_timer(
        self,
        *,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Create a countdown timer."""

        duration_seconds = command.parameters.get("duration_seconds")

        if (
            not isinstance(duration_seconds, int)
            or isinstance(duration_seconds, bool)
            or duration_seconds <= 0
        ):
            raise ValueError(
                "Timer command requires a positive integer "
                "'duration_seconds' parameter."
            )

        timer_name = command.parameters.get("name")

        if timer_name is not None and not isinstance(timer_name, str):
            raise ValueError("Timer name must be a string.")

        timer = await context.timer_manager.create_timer(
            duration_seconds=duration_seconds,
            name=timer_name,
        )

        formatted_duration = self._format_duration(duration_seconds)

        if timer.name.lower() == "timer":
            return AssistantResponse(text=f"Timer set for {formatted_duration}.")

        return AssistantResponse(
            text=f"{timer.name.capitalize()} timer set for {formatted_duration}."
        )

    def _list_timers(
        self,
        *,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Return the currently running timers."""

        timers = context.timer_manager.get_running_timers()

        if not timers:
            return AssistantResponse(text="You have no active timers.")

        if len(timers) == 1:
            timer = timers[0]
            remaining = context.timer_manager.remaining_seconds(timer.id)

            return AssistantResponse(
                text=self._format_timer_status(
                    timer=timer,
                    remaining_seconds=remaining,
                )
            )

        descriptions = [
            self._format_timer_status(
                timer=timer,
                remaining_seconds=(context.timer_manager.remaining_seconds(timer.id)),
            )
            for timer in timers
        ]

        return AssistantResponse(
            text=(f"You have {len(timers)} active timers: " + "; ".join(descriptions))
        )

    async def _cancel_timer(
        self,
        *,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Cancel one running timer."""

        timers = context.timer_manager.get_running_timers()

        if not timers:
            return AssistantResponse(text="You have no active timers.")

        requested_name = command.parameters.get("name")

        if requested_name is not None:
            if not isinstance(requested_name, str):
                raise ValueError("Timer name must be a string.")

            matching_timers = self._find_timers_by_name(
                timers=timers,
                requested_name=requested_name,
            )

            if not matching_timers:
                return AssistantResponse(
                    text=(f"I couldn't find an active timer named '{requested_name}'.")
                )

            if len(matching_timers) > 1:
                return AssistantResponse(
                    text=(f"More than one active timer is named '{requested_name}'.")
                )

            timer = matching_timers[0]

        else:
            if len(timers) > 1:
                return AssistantResponse(
                    text=(
                        "You have more than one active timer. "
                        "Please specify which timer to cancel."
                    )
                )

            timer = timers[0]

        cancelled_timer = await context.timer_manager.cancel_timer(timer.id)

        if cancelled_timer.name.lower() == "timer":
            return AssistantResponse(text="Timer cancelled.")

        return AssistantResponse(
            text=f"{cancelled_timer.name.capitalize()} timer cancelled."
        )

    @staticmethod
    def _find_timers_by_name(
        *,
        timers: tuple[Timer, ...],
        requested_name: str,
    ) -> tuple[Timer, ...]:
        """Return active timers whose names match exactly."""

        normalized_name = requested_name.strip().casefold()

        return tuple(
            timer
            for timer in timers
            if timer.name.strip().casefold() == normalized_name
        )

    @classmethod
    def _format_timer_status(
        cls,
        *,
        timer: Timer,
        remaining_seconds: int,
    ) -> str:
        """Return a human-readable timer status."""

        formatted_duration = cls._format_duration(remaining_seconds)

        if timer.name.lower() == "timer":
            return f"A timer has {formatted_duration} remaining."

        return f"The {timer.name} timer has {formatted_duration} remaining."

    @staticmethod
    def _format_duration(duration_seconds: int) -> str:
        """Return a natural-language duration."""

        if duration_seconds < 0:
            duration_seconds = 0

        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts: list[str] = []

        if hours:
            unit = "hour" if hours == 1 else "hours"
            parts.append(f"{hours} {unit}")

        if minutes:
            unit = "minute" if minutes == 1 else "minutes"
            parts.append(f"{minutes} {unit}")

        if seconds or not parts:
            unit = "second" if seconds == 1 else "seconds"
            parts.append(f"{seconds} {unit}")

        if len(parts) == 1:
            return parts[0]

        if len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"

        return f"{parts[0]}, {parts[1]} and {parts[2]}"
