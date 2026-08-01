"""Service for stopping active alarms."""

from assistant.context import ApplicationContext
from models.command import Command, CommandType
from models.response import AssistantResponse
from services.base import Service


class AlarmService(Service):
    """Handle alarm control commands."""

    @property
    def name(self) -> str:
        """Return the service name."""

        return "alarm"

    @property
    def supported_commands(self) -> frozenset[CommandType]:
        """Return supported command types."""

        return frozenset({CommandType.STOP_ALARM})

    async def execute(
        self,
        command: Command,
        context: ApplicationContext,
    ) -> AssistantResponse:
        """Stop currently active alarms."""

        if command.type not in self.supported_commands:
            raise ValueError(f"{self.name} cannot handle command '{command.type}'.")

        active_alarms = context.alarm_manager.get_active_alarms()

        if not active_alarms:
            return AssistantResponse(text="There is no active alarm.")

        stopped_alarms = await context.alarm_manager.stop_all()
        await context.audio_manager.stop_alarm()

        if len(stopped_alarms) == 1:
            alarm = stopped_alarms[0]

            if alarm.name.casefold() == "timer":
                return AssistantResponse(text="Alarm stopped.")

            return AssistantResponse(text=f"{alarm.name.capitalize()} alarm stopped.")

        return AssistantResponse(text=f"Stopped {len(stopped_alarms)} alarms.")
