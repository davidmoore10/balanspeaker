"""Registration and selection of assistant services."""

from models.command import CommandType
from services.base import Service


class DuplicateServiceError(ValueError):
    """Raised when two services use the same name."""


class DuplicateCommandHandlerError(ValueError):
    """Raised when two services handle the same command type."""


class ServiceRegistry:
    """Store services and map command types to handlers."""

    def __init__(self) -> None:
        self._services: dict[str, Service] = {}
        self._command_handlers: dict[CommandType, Service] = {}

    def register(self, service: Service) -> None:
        """Register a service and its supported command types."""

        service_name = service.name.strip().lower()

        if not service_name:
            raise ValueError("Service name cannot be empty.")

        if service_name in self._services:
            raise DuplicateServiceError(
                f"A service named '{service_name}' is already registered."
            )

        for command_type in service.supported_commands:
            existing_handler = self._command_handlers.get(command_type)

            if existing_handler is not None:
                raise DuplicateCommandHandlerError(
                    f"Command '{command_type}' is already handled by "
                    f"'{existing_handler.name}'."
                )

        self._services[service_name] = service

        for command_type in service.supported_commands:
            self._command_handlers[command_type] = service

    def find_handler(self, command_type: CommandType) -> Service | None:
        """Return the service registered for a command type."""

        return self._command_handlers.get(command_type)

    def get(self, service_name: str) -> Service | None:
        """Return a service by name."""

        normalized_name = service_name.strip().lower()
        return self._services.get(normalized_name)

    @property
    def service_names(self) -> tuple[str, ...]:
        """Return registered service names."""

        return tuple(self._services.keys())

    def __len__(self) -> int:
        """Return the number of registered services."""

        return len(self._services)
