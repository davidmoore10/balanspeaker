"""Registration and selection of assistant services."""

from services.base import Service


class DuplicateServiceError(ValueError):
    """Raised when two services use the same name."""


class ServiceRegistry:
    """Stores services and selects one for each user request."""

    def __init__(self) -> None:
        self._services: dict[str, Service] = {}

    def register(self, service: Service) -> None:
        """Register a service by its unique name."""

        service_name = service.name.strip().lower()

        if not service_name:
            raise ValueError("Service name cannot be empty.")

        if service_name in self._services:
            raise DuplicateServiceError(
                f"A service named '{service_name}' is already registered."
            )

        self._services[service_name] = service

    def find_handler(self, user_text: str) -> Service | None:
        """Return the first service able to handle the request."""

        for service in self._services.values():
            if service.can_handle(user_text):
                return service

        return None

    def get(self, service_name: str) -> Service | None:
        """Return a service by name."""

        normalized_name = service_name.strip().lower()
        return self._services.get(normalized_name)

    @property
    def service_names(self) -> tuple[str, ...]:
        """Return registered service names in registration order."""

        return tuple(self._services.keys())

    def __len__(self) -> int:
        """Return the number of registered services."""

        return len(self._services)
