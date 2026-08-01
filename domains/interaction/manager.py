"""Management of the assistant's interaction mode."""

from domains.interaction.mode import InteractionMode


class InteractionManager:
    """Track and control the assistant's interaction mode."""

    def __init__(
        self,
        *,
        initial_mode: InteractionMode = InteractionMode.COMMAND,
    ) -> None:
        self._mode = initial_mode

    @property
    def mode(self) -> InteractionMode:
        """Return the current interaction mode."""

        return self._mode

    @property
    def ai_mode_enabled(self) -> bool:
        """Return whether conversational AI is enabled."""

        return self._mode == InteractionMode.AI

    @property
    def command_mode_enabled(self) -> bool:
        """Return whether normal command mode is enabled."""

        return self._mode == InteractionMode.COMMAND

    @property
    def sleeping(self) -> bool:
        """Return whether the assistant is sleeping."""

        return self._mode == InteractionMode.SLEEP

    def enable_ai_mode(self) -> bool:
        """Enable AI mode.

        Returns True when the mode changed.
        """

        if self._mode == InteractionMode.AI:
            return False

        self._mode = InteractionMode.AI
        return True

    def disable_ai_mode(self) -> bool:
        """Return to deterministic command mode.

        Returns True when the mode changed.
        """

        if self._mode == InteractionMode.COMMAND:
            return False

        self._mode = InteractionMode.COMMAND
        return True

    def sleep(self) -> bool:
        """Place the assistant into sleep mode."""

        if self._mode == InteractionMode.SLEEP:
            return False

        self._mode = InteractionMode.SLEEP
        return True

    def wake(self) -> bool:
        """Wake the assistant into command mode."""

        if self._mode == InteractionMode.COMMAND:
            return False

        self._mode = InteractionMode.COMMAND
        return True
