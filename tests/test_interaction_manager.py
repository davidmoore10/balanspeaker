"""Tests for assistant interaction modes."""

from domains.interaction.manager import InteractionManager
from domains.interaction.mode import InteractionMode


def test_default_mode_is_command() -> None:
    """The assistant should start in safe command mode."""

    manager = InteractionManager()

    assert manager.mode == InteractionMode.COMMAND
    assert manager.command_mode_enabled
    assert not manager.ai_mode_enabled
    assert not manager.sleeping


def test_enable_ai_mode() -> None:
    """AI mode should be explicitly enabled."""

    manager = InteractionManager()

    changed = manager.enable_ai_mode()

    assert changed
    assert manager.mode == InteractionMode.AI
    assert manager.ai_mode_enabled


def test_enabling_ai_twice_is_idempotent() -> None:
    """Repeated enable requests should have no effect."""

    manager = InteractionManager()

    assert manager.enable_ai_mode()
    assert not manager.enable_ai_mode()


def test_disable_ai_mode() -> None:
    """Disabling AI should return to command mode."""

    manager = InteractionManager()
    manager.enable_ai_mode()

    changed = manager.disable_ai_mode()

    assert changed
    assert manager.mode == InteractionMode.COMMAND


def test_sleep_and_wake() -> None:
    """Sleep mode should be representable for future use."""

    manager = InteractionManager()

    assert manager.sleep()
    assert manager.sleeping

    assert manager.wake()
    assert manager.command_mode_enabled
