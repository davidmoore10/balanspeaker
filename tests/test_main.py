"""Integration tests for application construction."""

from uuid import UUID

import pytest

from ai.stub import StubChatbotProvider
from main import build_application
from speech.silent import SilentSpeechProvider

TEST_TIMER_ID = UUID("00000000-0000-0000-0000-000000000001")


def build_test_application():
    """Build the application with deterministic external providers."""

    return build_application(
        chatbot_provider=StubChatbotProvider(),
        speech_provider=SilentSpeechProvider(),
    )


@pytest.mark.asyncio
async def test_built_application_can_create_timer() -> None:
    """The production application should register the timer service."""

    assistant, _, context = build_test_application()

    response = await assistant.handle_text("timer 5")

    assert response.text == "Timer set for 5 seconds."
    assert context.event_bus.pending_count == 1


@pytest.mark.asyncio
async def test_built_application_can_create_named_timer() -> None:
    """Named timer commands should be available."""

    assistant, _, _ = build_test_application()

    response = await assistant.handle_text("set a pasta timer for 5 minutes")

    assert response.text == "Pasta timer set for 5 minutes."


@pytest.mark.asyncio
async def test_built_application_can_list_timers() -> None:
    """The production assistant should list active timers."""

    assistant, _, _ = build_test_application()

    await assistant.handle_text("timer 30")
    response = await assistant.handle_text("list timers")

    assert response.text in {
        "A timer has 30 seconds remaining.",
        "A timer has 29 seconds remaining.",
    }


@pytest.mark.asyncio
async def test_built_application_can_cancel_timer() -> None:
    """The production assistant should cancel active timers."""

    assistant, _, _ = build_test_application()

    await assistant.handle_text("timer 30")
    response = await assistant.handle_text("cancel timer")

    assert response.text == "Timer cancelled."


@pytest.mark.asyncio
async def test_built_application_can_stop_alarm() -> None:
    """The production assistant should expose alarm controls."""

    assistant, _, context = build_test_application()

    await context.alarm_manager.start_alarm(
        timer_id=TEST_TIMER_ID,
        name="Timer",
    )

    await context.audio_manager.start_alarm()

    response = await assistant.handle_text("stop alarm")

    assert response.text == "Alarm stopped."
    assert not context.alarm_manager.has_active_alarm
    assert not context.audio_manager.alarm_is_active


@pytest.mark.asyncio
async def test_built_application_still_handles_clock_request() -> None:
    """Adding other capabilities should not break the clock service."""

    assistant, _, _ = build_test_application()

    response = await assistant.handle_text("what time is it")

    assert response.text.startswith("The current time is ")
