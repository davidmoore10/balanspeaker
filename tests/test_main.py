"""Integration tests for application construction."""

import pytest

from main import build_application


@pytest.mark.asyncio
async def test_built_application_can_create_timer() -> None:
    """The production application should register the timer service."""

    assistant, _, event_bus = build_application()

    response = await assistant.handle_text("timer 5")

    assert response.text == "Timer set for 5 seconds."
    assert event_bus.pending_count == 1


@pytest.mark.asyncio
async def test_built_application_still_handles_clock_request() -> None:
    """Adding timers should not break the clock service."""

    assistant, _, _ = build_application()

    response = await assistant.handle_text("what time is it")

    assert response.text.startswith("The current time is ")
