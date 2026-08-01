"""Integration tests for application construction."""

import pytest

from main import build_application


@pytest.mark.asyncio
async def test_built_application_can_create_timer() -> None:
    assistant, _, event_bus = build_application()

    response = await assistant.handle_text("timer 5")

    assert response.text == "Timer set for 5 seconds."
    assert event_bus.pending_count == 1


@pytest.mark.asyncio
async def test_built_application_can_create_named_timer() -> None:
    assistant, _, _ = build_application()

    response = await assistant.handle_text("set a pasta timer for 5 minutes")

    assert response.text == "Pasta timer set for 5 minutes."


@pytest.mark.asyncio
async def test_built_application_can_list_timers() -> None:
    assistant, _, _ = build_application()

    await assistant.handle_text("timer 30")
    response = await assistant.handle_text("list timers")

    assert response.text == "A timer has 30 seconds remaining."


@pytest.mark.asyncio
async def test_built_application_can_cancel_timer() -> None:
    assistant, _, _ = build_application()

    await assistant.handle_text("timer 30")
    response = await assistant.handle_text("cancel timer")

    assert response.text == "Timer cancelled."


@pytest.mark.asyncio
async def test_built_application_still_handles_clock_request() -> None:
    assistant, _, _ = build_application()

    response = await assistant.handle_text("what time is it")

    assert response.text.startswith("The current time is ")
