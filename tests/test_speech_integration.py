"""Integration tests for application speech output."""

import pytest

from ai.stub import StubChatbotProvider
from config.settings import Settings
from main import build_application, deliver_response
from speech.manager import SpeechManager
from speech.silent import SilentSpeechProvider


@pytest.mark.asyncio
async def test_application_accepts_injected_speech_provider() -> None:
    """Application construction should support speech injection."""

    speech_provider = SilentSpeechProvider()

    _, _, context = build_application(
        chatbot_provider=StubChatbotProvider(),
        speech_provider=speech_provider,
        settings=Settings(
            chatbot_provider="stub",
            speech_provider="silent",
        ),
    )

    assert context.speech_provider is speech_provider
    assert context.speech_manager.provider is speech_provider


@pytest.mark.asyncio
async def test_deliver_response_prints_and_starts_speech(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Responses should be printed and submitted for speech."""

    speech_provider = SilentSpeechProvider()
    speech_manager = SpeechManager(provider=speech_provider)

    await deliver_response(
        assistant_name="Balanspeaker",
        response_text="The timer is set.",
        speech_manager=speech_manager,
    )

    await speech_manager.wait_until_idle()

    captured = capsys.readouterr()

    assert captured.out == ("Balanspeaker: The timer is set.\n")
    assert speech_provider.utterances == ("The timer is set.",)
