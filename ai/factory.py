"""Construction of configured chatbot providers."""

from ai.fallback import FallbackChatbotProvider
from ai.ollama_provider import OllamaChatbotProvider
from ai.provider import ChatbotProvider
from ai.stub import StubChatbotProvider
from config.settings import Settings


def build_chatbot_provider(
    settings: Settings,
) -> ChatbotProvider:
    """Build the chatbot provider selected by configuration."""

    fallback_provider = StubChatbotProvider()

    if settings.chatbot_provider == "stub":
        return fallback_provider

    if settings.chatbot_provider == "ollama":
        ollama_provider = OllamaChatbotProvider(
            model=settings.ollama_model,
            host=settings.ollama_host,
            temperature=settings.ollama_temperature,
            keep_alive=settings.ollama_keep_alive,
        )

        return FallbackChatbotProvider(
            primary=ollama_provider,
            fallback=fallback_provider,
        )

    raise ValueError(f"Unsupported chatbot provider: {settings.chatbot_provider}")
