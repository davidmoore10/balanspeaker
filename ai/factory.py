"""Construction of configured chatbot providers."""

from ai.fallback import FallbackChatbotProvider
from ai.ollama_provider import OllamaChatbotProvider
from ai.openai_provider import OpenAIChatbotProvider
from ai.provider import ChatbotProvider
from ai.stub import StubChatbotProvider
from config.settings import Settings


def build_chatbot_provider(
    settings: Settings,
) -> ChatbotProvider:
    """Build the selected provider and its fallbacks."""

    stub_provider = StubChatbotProvider()

    if settings.chatbot_provider == "stub":
        return stub_provider

    ollama_provider = OllamaChatbotProvider(
        model=settings.ollama_model,
        host=settings.ollama_host,
        temperature=settings.ollama_temperature,
        keep_alive=settings.ollama_keep_alive,
    )

    local_fallback = FallbackChatbotProvider(
        primary=ollama_provider,
        fallback=stub_provider,
    )

    if settings.chatbot_provider == "ollama":
        return local_fallback

    if settings.chatbot_provider == "openai":
        openai_provider = OpenAIChatbotProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_output_tokens=(settings.openai_max_output_tokens),
            timeout_seconds=(settings.openai_timeout_seconds),
        )

        return FallbackChatbotProvider(
            primary=openai_provider,
            fallback=local_fallback,
        )

    raise ValueError(f"Unsupported chatbot provider: {settings.chatbot_provider}")
