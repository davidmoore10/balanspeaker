"""Inspection helpers for nested chatbot providers."""

from ai.fallback import FallbackChatbotProvider
from ai.openai_provider import OpenAIChatbotProvider
from ai.provider import ChatbotProvider


def find_openai_provider(
    provider: ChatbotProvider,
) -> OpenAIChatbotProvider | None:
    """Find an OpenAI provider inside a fallback chain."""

    if isinstance(provider, OpenAIChatbotProvider):
        return provider

    if isinstance(provider, FallbackChatbotProvider):
        primary_result = find_openai_provider(provider.primary)

        if primary_result is not None:
            return primary_result

        return find_openai_provider(provider.fallback)

    return None
