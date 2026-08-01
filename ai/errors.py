"""Errors raised by chatbot providers."""


class ChatbotProviderError(RuntimeError):
    """Base exception raised by chatbot providers."""


class ChatbotProviderUnavailableError(ChatbotProviderError):
    """Raised when a chatbot provider cannot be reached or used."""


class ChatbotProviderResponseError(ChatbotProviderError):
    """Raised when a provider returns an unusable response."""


class ChatbotAuthenticationError(ChatbotProviderError):
    """Raised when provider credentials are missing or invalid."""
