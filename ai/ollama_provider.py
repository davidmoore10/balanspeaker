"""Local chatbot provider backed by Ollama."""

from collections.abc import Sequence
from typing import Any, Protocol

import httpx
from ollama import AsyncClient, ResponseError

from ai.errors import (
    ChatbotProviderResponseError,
    ChatbotProviderUnavailableError,
)
from ai.provider import ChatbotProvider
from domains.conversation.message import ConversationMessage
from domains.conversation.role import ConversationRole

DEFAULT_SYSTEM_PROMPT = """
You are Balanspeaker, a concise and helpful home voice assistant.

Answer the user's question directly and naturally. Remember relevant details
from earlier messages in the supplied conversation. Use metric measurements
unless the user asks otherwise.

Your response will eventually be spoken aloud, so:
- prefer short paragraphs;
- avoid markdown tables;
- avoid unnecessary headings;
- avoid long lists unless the user asks for one;
- do not mention being a language model;
- ask a brief clarifying question only when essential.

For food storage and safety questions, distinguish between quality guidance
and safety-critical guidance. State uncertainty clearly.
""".strip()


class OllamaChatClient(Protocol):
    """Subset of the Ollama client used by this provider."""

    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, str]],
        options: dict[str, Any],
        keep_alive: str,
    ) -> Any:
        """Generate one chat response."""


class OllamaChatbotProvider(ChatbotProvider):
    """Generate conversational responses using a local Ollama model."""

    def __init__(
        self,
        *,
        model: str,
        host: str = "http://localhost:11434",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.3,
        keep_alive: str = "5m",
        client: OllamaChatClient | None = None,
    ) -> None:
        cleaned_model = model.strip()
        cleaned_host = host.strip()
        cleaned_system_prompt = system_prompt.strip()

        if not cleaned_model:
            raise ValueError("Ollama model cannot be empty.")

        if not cleaned_host:
            raise ValueError("Ollama host cannot be empty.")

        if not cleaned_system_prompt:
            raise ValueError("Ollama system prompt cannot be empty.")

        if not 0 <= temperature <= 2:
            raise ValueError("Ollama temperature must be between zero and two.")

        if not keep_alive.strip():
            raise ValueError("Ollama keep-alive value cannot be empty.")

        self._model = cleaned_model
        self._host = cleaned_host
        self._system_prompt = cleaned_system_prompt
        self._temperature = temperature
        self._keep_alive = keep_alive.strip()
        self._client = client or AsyncClient(
            host=self._host,
            timeout=60.0,
        )

    @property
    def name(self) -> str:
        """Return the provider name."""

        return f"ollama:{self._model}"

    @property
    def model(self) -> str:
        """Return the configured Ollama model."""

        return self._model

    @property
    def host(self) -> str:
        """Return the configured Ollama host."""

        return self._host

    async def generate_response(
        self,
        *,
        history: tuple[ConversationMessage, ...],
    ) -> str:
        """Generate a response from local conversation history."""

        messages = self._build_messages(history)

        try:
            response = await self._client.chat(
                model=self._model,
                messages=messages,
                options={
                    "temperature": self._temperature,
                    "num_predict": 120,
                    "num_ctx": 2048,
                },
                keep_alive=self._keep_alive,
            )
        except ResponseError as error:
            raise ChatbotProviderUnavailableError(
                self._format_response_error(error)
            ) from error
        except (httpx.HTTPError, OSError) as error:
            raise ChatbotProviderUnavailableError(
                "The local Ollama service is unavailable."
            ) from error

        response_text = self._extract_response_text(response)

        if not response_text:
            raise ChatbotProviderResponseError("Ollama returned an empty response.")

        return response_text

    def _build_messages(
        self,
        history: tuple[ConversationMessage, ...],
    ) -> list[dict[str, str]]:
        """Convert domain messages into Ollama chat messages."""

        messages = [
            {
                "role": ConversationRole.SYSTEM.value,
                "content": self._system_prompt,
            }
        ]

        for message in history:
            messages.append(
                {
                    "role": message.role.value,
                    "content": message.content,
                }
            )

        return messages

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract text from an Ollama chat response."""

        message = getattr(response, "message", None)

        if message is None:
            return ""

        content = getattr(message, "content", None)

        if not isinstance(content, str):
            return ""

        return content.strip()

    @staticmethod
    def _format_response_error(error: ResponseError) -> str:
        """Return a readable Ollama error."""

        error_message = str(getattr(error, "error", "")).strip()
        status_code = getattr(error, "status_code", None)

        if status_code == 404:
            return (
                "The configured Ollama model is not installed. "
                "Pull the model before starting Balanspeaker."
            )

        if error_message:
            return f"Ollama could not generate a response: {error_message}"

        return "Ollama could not generate a response."
