"""Cloud chatbot provider backed by the OpenAI Responses API."""

from typing import Any, Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from ai.errors import (
    ChatbotAuthenticationError,
    ChatbotProviderResponseError,
    ChatbotProviderUnavailableError,
)
from ai.provider import ChatbotProvider
from ai.usage import (
    GPT_5_MINI_PRICING,
    TokenPricing,
    TokenUsage,
)
from domains.conversation.message import ConversationMessage

DEFAULT_OPENAI_INSTRUCTIONS = """
You are Balanspeaker, a concise and helpful home voice assistant.

Your answer will be spoken aloud. Answer directly and naturally, normally in
one short paragraph. Keep ordinary answers below approximately 120 words unless
the user explicitly asks for more detail.

Use the metric system exclusively unless the user explicitly requests another
unit system. Temperatures must be given in degrees Celsius. Use kilometres,
metres, kilograms, grams, litres and millilitres rather than imperial units.

Use British English spelling.

Remember relevant details from the supplied conversation. Interpret short
follow-up questions in the context of earlier messages.

Do not use markdown tables. Avoid headings, excessive bullet points, citations
or formatting that would sound unnatural when spoken.

For food storage, food safety, health, legal or financial topics, distinguish
uncertainty clearly and avoid inventing facts.
""".strip()


class OpenAIResponsesClient(Protocol):
    """Subset of the OpenAI client required by this provider."""

    responses: Any


class OpenAIChatbotProvider(ChatbotProvider):
    """Generate conversational responses using OpenAI."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5-mini",
        instructions: str = DEFAULT_OPENAI_INSTRUCTIONS,
        max_output_tokens: int = 500,
        timeout_seconds: float = 30.0,
        client: OpenAIResponsesClient | None = None,
        pricing: TokenPricing = GPT_5_MINI_PRICING,
    ) -> None:
        cleaned_api_key = api_key.strip()
        cleaned_model = model.strip()
        cleaned_instructions = instructions.strip()

        if not cleaned_api_key:
            raise ValueError("OpenAI API key cannot be empty.")

        if not cleaned_model:
            raise ValueError("OpenAI model cannot be empty.")

        if not cleaned_instructions:
            raise ValueError("OpenAI instructions cannot be empty.")

        if max_output_tokens <= 0:
            raise ValueError("OpenAI maximum output tokens must be greater than zero.")

        if timeout_seconds <= 0:
            raise ValueError("OpenAI timeout must be greater than zero.")

        self._model = cleaned_model
        self._instructions = cleaned_instructions
        self._max_output_tokens = max_output_tokens
        self._pricing = pricing

        self._client = client or AsyncOpenAI(
            api_key=cleaned_api_key,
            timeout=timeout_seconds,
            max_retries=1,
        )

        self._last_usage = TokenUsage()
        self._total_usage = TokenUsage()

    @property
    def name(self) -> str:
        """Return the provider name."""

        return f"openai:{self._model}"

    @property
    def model(self) -> str:
        """Return the configured model."""

        return self._model

    @property
    def last_usage(self) -> TokenUsage:
        """Return usage from the latest successful request."""

        return self._last_usage

    @property
    def total_usage(self) -> TokenUsage:
        """Return cumulative usage for this application run."""

        return self._total_usage

    @property
    def estimated_total_cost_usd(self) -> float:
        """Return the estimated cost for this application run."""

        return self._pricing.estimate_cost_usd(self._total_usage)

    async def generate_response(
        self,
        *,
        history: tuple[ConversationMessage, ...],
    ) -> str:
        """Generate a response using conversation history."""

        input_messages = self._build_input(history)

        try:
            response = await self._client.responses.create(
                model=self._model,
                instructions=self._instructions,
                input=input_messages,
                max_output_tokens=self._max_output_tokens,
                reasoning={
                    "effort": "low",
                },
                text={
                    "verbosity": "low",
                },
            )
        except AuthenticationError as error:
            raise ChatbotAuthenticationError(
                "The OpenAI API key is invalid or was rejected."
            ) from error
        except RateLimitError as error:
            raise ChatbotProviderUnavailableError(
                "The OpenAI API rate or usage limit was reached."
            ) from error
        except (APIConnectionError, APITimeoutError) as error:
            raise ChatbotProviderUnavailableError(
                "The OpenAI API is currently unavailable."
            ) from error
        except APIStatusError as error:
            raise ChatbotProviderUnavailableError(
                f"The OpenAI API returned an error with status {error.status_code}."
            ) from error

        usage = self._extract_usage(response)

        self._last_usage = usage
        self._total_usage = self._total_usage.add(usage)

        response_text = self._extract_output_text(response)

        if response_text:
            return response_text

        status = getattr(response, "status", None)
        incomplete_reason = self._extract_incomplete_reason(response)

        if incomplete_reason == "max_output_tokens":
            raise ChatbotProviderResponseError(
                "OpenAI used the available output-token budget "
                "before producing an answer. Increase "
                "BALANSPEAKER_OPENAI_MAX_OUTPUT_TOKENS."
            )

        if status == "incomplete":
            raise ChatbotProviderResponseError(
                "OpenAI returned an incomplete response"
                + (f": {incomplete_reason}." if incomplete_reason else ".")
            )

        raise ChatbotProviderResponseError("OpenAI returned an empty response.")

    @staticmethod
    def _build_input(
        history: tuple[ConversationMessage, ...],
    ) -> list[dict[str, str]]:
        """Convert conversation messages to Responses API input."""

        return [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in history
        ]

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        """Extract the convenience output-text property."""

        output_text = getattr(
            response,
            "output_text",
            None,
        )

        if not isinstance(output_text, str):
            return ""

        return output_text.strip()

    @staticmethod
    def _extract_incomplete_reason(
        response: Any,
    ) -> str | None:
        """Extract the reason for an incomplete response."""

        incomplete_details = getattr(
            response,
            "incomplete_details",
            None,
        )

        if incomplete_details is None:
            return None

        reason = getattr(
            incomplete_details,
            "reason",
            None,
        )

        if not isinstance(reason, str):
            return None

        cleaned_reason = reason.strip()

        return cleaned_reason or None

    @staticmethod
    def _extract_usage(response: Any) -> TokenUsage:
        """Extract token usage from an API response."""

        usage = getattr(response, "usage", None)

        if usage is None:
            return TokenUsage(request_count=1)

        input_tokens = OpenAIChatbotProvider._safe_int(
            getattr(usage, "input_tokens", 0)
        )

        output_tokens = OpenAIChatbotProvider._safe_int(
            getattr(usage, "output_tokens", 0)
        )

        input_details = getattr(
            usage,
            "input_tokens_details",
            None,
        )

        cached_tokens = 0

        if input_details is not None:
            cached_tokens = OpenAIChatbotProvider._safe_int(
                getattr(
                    input_details,
                    "cached_tokens",
                    0,
                )
            )

        return TokenUsage(
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens,
            request_count=1,
        )

    @staticmethod
    def _safe_int(value: Any) -> int:
        """Return a non-negative integer from metadata."""

        if isinstance(value, bool):
            return 0

        if not isinstance(value, int | float):
            return 0

        return max(0, int(value))
