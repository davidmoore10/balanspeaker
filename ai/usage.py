"""Token usage and estimated cost tracking."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token usage for one or more chatbot requests."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0

    def __post_init__(self) -> None:
        values = (
            self.input_tokens,
            self.cached_input_tokens,
            self.output_tokens,
            self.request_count,
        )

        if any(value < 0 for value in values):
            raise ValueError("Token usage values cannot be negative.")

    def add(self, other: "TokenUsage") -> "TokenUsage":
        """Return the combined token usage."""

        return TokenUsage(
            input_tokens=(self.input_tokens + other.input_tokens),
            cached_input_tokens=(self.cached_input_tokens + other.cached_input_tokens),
            output_tokens=(self.output_tokens + other.output_tokens),
            request_count=(self.request_count + other.request_count),
        )


@dataclass(frozen=True, slots=True)
class TokenPricing:
    """Prices in US dollars per million tokens."""

    input_per_million: float
    cached_input_per_million: float
    output_per_million: float

    def __post_init__(self) -> None:
        values = (
            self.input_per_million,
            self.cached_input_per_million,
            self.output_per_million,
        )

        if any(value < 0 for value in values):
            raise ValueError("Token prices cannot be negative.")

    def estimate_cost_usd(
        self,
        usage: TokenUsage,
    ) -> float:
        """Estimate the cost of token usage in US dollars."""

        uncached_input_tokens = max(
            0,
            usage.input_tokens - usage.cached_input_tokens,
        )

        input_cost = uncached_input_tokens / 1_000_000 * self.input_per_million

        cached_input_cost = (
            usage.cached_input_tokens / 1_000_000 * self.cached_input_per_million
        )

        output_cost = usage.output_tokens / 1_000_000 * self.output_per_million

        return input_cost + cached_input_cost + output_cost


GPT_5_MINI_PRICING = TokenPricing(
    input_per_million=0.25,
    cached_input_per_million=0.025,
    output_per_million=2.00,
)
