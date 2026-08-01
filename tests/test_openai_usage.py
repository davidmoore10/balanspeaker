"""Tests for token usage and cost estimation."""

import pytest

from ai.usage import (
    GPT_5_MINI_PRICING,
    TokenPricing,
    TokenUsage,
)


def test_usage_can_be_combined() -> None:
    """Token usage should accumulate correctly."""

    first = TokenUsage(
        input_tokens=100,
        cached_input_tokens=20,
        output_tokens=50,
        request_count=1,
    )

    second = TokenUsage(
        input_tokens=200,
        cached_input_tokens=40,
        output_tokens=80,
        request_count=1,
    )

    combined = first.add(second)

    assert combined == TokenUsage(
        input_tokens=300,
        cached_input_tokens=60,
        output_tokens=130,
        request_count=2,
    )


def test_negative_usage_is_rejected() -> None:
    """Token usage cannot contain negative values."""

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        TokenUsage(input_tokens=-1)


def test_gpt_5_mini_cost_estimate() -> None:
    """Cost should use cached and uncached rates."""

    usage = TokenUsage(
        input_tokens=1_000_000,
        cached_input_tokens=200_000,
        output_tokens=100_000,
        request_count=1,
    )

    cost = GPT_5_MINI_PRICING.estimate_cost_usd(usage)

    expected_cost = 0.8 * 0.25 + 0.2 * 0.025 + 0.1 * 2.00

    assert cost == pytest.approx(expected_cost)


def test_negative_pricing_is_rejected() -> None:
    """Token prices cannot be negative."""

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        TokenPricing(
            input_per_million=-1,
            cached_input_per_million=0,
            output_per_million=0,
        )
