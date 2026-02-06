"""Tests for the token counter module."""

import pytest
from hypothesis import given, strategies as st

from src.tokens.counter import ModelFamily, TokenCount, TokenCounter


class TestTokenCounter:
    """Tests for TokenCounter."""

    def test_count_empty_string(self) -> None:
        counter = TokenCounter(ModelFamily.GPT4O)
        result = counter.count("")
        assert result.token_count == 0
        assert result.estimated_input_cost_usd == 0.0

    def test_count_simple_text(self) -> None:
        counter = TokenCounter(ModelFamily.GPT4O)
        result = counter.count("Hello, world!")
        assert result.token_count > 0
        assert result.model == ModelFamily.GPT4O

    def test_count_returns_token_count_object(self) -> None:
        counter = TokenCounter(ModelFamily.GPT4)
        result = counter.count("Test text")
        assert isinstance(result, TokenCount)
        assert result.text == "Test text"
        assert result.context_window == 128_000

    def test_different_models_may_differ(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        gpt4 = TokenCounter(ModelFamily.GPT4).count(text)
        gpt4o = TokenCounter(ModelFamily.GPT4O).count(text)
        # Both should have non-zero counts
        assert gpt4.token_count > 0
        assert gpt4o.token_count > 0

    def test_cost_estimation_scales_with_length(self) -> None:
        counter = TokenCounter(ModelFamily.GPT4)
        short = counter.count("Hi")
        long = counter.count("Hi " * 100)
        assert long.estimated_input_cost_usd > short.estimated_input_cost_usd

    def test_utilization_calculation(self) -> None:
        counter = TokenCounter(ModelFamily.GPT35)
        result = counter.count("Test")
        # A few tokens in a 16k window should be very low utilization
        assert 0.0 < result.utilization < 0.01

    def test_remaining_tokens(self) -> None:
        counter = TokenCounter(ModelFamily.GPT4O)
        result = counter.count("Test")
        assert result.remaining_tokens == result.context_window - result.token_count

    def test_count_many(self) -> None:
        counter = TokenCounter(ModelFamily.GPT4O)
        results = counter.count_many(["Hello", "World", "Test"])
        assert len(results) == 3
        assert all(r.token_count > 0 for r in results)

    def test_encode_decode_roundtrip(self) -> None:
        counter = TokenCounter(ModelFamily.GPT4O)
        text = "Hello, world! This is a test."
        tokens = counter.encode(text)
        decoded = counter.decode(tokens)
        assert decoded == text

    def test_truncate_to_tokens_within_limit(self) -> None:
        counter = TokenCounter(ModelFamily.GPT4O)
        text = "Short text"
        result = counter.truncate_to_tokens(text, max_tokens=100)
        assert result == text

    def test_truncate_to_tokens_exceeds_limit(self) -> None:
        counter = TokenCounter(ModelFamily.GPT4O)
        text = "This is a longer text that should be truncated. " * 50
        result = counter.truncate_to_tokens(text, max_tokens=10)
        result_tokens = counter.count(result).token_count
        assert result_tokens <= 10

    def test_truncate_to_tokens_zero(self) -> None:
        counter = TokenCounter(ModelFamily.GPT4O)
        result = counter.truncate_to_tokens("Hello world", max_tokens=0)
        assert result == ""

    @given(st.text(min_size=1, max_size=500))
    def test_count_never_negative(self, text: str) -> None:
        counter = TokenCounter(ModelFamily.GPT4O)
        result = counter.count(text)
        assert result.token_count >= 0
        assert result.estimated_input_cost_usd >= 0.0

    def test_claude_model_context_window(self) -> None:
        counter = TokenCounter(ModelFamily.CLAUDE)
        result = counter.count("Test")
        assert result.context_window == 200_000

    def test_llama_zero_cost(self) -> None:
        counter = TokenCounter(ModelFamily.LLAMA)
        result = counter.count("This is self-hosted, so cost is zero.")
        assert result.estimated_input_cost_usd == 0.0
        assert result.estimated_output_cost_usd == 0.0
