"""Multi-model token counting with accurate tokenization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import tiktoken


class ModelFamily(Enum):
    """Supported model families for token counting."""

    GPT4 = "gpt-4"
    GPT4O = "gpt-4o"
    GPT35 = "gpt-3.5-turbo"
    CLAUDE = "claude"
    LLAMA = "llama"


# Model family to tiktoken encoding mapping.
# Claude and Llama use cl100k_base as a reasonable approximation.
_ENCODING_MAP: dict[ModelFamily, str] = {
    ModelFamily.GPT4: "cl100k_base",
    ModelFamily.GPT4O: "o200k_base",
    ModelFamily.GPT35: "cl100k_base",
    ModelFamily.CLAUDE: "cl100k_base",
    ModelFamily.LLAMA: "cl100k_base",
}

# Cost per 1M tokens (input, output) in USD.
_COST_PER_MILLION: dict[ModelFamily, tuple[float, float]] = {
    ModelFamily.GPT4: (30.0, 60.0),
    ModelFamily.GPT4O: (2.50, 10.0),
    ModelFamily.GPT35: (0.50, 1.50),
    ModelFamily.CLAUDE: (15.0, 75.0),
    ModelFamily.LLAMA: (0.0, 0.0),  # Self-hosted, cost varies
}

# Context window sizes in tokens.
_CONTEXT_WINDOWS: dict[ModelFamily, int] = {
    ModelFamily.GPT4: 128_000,
    ModelFamily.GPT4O: 128_000,
    ModelFamily.GPT35: 16_385,
    ModelFamily.CLAUDE: 200_000,
    ModelFamily.LLAMA: 128_000,
}


@dataclass(frozen=True)
class TokenCount:
    """Result of token counting with cost estimation."""

    text: str
    token_count: int
    model: ModelFamily
    estimated_input_cost_usd: float
    estimated_output_cost_usd: float
    context_window: int

    @property
    def utilization(self) -> float:
        """Fraction of context window used (0.0 to 1.0)."""
        if self.context_window == 0:
            return 0.0
        return self.token_count / self.context_window

    @property
    def remaining_tokens(self) -> int:
        """Tokens remaining in the context window."""
        return max(0, self.context_window - self.token_count)


class TokenCounter:
    """Count tokens accurately for multiple model families.

    Uses tiktoken for OpenAI models and approximations for others.
    Caches encodings for performance.

    Example:
        counter = TokenCounter(ModelFamily.GPT4O)
        result = counter.count("Hello, world!")
        print(f"{result.token_count} tokens, ${result.estimated_input_cost_usd:.6f}")
    """

    def __init__(self, model: ModelFamily = ModelFamily.GPT4O) -> None:
        self.model = model
        encoding_name = _ENCODING_MAP[model]
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> TokenCount:
        """Count tokens in text and estimate costs.

        Args:
            text: The text to count tokens for.

        Returns:
            TokenCount with token count, cost estimates, and context info.
        """
        tokens = self._encoding.encode(text)
        token_count = len(tokens)

        input_rate, output_rate = _COST_PER_MILLION[self.model]
        input_cost = (token_count / 1_000_000) * input_rate
        output_cost = (token_count / 1_000_000) * output_rate

        return TokenCount(
            text=text,
            token_count=token_count,
            model=self.model,
            estimated_input_cost_usd=input_cost,
            estimated_output_cost_usd=output_cost,
            context_window=_CONTEXT_WINDOWS[self.model],
        )

    def count_many(self, texts: list[str]) -> list[TokenCount]:
        """Count tokens for multiple texts."""
        return [self.count(text) for text in texts]

    def encode(self, text: str) -> list[int]:
        """Get raw token IDs."""
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._encoding.decode(tokens)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens, respecting token boundaries.

        Unlike character-level truncation, this ensures we never split
        a token in the middle, which can cause encoding issues downstream.

        Args:
            text: Text to truncate.
            max_tokens: Maximum number of tokens to keep.

        Returns:
            Truncated text that fits within max_tokens.
        """
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self._encoding.decode(truncated_tokens)
