"""Context Caching strategy — Anthropic-style prompt cache optimization.

Context caching is a technique to dramatically reduce costs on repeated API calls
by structuring prompts so the stable prefix (system prompt, tools, background docs)
can be served from the provider's KV-cache at 70-90% discount.

References:
    - Anthropic prompt caching: https://docs.anthropic.com/claude/docs/prompt-caching
    - OpenAI prompt caching: https://platform.openai.com/docs/guides/prompt-caching

Use when:
    - Same system prompt or document set is used across many queries
    - System prompt + tools exceeds 1024 tokens (OpenAI) or 2048 tokens (Anthropic)
    - Monthly request volume is high enough for cache savings to exceed overhead

Savings model:
    Without caching: cost = (system_tokens + context_tokens + query_tokens) × rate
    With caching:    cost = system_tokens × cache_rate + query_tokens × rate
    Break-even:      roughly 2+ requests per unique stable prefix
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CachedContext:
    """Result of applying the context caching strategy.

    Attributes:
        stable_prefix: The cacheable portion (system prompt + static docs).
        variable_suffix: The per-query variable portion.
        stable_token_count: Estimated tokens in the stable prefix.
        variable_token_count: Estimated tokens in the variable suffix.
        estimated_cache_savings_ratio: Fraction of tokens billable at cache rate.
    """

    stable_prefix: str
    variable_suffix: str
    stable_token_count: int
    variable_token_count: int
    estimated_cache_savings_ratio: float

    @property
    def full_context(self) -> str:
        """Full assembled context (stable + variable)."""
        if self.stable_prefix and self.variable_suffix:
            return self.stable_prefix + "\n\n" + self.variable_suffix
        return self.stable_prefix or self.variable_suffix

    @property
    def total_token_count(self) -> int:
        """Total tokens in full context."""
        return self.stable_token_count + self.variable_token_count


class ContextCaching:
    """Optimize context structure for maximum prompt cache hit rate.

    Separates context into a stable prefix (cached) and variable suffix (per-query),
    maximizing the fraction of tokens that can be served from the provider's cache
    at a significant discount (70-90% off standard input pricing).

    The strategy works by identifying which content is:
    - Stable (same across queries): system prompts, tool definitions, background docs
    - Variable (changes per query): user messages, retrieved chunks, timestamps

    The stable content is placed first so that the longest common prefix
    across sequential queries is maximized, enabling cache hits.

    Example:
        caching = ContextCaching(
            stable_prefix="You are a helpful assistant specialized in Python.\n\n" + python_docs
        )

        # For each user query:
        cached_ctx = caching(query="How do I use list comprehensions?")
        # stable_prefix is cached → 90% off; query tokens billed at full rate
        # Send cached_ctx.full_context to the API

    Args:
        stable_prefix: The static content to place in the cached portion.
            Should be your system prompt, tool definitions, and any background
            documents that don't change between queries.
        model: Model identifier for token counting (affects minimum cache sizes).
    """

    def __init__(self, stable_prefix: str = "", model: str = "gpt-4o") -> None:
        """Initialize context caching strategy.

        Args:
            stable_prefix: Static content to cache (system prompt, tools, docs).
            model: Model identifier for cache minimum threshold calculation.
        """
        self.stable_prefix = stable_prefix
        self.model = model

    def __call__(self, query: str, context: str = "") -> CachedContext:
        """Apply context caching strategy to a query.

        Places stable content first (cached), then retrieved context,
        then the user query last (variable). This ordering maximizes
        cache prefix length across sequential queries.

        Args:
            query: The user's query or current turn message.
            context: Retrieved context for this specific query (variable).

        Returns:
            CachedContext with stable/variable split and token estimates.
        """
        return self.apply(query=query, context=context)

    def apply(self, query: str, context: str = "") -> CachedContext:
        """Apply context caching strategy.

        Args:
            query: The user's current query.
            context: Per-query retrieved context (not cached).

        Returns:
            CachedContext with the stable/variable split and metadata.
        """
        # Stable: system prompt + any provided static context
        stable_parts = [self.stable_prefix] if self.stable_prefix else []
        stable = "\n\n".join(stable_parts)

        # Variable: retrieved context + current query
        variable_parts = []
        if context:
            variable_parts.append(context)
        if query:
            variable_parts.append(query)
        variable = "\n\n".join(variable_parts)

        # Estimate token counts (4 chars per token approximation)
        stable_tokens = max(1, len(stable) // 4)
        variable_tokens = max(1, len(variable) // 4)
        total_tokens = stable_tokens + variable_tokens

        # Savings ratio = fraction of tokens billable at cache rate
        cache_savings_ratio = stable_tokens / total_tokens if total_tokens > 0 else 0.0

        return CachedContext(
            stable_prefix=stable,
            variable_suffix=variable,
            stable_token_count=stable_tokens,
            variable_token_count=variable_tokens,
            estimated_cache_savings_ratio=cache_savings_ratio,
        )

    def with_prefix(self, additional_stable: str) -> "ContextCaching":
        """Return a new ContextCaching with additional stable content appended.

        Useful for building up a stable prefix incrementally (e.g., adding
        tool definitions, then background documents, then few-shot examples).

        Args:
            additional_stable: Content to add to the stable prefix.

        Returns:
            New ContextCaching instance with extended stable prefix.
        """
        separator = "\n\n" if self.stable_prefix else ""
        new_prefix = self.stable_prefix + separator + additional_stable
        return ContextCaching(stable_prefix=new_prefix, model=self.model)
