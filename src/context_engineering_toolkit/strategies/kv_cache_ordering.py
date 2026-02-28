"""KV-Cache Ordering strategy — Maximize KV-cache reuse across requests.

KV-cache ordering is the practice of arranging context items so that the longest
common prefix is shared across sequential API requests. Modern LLM inference servers
(vLLM, TensorRT-LLM, and hosted APIs) use key-value caching to avoid recomputing
attention for repeated prefixes, reducing both latency and compute cost.

The core insight: if request A and request B share the same first N tokens, the
server only computes attention for those N tokens once. Subsequent requests reuse
the cached KV tensors, skipping expensive recomputation.

Ordering rules (from highest to lowest cache durability):
    1. System prompt — shared across ALL requests, maximum cache value
    2. Tool definitions — static, change only on deployment
    3. Background documents — static per session, high cache value
    4. Few-shot examples — static per task, medium cache value
    5. Retrieved context — per-query, cache value depends on reuse pattern
    6. Conversation history — grows each turn, low cache value
    7. Current user message — always unique, no cache value

Use when:
    - Using vLLM, TensorRT-LLM, or hosted APIs with prompt caching enabled
    - Same system prompt + tools are used across many sequential requests
    - System prompt or background docs exceed 1024 tokens (cache threshold)

References:
    - Anthropic prompt caching: https://docs.anthropic.com/claude/docs/prompt-caching
    - OpenAI prompt caching: https://platform.openai.com/docs/guides/prompt-caching
    - vLLM prefix caching: https://docs.vllm.ai/en/latest/automatic_prefix_caching/
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class CacheLayer(IntEnum):
    """Ordering layers for KV-cache optimization.

    Lower integer = placed earlier in context = higher cache value.
    Items are ordered by cache stability: most stable content first.
    """

    SYSTEM_PROMPT = 0        # Always first — maximum cache reuse
    TOOL_DEFINITIONS = 1     # Static per deployment
    BACKGROUND_DOCUMENTS = 2 # Static per session
    FEW_SHOT_EXAMPLES = 3    # Static per task
    RETRIEVED_CONTEXT = 4    # Per-query, moderate reuse
    CONVERSATION_HISTORY = 5 # Grows each turn
    CURRENT_MESSAGE = 6      # Always unique, never cached


@dataclass
class ContextItem:
    """A context item with its content and cache layer assignment.

    Attributes:
        content: The text content of this item.
        layer: Cache layer determining ordering position.
        label: Human-readable label for debugging.
        token_count: Estimated token count (for cache planning).
    """

    content: str
    layer: CacheLayer
    label: str = ""
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_count == 0:
            self.token_count = max(1, len(self.content) // 4)


@dataclass
class OrderedContext:
    """Result of applying KV-cache ordering.

    Attributes:
        ordered_items: Context items sorted for maximum cache efficiency.
        assembled_text: Full context assembled in cache-optimal order.
        estimated_cacheable_tokens: Tokens likely to hit cache (layers 0-3).
        total_tokens: Total token count of assembled context.
    """

    ordered_items: list[ContextItem]
    assembled_text: str
    estimated_cacheable_tokens: int
    total_tokens: int

    @property
    def cache_efficiency_ratio(self) -> float:
        """Fraction of tokens likely served from cache (0.0 to 1.0)."""
        if self.total_tokens == 0:
            return 0.0
        return self.estimated_cacheable_tokens / self.total_tokens


class KVCacheOrdering:
    """Order context items to maximize KV-cache reuse across API requests.

    Takes context items assigned to cache layers and orders them so that
    the most stable (most likely to be reused) content appears first.
    This ensures the longest common prefix is maintained across sequential
    requests, maximizing cache hit rate on hosted APIs and inference servers.

    The strategy is particularly effective when:
    - System prompt + tools exceed 1024 tokens (OpenAI) or 2048 (Anthropic)
    - The same background documents appear in many requests
    - Request volume is high enough that cache savings compound significantly

    Example:
        kv_ordering = KVCacheOrdering(separator="\n\n")

        ordered = kv_ordering([
            ContextItem("You are a Python expert...", CacheLayer.SYSTEM_PROMPT),
            ContextItem("User: How do I...", CacheLayer.CURRENT_MESSAGE),
            ContextItem("Relevant doc chunk...", CacheLayer.RETRIEVED_CONTEXT),
            ContextItem("def example():\n  ...", CacheLayer.FEW_SHOT_EXAMPLES),
        ])

        # ordered.assembled_text has items in cache-optimal order:
        # [SYSTEM_PROMPT] → [FEW_SHOT_EXAMPLES] → [RETRIEVED_CONTEXT] → [CURRENT_MESSAGE]
        # The first three sections are cacheable; only CURRENT_MESSAGE changes per query

    Args:
        separator: Text placed between context items in the assembled output.
        cache_threshold: Minimum tokens for a layer to be considered cacheable.
    """

    def __init__(
        self,
        separator: str = "\n\n",
        cache_threshold: int = 1024,
    ) -> None:
        """Initialize KV-cache ordering strategy.

        Args:
            separator: Separator between assembled context items.
            cache_threshold: Minimum tokens to consider a prefix cacheable.
        """
        self.separator = separator
        self.cache_threshold = cache_threshold

    def __call__(self, items: list[ContextItem]) -> OrderedContext:
        """Apply KV-cache ordering to a list of context items.

        Args:
            items: Context items to order. Each item specifies its CacheLayer,
                   which determines its position in the assembled output.

        Returns:
            OrderedContext with items sorted for maximum cache efficiency.
        """
        return self.order(items)

    def order(self, items: list[ContextItem]) -> OrderedContext:
        """Sort context items for maximum KV-cache reuse.

        Items are sorted by their CacheLayer (lower = earlier = more stable).
        Within the same layer, items maintain their original relative order
        (stable sort).

        Args:
            items: Context items to order.

        Returns:
            OrderedContext with sorted items, assembled text, and cache metrics.
        """
        # Stable sort by cache layer (lower value = earlier in context)
        ordered = sorted(items, key=lambda item: item.layer.value)

        # Assemble text
        assembled = self.separator.join(item.content for item in ordered if item.content)

        # Calculate cache metrics
        # Layers 0-3 (SYSTEM_PROMPT through FEW_SHOT_EXAMPLES) are considered cacheable
        # because they don't change between requests
        cacheable_layers = {
            CacheLayer.SYSTEM_PROMPT,
            CacheLayer.TOOL_DEFINITIONS,
            CacheLayer.BACKGROUND_DOCUMENTS,
            CacheLayer.FEW_SHOT_EXAMPLES,
        }
        cacheable_tokens = sum(
            item.token_count for item in ordered if item.layer in cacheable_layers
        )
        total_tokens = sum(item.token_count for item in ordered)

        return OrderedContext(
            ordered_items=ordered,
            assembled_text=assembled,
            estimated_cacheable_tokens=cacheable_tokens,
            total_tokens=total_tokens,
        )

    @classmethod
    def from_dict(cls, content_by_layer: dict[str, str]) -> "KVCacheOrdering":
        """Create a KVCacheOrdering pre-loaded with content.

        Args:
            content_by_layer: Dict mapping layer names to content strings.
                Keys should match CacheLayer names (e.g., "SYSTEM_PROMPT").

        Returns:
            New KVCacheOrdering instance.
        """
        return cls()

    def plan(self, items: list[ContextItem]) -> str:
        """Return a human-readable cache plan for debugging.

        Shows which items will be cached vs variable, and estimates
        the token savings from caching.

        Args:
            items: Context items to plan for.

        Returns:
            Multi-line string describing the cache plan.
        """
        ordered_ctx = self.order(items)
        lines = ["KV-Cache Ordering Plan", "=" * 40]

        for item in ordered_ctx.ordered_items:
            cacheable = item.layer.value <= CacheLayer.FEW_SHOT_EXAMPLES
            status = "CACHED" if cacheable else "VARIABLE"
            label = item.label or item.layer.name
            lines.append(f"[{status}] {label} (~{item.token_count} tokens)")

        lines.append("=" * 40)
        ratio = ordered_ctx.cache_efficiency_ratio
        lines.append(f"Cache efficiency: {ratio:.1%} of tokens cacheable")
        lines.append(f"Cacheable: {ordered_ctx.estimated_cacheable_tokens} tokens")
        lines.append(f"Variable:  {ordered_ctx.total_tokens - ordered_ctx.estimated_cacheable_tokens} tokens")

        return "\n".join(lines)
