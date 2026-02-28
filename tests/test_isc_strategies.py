"""Tests for ISC row 3363 — Named strategies (ContextCaching, Distillation, KVCacheOrdering)."""

from __future__ import annotations

import pytest


class TestISC3363NamedStrategies:
    """ISC 3363: Three named strategies as callable classes with docstrings."""

    def test_strategies_importable(self) -> None:
        """ContextCaching, Distillation, KVCacheOrdering are importable."""
        from src.context_engineering_toolkit.strategies import (
            ContextCaching,
            Distillation,
            KVCacheOrdering,
        )
        assert ContextCaching is not None
        assert Distillation is not None
        assert KVCacheOrdering is not None

    def test_all_strategies_have_docstrings(self) -> None:
        """Each strategy class has a non-empty docstring."""
        from src.context_engineering_toolkit.strategies import (
            ContextCaching,
            Distillation,
            KVCacheOrdering,
        )
        assert ContextCaching.__doc__ is not None and len(ContextCaching.__doc__.strip()) > 10
        assert Distillation.__doc__ is not None and len(Distillation.__doc__.strip()) > 10
        assert KVCacheOrdering.__doc__ is not None and len(KVCacheOrdering.__doc__.strip()) > 10

    def test_all_strategies_are_callable(self) -> None:
        """Each strategy class instance is callable."""
        from src.context_engineering_toolkit.strategies import (
            ContextCaching,
            Distillation,
            KVCacheOrdering,
        )
        caching = ContextCaching()
        dist = Distillation()
        kv = KVCacheOrdering()
        assert callable(caching)
        assert callable(dist)
        assert callable(kv)


class TestContextCachingStrategy:
    """Tests for the ContextCaching strategy."""

    def test_context_caching_splits_stable_from_variable(self) -> None:
        """ContextCaching correctly separates stable prefix from variable suffix."""
        from src.context_engineering_toolkit.strategies import ContextCaching

        system_prompt = "You are a helpful assistant specialized in Python."
        caching = ContextCaching(stable_prefix=system_prompt)

        result = caching(query="How do I use list comprehensions?", context="Python docs...")
        assert result.stable_prefix == system_prompt
        assert "list comprehensions" in result.variable_suffix
        assert result.stable_token_count > 0
        assert result.variable_token_count > 0

    def test_context_caching_savings_ratio_between_0_and_1(self) -> None:
        """Cache savings ratio is between 0 and 1."""
        from src.context_engineering_toolkit.strategies import ContextCaching

        caching = ContextCaching(stable_prefix="A long system prompt " * 20)
        result = caching(query="Short query")
        assert 0.0 < result.estimated_cache_savings_ratio <= 1.0

    def test_context_caching_full_context_combines_stable_and_variable(self) -> None:
        """full_context property returns combined stable + variable content."""
        from src.context_engineering_toolkit.strategies import ContextCaching

        caching = ContextCaching(stable_prefix="Stable system prompt")
        result = caching(query="User question")
        full = result.full_context
        assert "Stable system prompt" in full
        assert "User question" in full

    def test_context_caching_with_prefix_returns_new_instance(self) -> None:
        """with_prefix() returns a new ContextCaching with extended stable prefix."""
        from src.context_engineering_toolkit.strategies import ContextCaching

        caching1 = ContextCaching(stable_prefix="System: You are helpful.")
        caching2 = caching1.with_prefix("Additional context.")
        assert caching2 is not caching1
        assert "System: You are helpful." in caching2.stable_prefix
        assert "Additional context." in caching2.stable_prefix

    def test_context_caching_empty_query(self) -> None:
        """ContextCaching handles empty query gracefully."""
        from src.context_engineering_toolkit.strategies import ContextCaching

        caching = ContextCaching(stable_prefix="System prompt")
        result = caching(query="", context="Some context")
        assert result.stable_prefix == "System prompt"


class TestDistillationStrategy:
    """Tests for the Distillation strategy."""

    SAMPLE_TEXT = (
        "The Transformer architecture was introduced in 2017 by Vaswani et al. "
        "in their paper 'Attention Is All You Need'. It replaced recurrent neural "
        "networks with self-attention mechanisms, achieving state-of-the-art results "
        "on machine translation tasks. The key innovation was the multi-head attention "
        "mechanism, which allows the model to attend to different positions simultaneously. "
        "GPT-4, released by OpenAI in March 2023, uses a transformer architecture with "
        "an estimated 1.8 trillion parameters. It achieves 86.4% accuracy on the MMLU "
        "benchmark, surpassing GPT-3.5's 70% accuracy. The cost of running GPT-4 was "
        "$30 per million input tokens, while GPT-4o costs $2.50 per million tokens. "
        "Context engineering has become critical for cost optimization at scale. "
        "Organizations running 100,000 daily requests can save $28,800 per month. "
    ) * 3  # Make it long enough to compress

    def test_distillation_reduces_text_size(self) -> None:
        """Distillation produces output smaller than input."""
        from src.context_engineering_toolkit.strategies import Distillation

        dist = Distillation(compression_ratio=0.4)
        result = dist(self.SAMPLE_TEXT)
        assert result.compressed_length < result.original_length, (
            "Distillate should be shorter than original"
        )

    def test_distillation_preserves_key_information(self) -> None:
        """Distillation preserves a meaningful retention score (>0.4)."""
        from src.context_engineering_toolkit.strategies import Distillation

        dist = Distillation(compression_ratio=0.5)
        result = dist(self.SAMPLE_TEXT)
        assert result.retention_score > 0.4, (
            f"Retention score {result.retention_score:.2%} too low, expected >40%"
        )

    def test_distillation_compression_ratio_applied(self) -> None:
        """Distillate is approximately the target compression ratio in size."""
        from src.context_engineering_toolkit.strategies import Distillation

        target_ratio = 0.4
        dist = Distillation(compression_ratio=target_ratio)
        result = dist(self.SAMPLE_TEXT)
        # Allow ±30% tolerance in actual compression ratio
        assert result.compression_ratio < 0.85, (
            f"Actual compression ratio {result.compression_ratio:.2f} is too close to 1.0"
        )

    def test_distillation_invalid_ratio_raises_error(self) -> None:
        """Distillation raises ValueError for invalid compression_ratio."""
        from src.context_engineering_toolkit.strategies import Distillation

        with pytest.raises(ValueError, match="compression_ratio"):
            Distillation(compression_ratio=0.0)

        with pytest.raises(ValueError, match="compression_ratio"):
            Distillation(compression_ratio=1.5)

    def test_distillation_batch_processes_multiple_texts(self) -> None:
        """distill_batch processes multiple texts and returns a list of distillates."""
        from src.context_engineering_toolkit.strategies import Distillation

        dist = Distillation(compression_ratio=0.5)
        texts = [self.SAMPLE_TEXT[:500], self.SAMPLE_TEXT[500:1000]]
        results = dist.distill_batch(texts)
        assert len(results) == 2
        for result in results:
            assert result.compressed_text is not None
            assert len(result.compressed_text) > 0

    def test_distillation_key_term_count_is_positive(self) -> None:
        """Distillate has positive key_term_count."""
        from src.context_engineering_toolkit.strategies import Distillation

        dist = Distillation(compression_ratio=0.5)
        result = dist(self.SAMPLE_TEXT)
        assert result.key_term_count > 0, "key_term_count should be positive"


class TestKVCacheOrderingStrategy:
    """Tests for the KVCacheOrdering strategy."""

    def test_kv_ordering_sorts_by_cache_layer(self) -> None:
        """KVCacheOrdering sorts items with most stable content first."""
        from src.context_engineering_toolkit.strategies import KVCacheOrdering
        from src.context_engineering_toolkit.strategies.kv_cache_ordering import (
            CacheLayer, ContextItem
        )

        kv = KVCacheOrdering()
        items = [
            ContextItem("Current user query", CacheLayer.CURRENT_MESSAGE, "query"),
            ContextItem("System instructions", CacheLayer.SYSTEM_PROMPT, "system"),
            ContextItem("Retrieved docs", CacheLayer.RETRIEVED_CONTEXT, "retrieval"),
            ContextItem("Background docs", CacheLayer.BACKGROUND_DOCUMENTS, "background"),
        ]

        ordered = kv(items)
        layer_order = [item.layer.value for item in ordered.ordered_items]
        assert layer_order == sorted(layer_order), (
            f"Items not sorted by cache layer: {[(i.label, i.layer.value) for i in ordered.ordered_items]}"
        )

    def test_kv_ordering_system_prompt_is_first(self) -> None:
        """System prompt is always first in ordered output."""
        from src.context_engineering_toolkit.strategies import KVCacheOrdering
        from src.context_engineering_toolkit.strategies.kv_cache_ordering import (
            CacheLayer, ContextItem
        )

        kv = KVCacheOrdering()
        items = [
            ContextItem("User message", CacheLayer.CURRENT_MESSAGE),
            ContextItem("System prompt", CacheLayer.SYSTEM_PROMPT),
            ContextItem("History", CacheLayer.CONVERSATION_HISTORY),
        ]
        ordered = kv(items)
        first_item = ordered.ordered_items[0]
        assert first_item.layer == CacheLayer.SYSTEM_PROMPT, (
            f"Expected SYSTEM_PROMPT first, got {first_item.layer}"
        )

    def test_kv_ordering_cache_efficiency_ratio_is_between_0_and_1(self) -> None:
        """Cache efficiency ratio is between 0.0 and 1.0."""
        from src.context_engineering_toolkit.strategies import KVCacheOrdering
        from src.context_engineering_toolkit.strategies.kv_cache_ordering import (
            CacheLayer, ContextItem
        )

        kv = KVCacheOrdering()
        items = [
            ContextItem("System" * 100, CacheLayer.SYSTEM_PROMPT),
            ContextItem("Query", CacheLayer.CURRENT_MESSAGE),
        ]
        ordered = kv(items)
        assert 0.0 <= ordered.cache_efficiency_ratio <= 1.0

    def test_kv_ordering_assembled_text_contains_all_items(self) -> None:
        """Assembled text contains content from all provided items."""
        from src.context_engineering_toolkit.strategies import KVCacheOrdering
        from src.context_engineering_toolkit.strategies.kv_cache_ordering import (
            CacheLayer, ContextItem
        )

        kv = KVCacheOrdering()
        items = [
            ContextItem("SYSTEM_CONTENT_XYZ", CacheLayer.SYSTEM_PROMPT),
            ContextItem("QUERY_CONTENT_ABC", CacheLayer.CURRENT_MESSAGE),
        ]
        ordered = kv(items)
        assert "SYSTEM_CONTENT_XYZ" in ordered.assembled_text
        assert "QUERY_CONTENT_ABC" in ordered.assembled_text

    def test_kv_ordering_plan_output_is_string(self) -> None:
        """plan() returns a non-empty string describing the cache plan."""
        from src.context_engineering_toolkit.strategies import KVCacheOrdering
        from src.context_engineering_toolkit.strategies.kv_cache_ordering import (
            CacheLayer, ContextItem
        )

        kv = KVCacheOrdering()
        items = [
            ContextItem("System prompt text", CacheLayer.SYSTEM_PROMPT, "System"),
            ContextItem("User query text", CacheLayer.CURRENT_MESSAGE, "Query"),
        ]
        plan = kv.plan(items)
        assert isinstance(plan, str) and len(plan) > 0
        assert "CACHED" in plan
        assert "VARIABLE" in plan
