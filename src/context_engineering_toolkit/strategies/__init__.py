"""Named context engineering strategies following Anthropic naming conventions.

This module provides three canonical strategies for context optimization:

1. ContextCaching — Optimize context structure for prompt cache hit rate
2. Distillation — Pre-compress documents into dense, reusable distillates
3. KVCacheOrdering — Order context items to maximize KV-cache reuse

Usage:
    from context_engineering_toolkit.strategies import (
        ContextCaching,
        Distillation,
        KVCacheOrdering,
    )

    # Apply context caching strategy
    caching = ContextCaching(stable_prefix=system_prompt)
    ordered = caching(query=user_query, context=retrieved_docs)

    # Distill a long document
    distillation = Distillation(compression_ratio=0.3)
    distillate = distillation(document_text)

    # Reorder for KV-cache efficiency
    kv_ordering = KVCacheOrdering()
    optimized = kv_ordering(context_items)
"""

from src.context_engineering_toolkit.strategies.context_caching import ContextCaching
from src.context_engineering_toolkit.strategies.distillation import Distillation
from src.context_engineering_toolkit.strategies.kv_cache_ordering import KVCacheOrdering

__all__ = ["ContextCaching", "Distillation", "KVCacheOrdering"]
