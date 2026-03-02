"""Distillation strategy — Pre-compress documents into dense, reusable distillates.

Distillation is the practice of pre-processing lengthy source documents into
compressed "distillates" — structured summaries that preserve key facts, entities,
and numerical data while reducing token count by 60-80%. Distillates are stored
alongside originals and used in budget-constrained RAG retrieval steps.

This differs from on-the-fly compression in that:
1. Distillation happens once (offline), not per-query
2. Distillates are stored and indexed separately from originals
3. The compression targets information preservation, not generation quality
4. Multiple distillation levels can be pre-computed (50%, 30%, 15% of original)

Use when:
    - Knowledge base documents are large (>4K tokens each)
    - Same documents are retrieved repeatedly across many queries
    - Token budget is tight and you need reliable compression quality
    - You want predictable retrieval cost per document

References:
    - Anthropic Claude context length guidelines (2026)
    - "Lost in the Middle" (Liu et al. 2023) — compress to keep critical info accessible
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tokens.counter import ModelFamily


@dataclass
class Distillate:
    """The result of distilling a document.

    Attributes:
        compressed_text: The compressed/distilled version of the source.
        original_length: Character count of the original document.
        compressed_length: Character count of the distillate.
        compression_ratio: Fraction of original size (e.g., 0.3 = 30%).
        key_term_count: Number of key terms preserved in distillate.
        retention_score: Estimated information retention (0.0 to 1.0).
    """

    compressed_text: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    key_term_count: int
    retention_score: float

    @property
    def size_reduction_pct(self) -> float:
        """Percentage reduction in size (e.g., 70.0 for 70% smaller)."""
        return (1.0 - self.compression_ratio) * 100.0


class Distillation:
    """Pre-compress documents into dense, reusable distillates.

    Applies extractive compression to reduce document size while preserving
    the maximum amount of key information. The resulting distillates can be
    stored in a vector database alongside their originals and retrieved
    preferentially when token budgets are tight.

    The distillation process uses TF-IDF-inspired sentence scoring to select
    the most information-dense sentences from each document. Unlike abstractive
    summarization, distillation is deterministic and does not require an LLM
    at distillation time.

    Example:
        distillation = Distillation(compression_ratio=0.3)

        # Distill a batch of documents offline (once)
        for doc in document_library:
            distillate = distillation(doc.text)
            doc_store.save_distillate(doc.id, distillate.compressed_text)

        # At query time: retrieve distillate instead of full document
        distillate_text = doc_store.get_distillate(doc_id)
        # Use in RAG context assembly: 70% fewer tokens at same relevance

    Args:
        compression_ratio: Target fraction of original tokens to retain.
            0.3 means the distillate will be approximately 30% of the
            original size. Typical range: 0.15 to 0.5.
        model: Model identifier for token counting during compression.
        preserve_structure: If True, maintain paragraph/section structure
            in the distillate. If False, re-rank sentences purely by score.
    """

    def __init__(
        self,
        compression_ratio: float = 0.3,
        model: str = "gpt-4o",
        preserve_structure: bool = True,
    ) -> None:
        """Initialize distillation strategy.

        Args:
            compression_ratio: Target size as fraction of original (0.0 to 1.0).
            model: Model identifier for token counting.
            preserve_structure: Whether to maintain document structure in output.
        """
        if not 0.0 < compression_ratio <= 1.0:
            raise ValueError(f"compression_ratio must be between 0 and 1, got {compression_ratio}")
        self.compression_ratio = compression_ratio
        self.model = model
        self.preserve_structure = preserve_structure

    def __call__(self, text: str) -> Distillate:
        """Distill a document into a compressed, information-dense representation.

        Args:
            text: The source document to distill.

        Returns:
            Distillate containing the compressed text and quality metrics.
        """
        return self.distill(text)

    def distill(self, text: str) -> Distillate:
        """Distill a document.

        Uses extractive compression to select the most information-dense
        sentences while respecting the target compression ratio.

        Args:
            text: Source document text.

        Returns:
            Distillate with compressed text and quality metrics.
        """
        from src.benchmarks.retention import RetentionBenchmark
        from src.compression.extractive import ExtractiveSummarizer
        from src.tokens.counter import TokenCounter

        model_family = self._resolve_model_family()
        counter = TokenCounter(model_family)
        summarizer = ExtractiveSummarizer(model=model_family)

        original_tokens = counter.count(text).token_count
        target_tokens = max(1, int(original_tokens * self.compression_ratio))

        compressed = summarizer.compress(
            text, target_tokens, preserve_order=self.preserve_structure
        )

        # Measure retention quality
        bench = RetentionBenchmark()
        retention = bench.evaluate(text, compressed)

        # Count key terms preserved
        import re

        orig_terms = set(re.findall(r"\b[A-Za-z]{4,}\b", text))
        comp_terms = set(re.findall(r"\b[A-Za-z]{4,}\b", compressed))
        key_term_count = len(orig_terms & comp_terms)

        actual_ratio = len(compressed) / len(text) if len(text) > 0 else 1.0

        return Distillate(
            compressed_text=compressed,
            original_length=len(text),
            compressed_length=len(compressed),
            compression_ratio=actual_ratio,
            key_term_count=key_term_count,
            retention_score=retention.overall_score,
        )

    def distill_batch(self, texts: list[str]) -> list[Distillate]:
        """Distill multiple documents.

        Args:
            texts: List of source documents to distill.

        Returns:
            List of Distillate objects in the same order as input.
        """
        return [self.distill(text) for text in texts]

    def _resolve_model_family(self) -> ModelFamily:
        """Resolve model name to ModelFamily enum."""
        from src.tokens.counter import ModelFamily

        mapping = {
            "gpt-4o": ModelFamily.GPT4O,
            "gpt-4": ModelFamily.GPT4,
            "gpt-3.5-turbo": ModelFamily.GPT35,
            "claude-sonnet": ModelFamily.CLAUDE,
            "claude": ModelFamily.CLAUDE,
            "llama-3.3": ModelFamily.LLAMA,
        }
        return mapping.get(self.model, ModelFamily.GPT4O)
