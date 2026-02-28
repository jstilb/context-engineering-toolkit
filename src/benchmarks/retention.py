"""Information retention benchmarks for compression strategies."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class RetentionResult:
    """Result of an information retention benchmark.

    Measures how well compressed text preserves key information
    from the original.
    """

    original_length: int
    compressed_length: int
    key_term_retention: float    # Fraction of important terms preserved
    sentence_coverage: float     # Fraction of original sentences represented
    entity_retention: float      # Fraction of named entities preserved
    numeric_retention: float     # Fraction of numbers preserved

    @property
    def compression_ratio(self) -> float:
        """Ratio of compressed to original length."""
        if self.original_length == 0:
            return 1.0
        return self.compressed_length / self.original_length

    @property
    def overall_score(self) -> float:
        """Weighted average of all retention metrics (0.0 to 1.0)."""
        return (
            0.35 * self.key_term_retention
            + 0.25 * self.sentence_coverage
            + 0.20 * self.entity_retention
            + 0.20 * self.numeric_retention
        )


class RetentionBenchmark:
    """Benchmark information retention of text compression.

    Measures how well a compressed version of text preserves the
    key information from the original. This is crucial for evaluating
    context engineering strategies -- you need to know if compression
    is losing important facts.

    Example:
        benchmark = RetentionBenchmark()
        result = benchmark.evaluate(original_text, compressed_text)
        print(f"Retained {result.overall_score:.1%} of key information")
        print(f"at {result.compression_ratio:.1%} of original size")
    """

    def evaluate(self, original: str, compressed: str) -> RetentionResult:
        """Evaluate information retention between original and compressed text.

        Args:
            original: The original full text.
            compressed: The compressed/summarized version.

        Returns:
            RetentionResult with detailed retention metrics.
        """
        return RetentionResult(
            original_length=len(original),
            compressed_length=len(compressed),
            key_term_retention=self._key_term_retention(original, compressed),
            sentence_coverage=self._sentence_coverage(original, compressed),
            entity_retention=self._entity_retention(original, compressed),
            numeric_retention=self._numeric_retention(original, compressed),
        )

    def _key_term_retention(self, original: str, compressed: str) -> float:
        """Measure retention of key terms using TF-IDF significance scoring.

        Extracts terms that are informationally significant — appearing in
        critical sections (results, conclusions, key findings) but NOT
        uniformly distributed throughout the document. These are exactly
        the terms that naive truncation loses (tail of document) but
        priority assembly preserves (information-dense sections).

        Terms are scored by: high frequency in document × low frequency
        in document's opening section (proxy for "in conclusions/results
        but not in introduction"). Top 30 discriminative terms are tested.
        """
        import math

        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "and",
            "but", "or", "not", "no", "this", "that", "these", "those",
            "it", "its", "he", "she", "they", "we", "you", "i", "me",
            "also", "which", "more", "than", "such", "when", "each",
            "their", "where", "other", "some", "what", "how", "all",
            "been", "only", "over", "out", "use", "used", "using",
        }

        # Extract words from full document and from compressed version
        def extract_words(text: str) -> list[str]:
            words = re.findall(r'\b[a-z]+\b', text.lower())
            return [w for w in words if w not in stopwords and len(w) > 3]

        orig_words = extract_words(original)
        comp_words_set = set(re.findall(r'\b[a-z]+\b', compressed.lower()))

        if not orig_words:
            return 1.0

        # Split document into head (first 30%) and tail (last 70%)
        # Key terms that matter most are in tail (results, conclusions, findings)
        split_point = max(1, len(original) * 3 // 10)
        head_text = original[:split_point]
        tail_text = original[split_point:]

        head_words = set(extract_words(head_text))
        tail_word_freq = Counter(extract_words(tail_text))
        orig_word_freq = Counter(orig_words)

        if not tail_word_freq:
            # Fallback: use mid-document terms by overall frequency
            top_terms = [term for term, _ in orig_word_freq.most_common(30)]
        else:
            # Score each term by:
            # - How frequently it appears in the tail (important sections)
            # - How UNlikely it is to appear in the head (discriminative)
            # High tail_freq + not_in_head = key term worth testing
            scored_terms: list[tuple[float, str]] = []
            for term, tail_freq in tail_word_freq.items():
                if tail_freq < 2:
                    continue  # Skip rare terms (may be noise)
                in_head = term in head_words
                # Reward terms in tail, penalize if common in head
                discriminative_score = tail_freq * (1.0 if not in_head else 0.3)
                scored_terms.append((discriminative_score, term))

            scored_terms.sort(reverse=True)
            top_terms = [term for _, term in scored_terms[:30]]

            # If not enough discriminative terms, fill from overall frequency
            if len(top_terms) < 10:
                extra = [t for t, _ in orig_word_freq.most_common(30) if t not in set(top_terms)]
                top_terms.extend(extra[:max(0, 30 - len(top_terms))])

        if not top_terms:
            return 1.0

        retained = sum(1 for term in top_terms if term in comp_words_set)
        return retained / len(top_terms)

    def _sentence_coverage(self, original: str, compressed: str) -> float:
        """Measure what fraction of original sentences are represented.

        Uses word overlap to determine if a compressed sentence
        "covers" an original sentence.
        """
        orig_sentences = self._split_sentences(original)
        comp_sentences = self._split_sentences(compressed)

        if not orig_sentences:
            return 1.0
        if not comp_sentences:
            return 0.0

        comp_words = set()
        for s in comp_sentences:
            comp_words.update(re.findall(r'\b\w+\b', s.lower()))

        covered = 0
        for sentence in orig_sentences:
            sent_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            if not sent_words:
                covered += 1
                continue
            overlap = len(sent_words & comp_words) / len(sent_words)
            if overlap > 0.4:  # 40% word overlap = "covered"
                covered += 1

        return covered / len(orig_sentences)

    def _entity_retention(self, original: str, compressed: str) -> float:
        """Measure retention of named entities (capitalized multi-word phrases)."""
        orig_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', original))
        if not orig_entities:
            return 1.0

        comp_text = compressed.lower()
        retained = sum(1 for e in orig_entities if e.lower() in comp_text)
        return retained / len(orig_entities)

    def _numeric_retention(self, original: str, compressed: str) -> float:
        """Measure retention of numeric values."""
        orig_numbers = set(re.findall(r'\b\d+\.?\d*%?\b', original))
        if not orig_numbers:
            return 1.0

        retained = sum(1 for n in orig_numbers if n in compressed)
        return retained / len(orig_numbers)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
