"""Extractive compression: select most information-dense sentences."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from src.tokens.counter import ModelFamily, TokenCounter


@dataclass(frozen=True)
class ScoredSentence:
    """A sentence with its information density score."""

    text: str
    score: float
    position: int  # Original position in document
    token_count: int


class ExtractiveSummarizer:
    """Extractive compression using TF-IDF-inspired sentence scoring.

    Selects the most information-dense sentences from a document to fit
    within a target token budget. Uses position bias (earlier sentences
    tend to be more important) and term frequency scoring.

    Example:
        summarizer = ExtractiveSummarizer(model=ModelFamily.GPT4O)
        compressed = summarizer.compress(long_text, target_tokens=500)
    """

    def __init__(self, model: ModelFamily = ModelFamily.GPT4O) -> None:
        self._counter = TokenCounter(model)

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex heuristics.

        Handles common abbreviations and decimal numbers to avoid
        false splits.

        Args:
            text: Input text to split.

        Returns:
            List of sentence strings.
        """
        # Protect common abbreviations
        protected = text
        abbreviations = [
            "Mr.",
            "Mrs.",
            "Dr.",
            "Prof.",
            "Inc.",
            "Ltd.",
            "vs.",
            "etc.",
            "e.g.",
            "i.e.",
        ]
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r"(?<=[.!?])\s+", protected)

        # Restore abbreviations
        sentences = [s.replace("<DOT>", ".").strip() for s in sentences]

        # Filter empty sentences
        return [s for s in sentences if s]

    def score_sentences(self, sentences: list[str]) -> list[ScoredSentence]:
        """Score sentences by information density.

        Scoring factors:
        1. Term frequency (words appearing in fewer sentences score higher)
        2. Sentence length (medium-length sentences preferred)
        3. Position bias (earlier sentences weighted higher)
        4. Proper noun / number bonus

        Args:
            sentences: List of sentences to score.

        Returns:
            List of ScoredSentence objects with scores.
        """
        if not sentences:
            return []

        # Build document frequency for each word
        word_doc_freq: Counter[str] = Counter()
        sentence_words: list[list[str]] = []

        for sentence in sentences:
            words = self._tokenize_words(sentence)
            sentence_words.append(words)
            unique_words = set(words)
            for word in unique_words:
                word_doc_freq[word] += 1

        num_sentences = len(sentences)
        scored: list[ScoredSentence] = []

        for i, (sentence, words) in enumerate(zip(sentences, sentence_words, strict=False)):
            if not words:
                continue

            # TF-IDF inspired score
            tf_idf_score = 0.0
            for word in words:
                df = word_doc_freq.get(word, 1)
                idf = math.log(num_sentences / df) if df > 0 else 0
                tf = words.count(word) / len(words)
                tf_idf_score += tf * idf

            tf_idf_score /= len(words)  # Normalize by sentence length

            # Length preference: penalize very short or very long sentences
            word_count = len(words)
            length_score = 1.0
            if word_count < 5:
                length_score = 0.5
            elif word_count > 50:
                length_score = 0.7

            # Position bias: first 20% of sentences get a boost
            position_ratio = i / max(1, num_sentences - 1)
            position_score = 1.0 + max(0.0, 0.3 * (1.0 - position_ratio * 5))

            # Proper noun / number bonus
            entity_score = 1.0
            if re.search(r"\b[A-Z][a-z]+\b", sentence):
                entity_score += 0.1
            if re.search(r"\b\d+\.?\d*%?\b", sentence):
                entity_score += 0.1

            final_score = tf_idf_score * length_score * position_score * entity_score

            token_count = self._counter.count(sentence).token_count

            scored.append(
                ScoredSentence(
                    text=sentence,
                    score=final_score,
                    position=i,
                    token_count=token_count,
                )
            )

        return scored

    def compress(
        self,
        text: str,
        target_tokens: int,
        preserve_order: bool = True,
    ) -> str:
        """Compress text to fit within target token count.

        Selects the highest-scoring sentences that fit within the
        token budget. Optionally preserves original sentence order
        for coherent output.

        Args:
            text: Input text to compress.
            target_tokens: Maximum number of tokens in output.
            preserve_order: If True, selected sentences maintain original order.

        Returns:
            Compressed text fitting within target_tokens.
        """
        sentences = self.split_sentences(text)
        scored = self.score_sentences(sentences)

        if not scored:
            return ""

        # Check if text already fits
        total_tokens = sum(s.token_count for s in scored)
        if total_tokens <= target_tokens:
            return text

        # Greedy selection by score, respecting token budget
        sorted_by_score = sorted(scored, key=lambda s: s.score, reverse=True)
        selected: list[ScoredSentence] = []
        used_tokens = 0

        for sentence in sorted_by_score:
            if used_tokens + sentence.token_count <= target_tokens:
                selected.append(sentence)
                used_tokens += sentence.token_count

        if not selected:
            # Even the highest-scoring sentence does not fit; truncate it
            best = sorted_by_score[0]
            truncated = self._counter.truncate_to_tokens(best.text, target_tokens)
            return truncated

        if preserve_order:
            selected.sort(key=lambda s: s.position)

        return " ".join(s.text for s in selected)

    def compress_with_ratio(self, text: str, ratio: float) -> str:
        """Compress text to a target ratio of original token count.

        Args:
            text: Input text.
            ratio: Target ratio (0.0 to 1.0). E.g., 0.3 means 30% of original.

        Returns:
            Compressed text.
        """
        if ratio <= 0.0:
            return ""
        if ratio >= 1.0:
            return text

        original_tokens = self._counter.count(text).token_count
        target_tokens = int(original_tokens * ratio)
        return self.compress(text, target_tokens)

    @staticmethod
    def _tokenize_words(text: str) -> list[str]:
        """Simple word tokenization with lowercasing and stopword removal."""
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "and",
            "but",
            "or",
            "not",
            "no",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "he",
            "she",
            "they",
            "we",
            "you",
            "i",
            "me",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
        }
        words = re.findall(r"\b[a-z]+\b", text.lower())
        return [w for w in words if w not in stopwords and len(w) > 1]
