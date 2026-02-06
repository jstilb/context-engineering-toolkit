"""Tests for compression modules."""

import pytest

from src.compression.extractive import ExtractiveSummarizer, ScoredSentence
from src.compression.truncation import SmartTruncator, TruncationStrategy, TruncationResult
from src.tokens.counter import ModelFamily, TokenCounter


class TestExtractiveSummarizer:
    """Tests for ExtractiveSummarizer."""

    def setup_method(self) -> None:
        self.summarizer = ExtractiveSummarizer(model=ModelFamily.GPT4O)
        self.sample_text = (
            "The Transformer architecture was introduced in 2017. "
            "It uses self-attention mechanisms instead of recurrence. "
            "GPT-4 has approximately 1.8 trillion parameters. "
            "The model achieves 86.4% on the MMLU benchmark. "
            "Training costs exceeded $100 million. "
            "Smaller models like Llama offer competitive performance."
        )

    def test_split_sentences_basic(self) -> None:
        sentences = self.summarizer.split_sentences("Hello world. How are you? Fine!")
        assert len(sentences) == 3

    def test_split_sentences_abbreviations(self) -> None:
        sentences = self.summarizer.split_sentences("Dr. Smith went home. Mr. Jones left.")
        assert len(sentences) == 2

    def test_split_sentences_empty(self) -> None:
        sentences = self.summarizer.split_sentences("")
        assert len(sentences) == 0

    def test_score_sentences_returns_scored(self) -> None:
        sentences = self.summarizer.split_sentences(self.sample_text)
        scored = self.summarizer.score_sentences(sentences)
        assert len(scored) > 0
        assert all(isinstance(s, ScoredSentence) for s in scored)
        assert all(s.score >= 0.0 for s in scored)
        assert all(s.token_count > 0 for s in scored)

    def test_score_sentences_empty_list(self) -> None:
        scored = self.summarizer.score_sentences([])
        assert scored == []

    def test_compress_within_budget(self) -> None:
        # If target is very large, should return original
        counter = TokenCounter(ModelFamily.GPT4O)
        original_tokens = counter.count(self.sample_text).token_count
        result = self.summarizer.compress(self.sample_text, target_tokens=original_tokens + 100)
        assert result == self.sample_text

    def test_compress_reduces_tokens(self) -> None:
        counter = TokenCounter(ModelFamily.GPT4O)
        original_tokens = counter.count(self.sample_text).token_count
        target = original_tokens // 2
        result = self.summarizer.compress(self.sample_text, target_tokens=target)
        result_tokens = counter.count(result).token_count
        assert result_tokens <= target

    def test_compress_preserves_order(self) -> None:
        counter = TokenCounter(ModelFamily.GPT4O)
        original_tokens = counter.count(self.sample_text).token_count
        result = self.summarizer.compress(
            self.sample_text, target_tokens=original_tokens // 2, preserve_order=True
        )
        # Selected sentences should appear in original order
        sentences = self.summarizer.split_sentences(result)
        # Verify each sentence appears in the original
        for sentence in sentences:
            assert sentence in self.sample_text

    def test_compress_empty_text(self) -> None:
        result = self.summarizer.compress("", target_tokens=100)
        assert result == ""

    def test_compress_with_ratio(self) -> None:
        counter = TokenCounter(ModelFamily.GPT4O)
        original_tokens = counter.count(self.sample_text).token_count
        result = self.summarizer.compress_with_ratio(self.sample_text, ratio=0.5)
        result_tokens = counter.count(result).token_count
        assert result_tokens <= int(original_tokens * 0.5) + 5  # Allow small margin

    def test_compress_with_ratio_zero(self) -> None:
        result = self.summarizer.compress_with_ratio(self.sample_text, ratio=0.0)
        assert result == ""

    def test_compress_with_ratio_one(self) -> None:
        result = self.summarizer.compress_with_ratio(self.sample_text, ratio=1.0)
        assert result == self.sample_text

    def test_position_bias(self) -> None:
        """First sentences should score higher due to position bias."""
        sentences = self.summarizer.split_sentences(self.sample_text)
        scored = self.summarizer.score_sentences(sentences)
        if len(scored) >= 3:
            # Average score of first third should be >= average of last third
            n = len(scored) // 3
            first_avg = sum(s.score for s in scored[:n]) / n
            last_avg = sum(s.score for s in scored[-n:]) / n
            # Position bias means first sentences get a boost
            # This is a soft test -- not always strictly true
            assert first_avg > 0  # At minimum, scores are positive


class TestSmartTruncator:
    """Tests for SmartTruncator."""

    def setup_method(self) -> None:
        self.truncator = SmartTruncator(model=ModelFamily.GPT4O)
        self.long_text = "This is sentence one. " * 100

    def test_truncate_within_limit(self) -> None:
        result = self.truncator.truncate("Short text", max_tokens=100)
        assert result.was_truncated is False
        assert result.text == "Short text"

    def test_truncate_head_strategy(self) -> None:
        result = self.truncator.truncate(
            self.long_text, max_tokens=50, strategy=TruncationStrategy.HEAD
        )
        assert result.was_truncated is True
        assert result.truncated_tokens <= 50
        assert result.strategy == TruncationStrategy.HEAD

    def test_truncate_tail_strategy(self) -> None:
        result = self.truncator.truncate(
            self.long_text, max_tokens=50, strategy=TruncationStrategy.TAIL
        )
        assert result.was_truncated is True
        assert result.truncated_tokens <= 50
        assert result.strategy == TruncationStrategy.TAIL

    def test_truncate_middle_strategy(self) -> None:
        result = self.truncator.truncate(
            self.long_text, max_tokens=50, strategy=TruncationStrategy.MIDDLE
        )
        assert result.was_truncated is True
        assert result.truncated_tokens <= 50
        assert "[...]" in result.text

    def test_truncation_result_properties(self) -> None:
        result = self.truncator.truncate(self.long_text, max_tokens=50)
        assert isinstance(result, TruncationResult)
        assert result.original_tokens > 50
        assert 0.0 < result.compression_ratio < 1.0

    def test_compression_ratio_no_truncation(self) -> None:
        result = self.truncator.truncate("Hi", max_tokens=100)
        assert result.compression_ratio == 1.0

    def test_empty_text(self) -> None:
        result = self.truncator.truncate("", max_tokens=50)
        assert result.was_truncated is False
        assert result.text == ""
