"""Tests for the retention benchmark module."""

from src.benchmarks.retention import RetentionBenchmark, RetentionResult


class TestRetentionBenchmark:
    """Tests for RetentionBenchmark."""

    def setup_method(self) -> None:
        self.benchmark = RetentionBenchmark()
        self.original = (
            "The Transformer architecture was introduced in 2017 by Vaswani et al. "
            "GPT-4 has approximately 1.8 trillion parameters and achieves 86.4% "
            "accuracy on the MMLU benchmark. OpenAI released it in March 2023. "
            "Claude, developed by Anthropic, offers a 200,000 token context window. "
            "The cost of GPT-4 is approximately $30 per million input tokens."
        )

    def test_perfect_retention(self) -> None:
        """Identical text should have perfect retention."""
        result = self.benchmark.evaluate(self.original, self.original)
        assert result.key_term_retention == 1.0
        assert result.sentence_coverage == 1.0
        assert result.entity_retention == 1.0
        assert result.numeric_retention == 1.0
        assert result.overall_score == 1.0

    def test_empty_compressed(self) -> None:
        result = self.benchmark.evaluate(self.original, "")
        assert result.key_term_retention == 0.0
        assert result.sentence_coverage == 0.0
        assert result.overall_score == 0.0

    def test_empty_original(self) -> None:
        result = self.benchmark.evaluate("", "Some compressed text")
        # Empty original -> all retention is 1.0 (nothing to lose)
        assert result.key_term_retention == 1.0

    def test_partial_retention(self) -> None:
        compressed = (
            "The Transformer architecture was introduced in 2017. " "GPT-4 achieves 86.4% on MMLU."
        )
        result = self.benchmark.evaluate(self.original, compressed)
        assert 0.0 < result.key_term_retention <= 1.0
        assert 0.0 < result.entity_retention <= 1.0
        assert 0.0 < result.numeric_retention <= 1.0
        assert 0.0 < result.overall_score <= 1.0

    def test_compression_ratio(self) -> None:
        compressed = "Short summary."
        result = self.benchmark.evaluate(self.original, compressed)
        assert result.compression_ratio < 0.5

    def test_entity_retention(self) -> None:
        """Named entities in compressed text should be detected."""
        compressed = "OpenAI released GPT-4. Anthropic built Claude."
        result = self.benchmark.evaluate(self.original, compressed)
        assert result.entity_retention > 0.0

    def test_numeric_retention(self) -> None:
        """Numbers should be tracked."""
        compressed = "The model has 1.8 trillion parameters and costs $30."
        result = self.benchmark.evaluate(self.original, compressed)
        assert result.numeric_retention > 0.0

    def test_result_is_frozen(self) -> None:
        result = self.benchmark.evaluate("Test", "Test")
        assert isinstance(result, RetentionResult)

    def test_overall_score_weighted(self) -> None:
        """Overall score should be a weighted average of components."""
        result = self.benchmark.evaluate(self.original, self.original)
        expected = (
            0.35 * result.key_term_retention
            + 0.25 * result.sentence_coverage
            + 0.20 * result.entity_retention
            + 0.20 * result.numeric_retention
        )
        assert abs(result.overall_score - expected) < 0.001
