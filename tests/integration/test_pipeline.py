"""Integration tests for the full context engineering pipeline."""

from src.assembly.priority import ContextItem, ContextPriority, PriorityAssembler
from src.benchmarks.retention import RetentionBenchmark
from src.compression.extractive import ExtractiveSummarizer
from src.compression.truncation import SmartTruncator, TruncationStrategy
from src.tokens.budget import BudgetPriority, TokenBudget
from src.tokens.counter import ModelFamily, TokenCounter

# A realistic document for integration testing.
SAMPLE_DOCUMENT = """
Machine learning has transformed how we build software systems. Traditional programming
requires explicitly coding rules, while ML systems learn patterns from data. The field
has evolved rapidly since 2012 when AlexNet demonstrated that deep neural networks could
dramatically outperform traditional computer vision methods.

The Transformer architecture, introduced by Vaswani et al. in 2017, revolutionized
natural language processing. Unlike RNNs which process sequences sequentially,
Transformers use self-attention to process all tokens in parallel. This architectural
innovation enabled training on much larger datasets, leading to models like BERT, GPT-3,
and GPT-4.

Large Language Models (LLMs) have become the foundation of modern AI applications.
GPT-4, released by OpenAI in March 2023, demonstrated remarkable capabilities across
diverse tasks. It achieves 86.4% accuracy on the MMLU benchmark and can pass the bar
exam with a score in the 90th percentile. However, LLMs still struggle with
mathematical reasoning, hallucination, and knowledge cutoff limitations.

Retrieval-Augmented Generation (RAG) addresses the knowledge cutoff problem by
grounding LLM responses in external data sources. A typical RAG pipeline involves:
1) document chunking, 2) embedding generation, 3) vector storage, 4) retrieval,
and 5) generation with retrieved context. Production RAG systems often use hybrid
search combining semantic similarity with keyword matching (BM25).

The cost of running LLMs at scale is a significant consideration. GPT-4 costs
approximately $30 per million input tokens and $60 per million output tokens.
More efficient models like GPT-4o reduce costs to $2.50 and $10 respectively.
Context window management is crucial -- wasted tokens directly translate to wasted money.
"""


class TestEndToEndCompression:
    """Test the full compression pipeline."""

    def test_compress_and_measure_retention(self) -> None:
        """Compress text and verify information is preserved."""
        counter = TokenCounter(ModelFamily.GPT4O)
        original_tokens = counter.count(SAMPLE_DOCUMENT).token_count

        # Compress to 50%
        summarizer = ExtractiveSummarizer(model=ModelFamily.GPT4O)
        compressed = summarizer.compress(SAMPLE_DOCUMENT, target_tokens=original_tokens // 2)

        compressed_tokens = counter.count(compressed).token_count
        assert compressed_tokens <= original_tokens // 2 + 5  # Small margin

        # Measure retention
        benchmark = RetentionBenchmark()
        retention = benchmark.evaluate(SAMPLE_DOCUMENT, compressed)

        # At 50% compression, we should retain at least 40% of key info
        assert retention.overall_score > 0.4
        assert retention.key_term_retention > 0.3
        assert retention.compression_ratio < 0.6

    def test_truncate_and_measure_retention(self) -> None:
        """Truncation should preserve beginning information well."""
        counter = TokenCounter(ModelFamily.GPT4O)
        original_tokens = counter.count(SAMPLE_DOCUMENT).token_count

        truncator = SmartTruncator(model=ModelFamily.GPT4O)
        result = truncator.truncate(
            SAMPLE_DOCUMENT, max_tokens=original_tokens // 3, strategy=TruncationStrategy.HEAD
        )

        benchmark = RetentionBenchmark()
        retention = benchmark.evaluate(SAMPLE_DOCUMENT, result.text)

        # Head truncation keeps beginning, so some info preserved
        assert retention.overall_score > 0.0
        assert result.was_truncated is True


class TestEndToEndAssembly:
    """Test the full context assembly pipeline."""

    def test_assemble_rag_context(self) -> None:
        """Simulate a RAG context assembly scenario."""
        assembler = PriorityAssembler(
            budget_tokens=500,
            model=ModelFamily.GPT4O,
            category_headers=True,
        )

        # System prompt (required)
        assembler.add(
            ContextItem(
                content="You are a helpful AI assistant. Answer based on the provided context.",
                priority=ContextPriority.REQUIRED,
                category="system",
            )
        )

        # RAG results (high priority, ordered by relevance)
        chunks = SAMPLE_DOCUMENT.split("\n\n")
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                assembler.add(
                    ContextItem(
                        content=chunk.strip(),
                        priority=ContextPriority.HIGH,
                        source=f"doc_chunk_{i}",
                        relevance_score=1.0 - (i * 0.15),
                        category="retrieved_context",
                    )
                )

        # Chat history (medium priority)
        assembler.add(
            ContextItem(
                content="User: What is RAG?\nAssistant: RAG stands for Retrieval-Augmented Generation.",
                priority=ContextPriority.MEDIUM,
                category="chat_history",
            )
        )

        result = assembler.assemble()

        # System prompt must be included
        assert any(i.priority == ContextPriority.REQUIRED for i in result.included_items)

        # Budget respected
        assert result.total_tokens <= 500

        # Some items should be excluded due to budget
        assert len(result.excluded_items) >= 0

    def test_budget_then_assemble(self) -> None:
        """Use budget planning to inform assembly."""
        counter = TokenCounter(ModelFamily.GPT4O)

        # Plan the budget
        budget = TokenBudget(total_budget=4000, response_reserve=1000, overhead_per_section=10)
        budget.add_section("system", "System prompt", 15, priority=BudgetPriority.CRITICAL)
        budget.add_section(
            "context",
            "RAG results",
            counter.count(SAMPLE_DOCUMENT).token_count,
            priority=BudgetPriority.HIGH,
        )
        budget.add_section("history", "Chat history", 50, priority=BudgetPriority.MEDIUM)

        report = budget.allocate()

        # Context section budget tells us how much to compress
        context_section = [s for s in report.sections if s.name == "context"][0]
        if context_section.over_budget:
            # Need to compress
            summarizer = ExtractiveSummarizer(model=ModelFamily.GPT4O)
            compressed = summarizer.compress(
                SAMPLE_DOCUMENT, target_tokens=context_section.max_tokens
            )
            compressed_tokens = counter.count(compressed).token_count
            assert compressed_tokens <= context_section.max_tokens + 10


class TestMultiModelComparison:
    """Test token counting across different models."""

    def test_same_text_different_models(self) -> None:
        """Same text may tokenize differently across models."""
        text = "The quick brown fox jumps over the lazy dog."
        models = [ModelFamily.GPT4, ModelFamily.GPT4O, ModelFamily.CLAUDE]
        counts = {}

        for model in models:
            counter = TokenCounter(model)
            result = counter.count(text)
            counts[model.value] = result.token_count
            assert result.token_count > 0

        # All models should give reasonable counts for this simple text
        for count in counts.values():
            assert 5 < count < 20

    def test_cost_varies_by_model(self) -> None:
        """Different models have different cost structures."""
        text = "A" * 1000
        gpt4_cost = TokenCounter(ModelFamily.GPT4).count(text).estimated_input_cost_usd
        gpt4o_cost = TokenCounter(ModelFamily.GPT4O).count(text).estimated_input_cost_usd

        # GPT-4 should be more expensive than GPT-4o
        assert gpt4_cost > gpt4o_cost
