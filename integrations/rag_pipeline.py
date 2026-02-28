"""Integration: Context Engineering Toolkit as context manager in a RAG pipeline.

This module shows how to use the context-engineering-toolkit as a context manager
within a RAG (Retrieval-Augmented Generation) pipeline built with modern-rag-pipeline.

It wraps the toolkit as a Python context manager (`with ContextEngineeringToolkit(...)`)
that applies context compression and priority assembly at the retrieval step,
reducing token usage and improving information density before LLM generation.

Cross-reference: github.com/jstilb/modern-rag-pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path so imports work when run directly
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any

# Integration with modern_rag_pipeline
# Cross-reference: github.com/jstilb/modern-rag-pipeline
try:
    from modern_rag_pipeline import Document, RetrievalResult, RAGConfig  # type: ignore[import]
    MODERN_RAG_AVAILABLE = True
except ImportError:
    MODERN_RAG_AVAILABLE = False
    # Stub types for when modern_rag_pipeline is not installed
    Document = Any  # type: ignore[misc,assignment]
    RetrievalResult = Any  # type: ignore[misc,assignment]
    RAGConfig = Any  # type: ignore[misc,assignment]

from src.context_engineering_toolkit import ContextEngineeringToolkit
from src.compression.extractive import ExtractiveSummarizer
from src.assembly.priority import PriorityAssembler, ContextItem, ContextPriority
from src.tokens.counter import ModelFamily, TokenCounter


def build_optimized_context(
    retrieved_documents: list[Any],
    system_prompt: str,
    user_query: str,
    model: str = "claude-sonnet",
    token_budget: int = 4096,
) -> str:
    """Build an optimized context string using the toolkit as a context manager.

    Demonstrates the context-engineering-toolkit used in a retrieval step
    of a RAG pipeline. Applies extractive compression to each retrieved
    document, then assembles all components (system, docs, query) within
    the token budget using priority ordering.

    Integration: github.com/jstilb/modern-rag-pipeline

    Args:
        retrieved_documents: Documents returned from the retrieval step.
            Each document should have a `.text` or `.content` attribute,
            or be a string.
        system_prompt: The system prompt to include first (highest priority).
        user_query: The user's current query (included as low priority context).
        model: Model identifier for token counting and profile selection.
        token_budget: Maximum tokens for the assembled context.

    Returns:
        Assembled context string ready to send to the LLM.

    Example:
        # Using with modern-rag-pipeline (github.com/jstilb/modern-rag-pipeline):
        #
        #   retriever = RAGPipeline(config=RAGConfig(...))
        #   results = retriever.retrieve(query=user_query, top_k=5)
        #
        #   context = build_optimized_context(
        #       retrieved_documents=results.documents,
        #       system_prompt="You are a helpful assistant.",
        #       user_query=user_query,
        #       model="claude-sonnet",
        #       token_budget=4096,
        #   )
    """
    with ContextEngineeringToolkit(model=model, budget=token_budget) as ctx:
        # Extract text from document objects (handle both string and object inputs)
        doc_texts = []
        for doc in retrieved_documents:
            if isinstance(doc, str):
                doc_texts.append(doc)
            elif hasattr(doc, "text"):
                doc_texts.append(doc.text)
            elif hasattr(doc, "content"):
                doc_texts.append(doc.content)
            elif hasattr(doc, "page_content"):
                doc_texts.append(doc.page_content)
            else:
                doc_texts.append(str(doc))

        # Compress each retrieved document individually
        compressed_docs = []
        for doc_text in doc_texts:
            compressed = ctx.compress(doc_text)
            compressed_docs.append(compressed)

        # Assemble all components with priority ordering:
        # 1. System prompt (highest priority — always included)
        # 2. Compressed retrieved documents (high priority)
        # 3. User query (medium priority — should always fit)
        context_items = [system_prompt] + compressed_docs + [user_query]
        assembled = ctx.assemble(context_items)

        return assembled


def create_rag_context_manager(
    model: str = "claude-sonnet",
    token_budget: int = 4096,
    profile_dir: str | None = None,
) -> "ContextEngineeringToolkit":
    """Factory function to create a configured toolkit context manager.

    Returns a ContextEngineeringToolkit configured for use in RAG pipelines.
    Use this as a context manager with `with` statement.

    Integration: github.com/jstilb/modern-rag-pipeline

    Args:
        model: Model identifier (matches profile YAML filenames).
        token_budget: Maximum tokens for assembled context.
        profile_dir: Directory containing model profile YAML files.

    Returns:
        ContextEngineeringToolkit configured for RAG use.

    Example:
        with ContextEngineeringToolkit(model="claude-sonnet", budget=8000) as ctx:
            compressed_doc = ctx.compress(retrieved_document_text)
            final_context = ctx.assemble([system_prompt, compressed_doc, user_query])
            response = llm.generate(final_context)
    """
    return ContextEngineeringToolkit(
        model=model,
        budget=token_budget,
        profile_dir=profile_dir,
    )


class RAGContextOptimizer:
    """High-level optimizer that integrates toolkit with modern-rag-pipeline.

    Wraps a retrieval pipeline and applies context engineering at each
    retrieval step: compresses documents, enforces token budgets, and
    orders content for optimal model performance.

    Integration with modern_rag_pipeline:
        See github.com/jstilb/modern-rag-pipeline for the retrieval
        pipeline this class is designed to wrap.

    Example:
        optimizer = RAGContextOptimizer(
            model="claude-sonnet",
            token_budget=8000,
        )

        # Use as context manager
        with ContextEngineeringToolkit(model="claude-sonnet", budget=8000) as ctx:
            for query in user_queries:
                # Simulate retrieval (in production: use modern-rag-pipeline)
                retrieved_docs = retriever.search(query, top_k=5)

                # Apply context engineering
                optimized_context = build_optimized_context(
                    retrieved_documents=retrieved_docs,
                    system_prompt=system_prompt,
                    user_query=query,
                    model="claude-sonnet",
                    token_budget=8000,
                )
    """

    def __init__(
        self,
        model: str = "claude-sonnet",
        token_budget: int = 4096,
        compression_ratio: float | None = None,
    ) -> None:
        """Initialize the RAG context optimizer.

        Args:
            model: Model identifier for token counting and compression.
            token_budget: Maximum context tokens per request.
            compression_ratio: Override compression ratio (None = use profile default).
        """
        self.model = model
        self.token_budget = token_budget
        self.compression_ratio = compression_ratio

        model_family_map = {
            "gpt-4o": ModelFamily.GPT4O,
            "gpt-4": ModelFamily.GPT4,
            "claude-sonnet": ModelFamily.CLAUDE,
            "claude": ModelFamily.CLAUDE,
            "llama-3.3": ModelFamily.LLAMA,
            "llama": ModelFamily.LLAMA,
        }
        self._model_family = model_family_map.get(model, ModelFamily.GPT4O)
        self._counter = TokenCounter(self._model_family)
        self._summarizer = ExtractiveSummarizer(model=self._model_family)

    def optimize_retrieval_context(
        self,
        documents: list[Any],
        system_prompt: str = "",
        query: str = "",
    ) -> str:
        """Optimize retrieved documents for LLM context.

        Compresses documents to fit within the token budget while
        preserving maximum information density. Uses priority assembly
        to include system prompt and query alongside compressed docs.

        Integration: github.com/jstilb/modern-rag-pipeline

        Args:
            documents: Retrieved document objects or strings.
            system_prompt: Static system instructions (highest priority).
            query: User query to include (included after documents).

        Returns:
            Assembled context string within token_budget.
        """
        with ContextEngineeringToolkit(model=self.model, budget=self.token_budget) as ctx:
            return build_optimized_context(
                retrieved_documents=documents,
                system_prompt=system_prompt,
                user_query=query,
                model=self.model,
                token_budget=self.token_budget,
            )


# Demo / smoke test when run directly
if __name__ == "__main__":
    print("Context Engineering Toolkit + RAG Pipeline Integration Demo")
    print("Cross-reference: github.com/jstilb/modern-rag-pipeline")
    print("=" * 60)

    # Simulate retrieved documents (in production: use modern-rag-pipeline)
    mock_documents = [
        "The Transformer architecture introduced in 2017 uses self-attention mechanisms "
        "to process sequences in parallel rather than sequentially. This allows for "
        "significantly faster training and better long-range dependency modeling compared "
        "to RNNs and LSTMs. The key innovation is the multi-head attention mechanism "
        "which allows the model to jointly attend to information from different subspaces.",

        "BERT (Bidirectional Encoder Representations from Transformers) is pre-trained "
        "on masked language modeling and next sentence prediction tasks. The bidirectional "
        "training gives BERT a deeper understanding of language context compared to "
        "left-to-right models like GPT. BERT achieves state-of-the-art on 11 NLP tasks "
        "including question answering with 93.2% F1 on SQuAD v1.1.",

        "RAG (Retrieval-Augmented Generation) combines parametric memory (model weights) "
        "with non-parametric memory (retrieved documents) to improve factual accuracy. "
        "Dense Passage Retrieval (DPR) is used to retrieve relevant Wikipedia passages "
        "which are then provided as context to a seq2seq generator. This approach achieves "
        "44.5% exact match on Natural Questions open-domain QA.",
    ]

    system_prompt = (
        "You are an expert in natural language processing and deep learning. "
        "Answer questions based on the provided context. Be concise and accurate."
    )
    user_query = "How does the Transformer architecture differ from BERT?"

    print("\nInput:")
    print(f"  Documents: {len(mock_documents)} retrieved docs")
    print(f"  Total raw chars: {sum(len(d) for d in mock_documents):,}")
    print(f"  System prompt: {len(system_prompt)} chars")
    print(f"  Query: {user_query}")

    # Use as context manager — the recommended pattern
    with ContextEngineeringToolkit(model="claude-sonnet", budget=1024) as ctx:
        optimized = build_optimized_context(
            retrieved_documents=mock_documents,
            system_prompt=system_prompt,
            user_query=user_query,
            model="claude-sonnet",
            token_budget=1024,
        )

    print(f"\nOptimized context: {len(optimized):,} chars")
    print(f"Preview: {optimized[:300]}...")
    print("\nIntegration demo complete.")
