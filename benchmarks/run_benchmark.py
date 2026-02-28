#!/usr/bin/env python3
"""
Benchmark runner: compare naive truncation vs extractive compression vs priority assembly
on 10 real-world documents across 3 categories (papers, news, code).

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --output results.json
    python benchmarks/run_benchmark.py --target-ratio 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path so we can import src.*
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmarks.retention import RetentionBenchmark
from src.compression.extractive import ExtractiveSummarizer
from src.compression.truncation import SmartTruncator, TruncationStrategy
from src.assembly.priority import PriorityAssembler, ContextItem, ContextPriority
from src.tokens.counter import ModelFamily, TokenCounter


DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Document categories for the 10 benchmark documents
DOCUMENT_METADATA = {
    "paper_attention_is_all_you_need.txt": {
        "category": "academic_paper",
        "title": "Attention Is All You Need",
    },
    "paper_bert_pretraining.txt": {
        "category": "academic_paper",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
    },
    "paper_rag_retrieval_augmented.txt": {
        "category": "academic_paper",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    },
    "news_ai_regulation_2026.txt": {
        "category": "news_article",
        "title": "EU AI Act Enters Full Enforcement Phase",
    },
    "news_llm_cost_trends_2026.txt": {
        "category": "news_article",
        "title": "LLM API Costs Drop 90% Over Three Years",
    },
    "news_context_window_race.txt": {
        "category": "news_article",
        "title": "The Context Window Arms Race: 4K to 1M Tokens",
    },
    "news_context_engineering_2026.txt": {
        "category": "news_article",
        "title": "Context Engineering: The Discipline That Defines Production AI in 2026",
    },
    "code_transformer_implementation.txt": {
        "category": "code",
        "title": "Transformer Implementation (PyTorch)",
    },
    "code_vector_database.txt": {
        "category": "code",
        "title": "Vector Database Abstraction Layer",
    },
    "code_rag_pipeline_impl.txt": {
        "category": "code",
        "title": "Production RAG Pipeline Implementation",
    },
}


def run_benchmark(target_ratio: float = 0.35) -> dict:
    """Run the full benchmark suite over all 10 documents.

    Args:
        target_ratio: Compression target as fraction of original tokens.
                      Default 0.475 = 47.5% of original size.

    Returns:
        Dict with per-document results and aggregate statistics.
    """
    benchmark = RetentionBenchmark()
    counter = TokenCounter(ModelFamily.GPT4O)
    extractive = ExtractiveSummarizer(model=ModelFamily.GPT4O)
    truncator = SmartTruncator(model=ModelFamily.GPT4O)

    results = []

    doc_files = sorted(DOCUMENTS_DIR.glob("*.txt"))
    if len(doc_files) != 10:
        raise RuntimeError(
            f"Expected 10 benchmark documents, found {len(doc_files)} in {DOCUMENTS_DIR}"
        )

    for doc_path in doc_files:
        filename = doc_path.name
        meta = DOCUMENT_METADATA.get(filename, {"category": "unknown", "title": filename})

        original_text = doc_path.read_text()
        original_tokens = counter.count(original_text).token_count
        target_tokens = max(1, int(original_tokens * target_ratio))

        # Method 1: Naive truncation (head)
        naive_result = truncator.truncate(original_text, target_tokens, TruncationStrategy.HEAD)
        naive_retention = benchmark.evaluate(original_text, naive_result.text)

        # Method 2: Extractive compression
        extractive_compressed = extractive.compress(original_text, target_tokens)
        extractive_retention = benchmark.evaluate(original_text, extractive_compressed)

        # Method 3: Priority assembly
        # Split text into sentences and score each as a context item
        sentences = extractive.split_sentences(original_text)
        scored_sentences = extractive.score_sentences(sentences)

        assembler = PriorityAssembler(
            budget_tokens=target_tokens,
            model=ModelFamily.GPT4O,
            separator=" ",
            category_headers=False,
        )
        for scored in scored_sentences:
            # Convert sentence score to priority level
            if scored.score > 0.05:
                priority = ContextPriority.HIGH
            elif scored.score > 0.02:
                priority = ContextPriority.MEDIUM
            else:
                priority = ContextPriority.LOW

            assembler.add(ContextItem(
                content=scored.text,
                priority=priority,
                relevance_score=min(scored.score, 1.0),
            ))

        assembly_result = assembler.assemble()
        priority_retention = benchmark.evaluate(original_text, assembly_result.assembled_text)

        doc_result = {
            "filename": filename,
            "category": meta["category"],
            "title": meta["title"],
            "original_tokens": original_tokens,
            "target_tokens": target_tokens,
            "compression_ratio": target_ratio,
            "methods": {
                "naive_truncation": {
                    "key_term_retention": round(naive_retention.key_term_retention, 4),
                    "entity_retention": round(naive_retention.entity_retention, 4),
                    "sentence_coverage": round(naive_retention.sentence_coverage, 4),
                    "overall_score": round(naive_retention.overall_score, 4),
                },
                "extractive_compression": {
                    "key_term_retention": round(extractive_retention.key_term_retention, 4),
                    "entity_retention": round(extractive_retention.entity_retention, 4),
                    "sentence_coverage": round(extractive_retention.sentence_coverage, 4),
                    "overall_score": round(extractive_retention.overall_score, 4),
                },
                "priority_assembly": {
                    "key_term_retention": round(priority_retention.key_term_retention, 4),
                    "entity_retention": round(priority_retention.entity_retention, 4),
                    "sentence_coverage": round(priority_retention.sentence_coverage, 4),
                    "overall_score": round(priority_retention.overall_score, 4),
                },
            },
            "headline_ratio": round(
                priority_retention.key_term_retention / naive_retention.key_term_retention
                if naive_retention.key_term_retention > 0 else 0.0,
                4,
            ),
        }
        results.append(doc_result)

    # Aggregate statistics
    avg_naive_ktr = sum(r["methods"]["naive_truncation"]["key_term_retention"] for r in results) / len(results)
    avg_priority_ktr = sum(r["methods"]["priority_assembly"]["key_term_retention"] for r in results) / len(results)
    aggregate_ratio = avg_priority_ktr / avg_naive_ktr if avg_naive_ktr > 0 else 0.0

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["headline_ratio"])

    category_summary = {
        cat: round(sum(ratios) / len(ratios), 4)
        for cat, ratios in categories.items()
    }

    return {
        "benchmark_version": "1.0.0",
        "model": "gpt-4o",
        "target_ratio": target_ratio,
        "document_count": len(results),
        "categories": sorted(categories.keys()),
        "headline_stat": {
            "priority_vs_naive_key_term_retention_ratio": round(aggregate_ratio, 4),
            "avg_naive_key_term_retention": round(avg_naive_ktr, 4),
            "avg_priority_key_term_retention": round(avg_priority_ktr, 4),
            "description": (
                f"Priority assembly retains {aggregate_ratio:.1f}x more key information "
                f"than naive truncation at {target_ratio:.0%} compression ratio"
            ),
        },
        "category_summary": category_summary,
        "documents": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark context compression methods on 10 real-world documents"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write results to JSON file (default: print to stdout)"
    )
    parser.add_argument(
        "--target-ratio", type=float, default=0.35,
        help="Compression target as fraction of original tokens (default: 0.35)"
    )
    args = parser.parse_args()

    print("Running benchmark on 10 real-world documents...", file=sys.stderr)
    print(f"  Target compression ratio: {args.target_ratio:.1%}", file=sys.stderr)

    results = run_benchmark(target_ratio=args.target_ratio)

    # Print headline stat
    headline = results["headline_stat"]
    ratio = headline["priority_vs_naive_key_term_retention_ratio"]
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"HEADLINE: {headline['description']}", file=sys.stderr)
    print(f"  Avg naive key-term retention: {headline['avg_naive_key_term_retention']:.1%}", file=sys.stderr)
    print(f"  Avg priority key-term retention: {headline['avg_priority_key_term_retention']:.1%}", file=sys.stderr)
    print(f"  Ratio: {ratio:.2f}x", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Per-category summary
    print("\nCategory breakdown (priority/naive ratio):", file=sys.stderr)
    for cat, cat_ratio in results["category_summary"].items():
        print(f"  {cat}: {cat_ratio:.2f}x", file=sys.stderr)

    json_output = json.dumps(results, indent=2)

    if args.output:
        Path(args.output).write_text(json_output)
        print(f"\nResults written to: {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
