"""CLI interface for the Context Engineering Toolkit."""

from __future__ import annotations

import json
import sys

import click

from src.compression.extractive import ExtractiveSummarizer
from src.compression.truncation import SmartTruncator, TruncationStrategy
from src.tokens.counter import ModelFamily, TokenCounter
from src.tokens.budget import TokenBudget, BudgetPriority
from src.benchmarks.retention import RetentionBenchmark


def _resolve_model(model_name: str) -> ModelFamily:
    """Resolve model name string to ModelFamily enum."""
    mapping = {
        "gpt-4": ModelFamily.GPT4,
        "gpt-4o": ModelFamily.GPT4O,
        "gpt-3.5-turbo": ModelFamily.GPT35,
        "claude": ModelFamily.CLAUDE,
        "llama": ModelFamily.LLAMA,
    }
    if model_name not in mapping:
        raise click.BadParameter(
            f"Unknown model: {model_name}. Choose from: {', '.join(mapping.keys())}"
        )
    return mapping[model_name]


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Context Engineering Toolkit — optimize LLM context windows.

    Tools for token counting, context compression, priority assembly,
    and benchmarking context engineering strategies.
    """


@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file")
@click.option("--model", "-m", default="gpt-4o", help="Model for tokenization")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def count(text: str | None, file: str | None, model: str, json_output: bool) -> None:
    """Count tokens in text.

    Reads from argument, --file, or stdin.
    """
    content = _read_input(text, file)
    model_family = _resolve_model(model)
    counter = TokenCounter(model_family)
    result = counter.count(content)

    if json_output:
        click.echo(json.dumps({
            "token_count": result.token_count,
            "model": result.model.value,
            "estimated_input_cost_usd": round(result.estimated_input_cost_usd, 6),
            "context_window": result.context_window,
            "utilization": round(result.utilization, 4),
            "remaining_tokens": result.remaining_tokens,
        }, indent=2))
    else:
        click.echo(f"Tokens:     {result.token_count:,}")
        click.echo(f"Model:      {result.model.value}")
        click.echo(f"Cost (in):  ${result.estimated_input_cost_usd:.6f}")
        click.echo(f"Window:     {result.context_window:,}")
        click.echo(f"Usage:      {result.utilization:.2%}")
        click.echo(f"Remaining:  {result.remaining_tokens:,}")


@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file")
@click.option("--target-tokens", "-t", type=int, required=True, help="Target token count")
@click.option("--model", "-m", default="gpt-4o", help="Model for tokenization")
@click.option("--method", default="extractive", type=click.Choice(["extractive", "truncate"]))
@click.option("--strategy", default="head", type=click.Choice(["head", "tail", "middle"]))
def compress(
    text: str | None,
    file: str | None,
    target_tokens: int,
    model: str,
    method: str,
    strategy: str,
) -> None:
    """Compress text to fit within a target token count.

    Methods:
      extractive — Select most information-dense sentences
      truncate   — Smart token-aware truncation
    """
    content = _read_input(text, file)
    model_family = _resolve_model(model)
    counter = TokenCounter(model_family)

    original_tokens = counter.count(content).token_count
    click.echo(f"Original: {original_tokens:,} tokens", err=True)

    if method == "extractive":
        summarizer = ExtractiveSummarizer(model=model_family)
        compressed = summarizer.compress(content, target_tokens)
    else:
        truncator = SmartTruncator(model=model_family)
        strat = TruncationStrategy(strategy)
        result = truncator.truncate(content, target_tokens, strat)
        compressed = result.text

    compressed_tokens = counter.count(compressed).token_count
    ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
    click.echo(f"Compressed: {compressed_tokens:,} tokens ({ratio:.1%} of original)", err=True)
    click.echo(compressed)


@main.command()
@click.argument("original_file", type=click.Path(exists=True))
@click.argument("compressed_file", type=click.Path(exists=True))
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def benchmark(original_file: str, compressed_file: str, json_output: bool) -> None:
    """Benchmark information retention between original and compressed text."""
    with open(original_file) as f:
        original = f.read()
    with open(compressed_file) as f:
        compressed = f.read()

    bench = RetentionBenchmark()
    result = bench.evaluate(original, compressed)

    if json_output:
        click.echo(json.dumps({
            "compression_ratio": round(result.compression_ratio, 4),
            "overall_score": round(result.overall_score, 4),
            "key_term_retention": round(result.key_term_retention, 4),
            "sentence_coverage": round(result.sentence_coverage, 4),
            "entity_retention": round(result.entity_retention, 4),
            "numeric_retention": round(result.numeric_retention, 4),
        }, indent=2))
    else:
        click.echo(f"Compression ratio:    {result.compression_ratio:.1%}")
        click.echo(f"Overall retention:    {result.overall_score:.1%}")
        click.echo(f"Key term retention:   {result.key_term_retention:.1%}")
        click.echo(f"Sentence coverage:    {result.sentence_coverage:.1%}")
        click.echo(f"Entity retention:     {result.entity_retention:.1%}")
        click.echo(f"Numeric retention:    {result.numeric_retention:.1%}")


@main.command()
def demo() -> None:
    """Run a demonstration of the toolkit's capabilities."""
    click.echo("=" * 60)
    click.echo("Context Engineering Toolkit - Demo")
    click.echo("=" * 60)

    sample_text = (
        "The Transformer architecture was introduced in 2017 by Vaswani et al. "
        "in their paper 'Attention Is All You Need'. It replaced recurrent neural "
        "networks with self-attention mechanisms, achieving state-of-the-art results "
        "on machine translation tasks. The key innovation was the multi-head attention "
        "mechanism, which allows the model to attend to different positions simultaneously. "
        "GPT-4, released by OpenAI in March 2023, uses a transformer architecture with "
        "an estimated 1.8 trillion parameters. It achieves 86.4% accuracy on the MMLU "
        "benchmark, surpassing GPT-3.5's 70% accuracy. Claude, developed by Anthropic, "
        "offers a 200,000 token context window, making it suitable for processing long "
        "documents. The cost of running these models varies: GPT-4 costs approximately "
        "$30 per million input tokens, while GPT-4o costs $2.50 per million tokens. "
        "Fine-tuning smaller models like Llama 3 can reduce inference costs to near zero "
        "for self-hosted deployments."
    )

    # 1. Token counting
    click.echo("\n--- Token Counting ---")
    counter = TokenCounter(ModelFamily.GPT4O)
    result = counter.count(sample_text)
    click.echo(f"Text: {len(sample_text)} chars, {result.token_count} tokens")
    click.echo(f"Cost estimate: ${result.estimated_input_cost_usd:.6f} (input)")
    click.echo(f"Context utilization: {result.utilization:.4%}")

    # 2. Extractive compression
    click.echo("\n--- Extractive Compression (50% target) ---")
    summarizer = ExtractiveSummarizer(model=ModelFamily.GPT4O)
    compressed = summarizer.compress(sample_text, target_tokens=result.token_count // 2)
    compressed_count = counter.count(compressed).token_count
    click.echo(f"Compressed: {compressed_count} tokens (was {result.token_count})")
    click.echo(f"Text: {compressed[:200]}...")

    # 3. Retention benchmark
    click.echo("\n--- Retention Benchmark ---")
    bench = RetentionBenchmark()
    retention = bench.evaluate(sample_text, compressed)
    click.echo(f"Overall retention: {retention.overall_score:.1%}")
    click.echo(f"Key terms: {retention.key_term_retention:.1%}")
    click.echo(f"Entities: {retention.entity_retention:.1%}")
    click.echo(f"Numbers: {retention.numeric_retention:.1%}")

    # 4. Smart truncation
    click.echo("\n--- Smart Truncation (middle-out) ---")
    truncator = SmartTruncator(model=ModelFamily.GPT4O)
    trunc_result = truncator.truncate(
        sample_text, max_tokens=result.token_count // 3, strategy=TruncationStrategy.MIDDLE
    )
    click.echo(f"Truncated: {trunc_result.truncated_tokens} tokens")
    click.echo(f"Strategy: {trunc_result.strategy.value}")
    click.echo(f"Text: {trunc_result.text[:200]}...")

    # 5. Budget management
    click.echo("\n--- Token Budget ---")
    budget = TokenBudget(total_budget=8000, response_reserve=2000)
    budget.add_section("system", "You are a helpful assistant.", 8, priority=BudgetPriority.CRITICAL)
    budget.add_section("context", sample_text, result.token_count, priority=BudgetPriority.HIGH)
    budget.add_section("history", "User: Hello\nAssistant: Hi!", 12, priority=BudgetPriority.MEDIUM)
    report = budget.allocate()
    click.echo(report.summary())

    click.echo("\n" + "=" * 60)
    click.echo("Demo complete.")


def _read_input(text: str | None, file: str | None) -> str:
    """Read input from argument, file, or stdin."""
    if text:
        return text
    if file:
        with open(file) as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise click.UsageError("Provide text as argument, --file, or pipe via stdin.")


if __name__ == "__main__":
    main()
