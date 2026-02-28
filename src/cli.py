"""CLI interface for the Context Engineering Toolkit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import click

from src.compression.extractive import ExtractiveSummarizer
from src.compression.truncation import SmartTruncator, TruncationStrategy
from src.tokens.counter import ModelFamily, TokenCounter
from src.tokens.budget import TokenBudget, BudgetPriority
from src.benchmarks.retention import RetentionBenchmark


# Profiles directory (relative to the package root)
_DEFAULT_PROFILES_DIR = Path(__file__).parent.parent / "profiles"

# 2026 pricing per million tokens (USD). Source: provider docs 2026-02
PRICING_2026: dict[str, dict[str, float]] = {
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
        "cached_input": 1.25,
    },
    "claude-sonnet": {
        "input": 3.00,
        "output": 15.00,
        "cached_input": 0.30,
    },
    "llama-3.3": {
        "input": 0.59,
        "output": 0.79,
        "cached_input": 0.59,  # No caching discount for self-hosted
    },
    "gemini-2.0-flash": {
        "input": 0.075,
        "output": 0.30,
        "cached_input": 0.01875,
    },
    # Aliases
    "gpt-4": {
        "input": 30.00,
        "output": 60.00,
        "cached_input": 15.00,
    },
    "gpt-3.5-turbo": {
        "input": 0.50,
        "output": 1.50,
        "cached_input": 0.25,
    },
    "claude": {
        "input": 3.00,
        "output": 15.00,
        "cached_input": 0.30,
    },
    "llama": {
        "input": 0.59,
        "output": 0.79,
        "cached_input": 0.59,
    },
}

KNOWN_MODEL_NAMES = list(PRICING_2026.keys())


def _resolve_model(model_name: str) -> ModelFamily:
    """Resolve model name string to ModelFamily enum."""
    mapping = {
        "gpt-4": ModelFamily.GPT4,
        "gpt-4o": ModelFamily.GPT4O,
        "gpt-3.5-turbo": ModelFamily.GPT35,
        "claude": ModelFamily.CLAUDE,
        "claude-sonnet": ModelFamily.CLAUDE,
        "llama": ModelFamily.LLAMA,
        "llama-3.3": ModelFamily.LLAMA,
        "gemini-2.0-flash": ModelFamily.GPT4O,  # Use GPT4O tokenizer as approximation
    }
    if model_name not in mapping:
        raise click.BadParameter(
            f"Unknown model: {model_name}. Choose from: {', '.join(mapping.keys())}"
        )
    return mapping[model_name]


def _load_profile(profile_name: str, profiles_dir: Optional[Path] = None) -> dict[str, Any]:
    """Load a model profile YAML file.

    Args:
        profile_name: Model name (e.g., "gpt-4o", "claude-sonnet").
        profiles_dir: Directory containing profile YAML files.

    Returns:
        Parsed profile dict.

    Raises:
        click.BadParameter: If profile not found.
    """
    try:
        import yaml
    except ImportError:
        raise click.ClickException("PyYAML is required for --profile. Install with: pip install pyyaml")

    search_dirs = [profiles_dir] if profiles_dir else [_DEFAULT_PROFILES_DIR]

    for search_dir in search_dirs:
        profile_path = search_dir / f"{profile_name}.yaml"
        if profile_path.exists():
            return yaml.safe_load(profile_path.read_text())  # type: ignore[no-any-return]

    # List available profiles for helpful error
    available: list[str] = []
    for search_dir in search_dirs:
        if search_dir.exists():
            available.extend(p.stem for p in search_dir.glob("*.yaml"))

    msg = f"Profile '{profile_name}' not found."
    if available:
        msg += f" Available profiles: {', '.join(sorted(available))}"
    raise click.BadParameter(msg, param_hint="--profile")


@click.group()
@click.version_option(version="0.2.0")
def main() -> None:
    """Context Engineering Toolkit — optimize LLM context windows.

    Tools for token counting, context compression, priority assembly,
    benchmarking, cost calculation, and context engineering strategies.
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
@click.option("--target-tokens", "-t", type=int, required=False, help="Target token count")
@click.option("--model", "-m", default="gpt-4o", help="Model for tokenization")
@click.option("--profile", "-p", default=None, help="Model profile name (e.g., gpt-4o, claude-sonnet)")
@click.option("--method", default="extractive", type=click.Choice(["extractive", "truncate"]))
@click.option("--strategy", default="head", type=click.Choice(["head", "tail", "middle"]))
def compress(
    text: str | None,
    file: str | None,
    target_tokens: int | None,
    model: str,
    profile: str | None,
    method: str,
    strategy: str,
) -> None:
    """Compress text to fit within a target token count.

    Methods:
      extractive — Select most information-dense sentences
      truncate   — Smart token-aware truncation
    """
    content = _read_input(text, file)

    # Profile overrides model and provides default compression ratio
    compression_ratio: float | None = None
    if profile is not None:
        profile_data = _load_profile(profile)
        # Map profile model_id to our model names
        model = profile_data.get("model_id", model)
        compression_ratio = profile_data.get("optimal_compression_ratio")

    model_family = _resolve_model(model)
    counter = TokenCounter(model_family)

    original_tokens = counter.count(content).token_count
    click.echo(f"Original: {original_tokens:,} tokens", err=True)

    # Determine target tokens
    if target_tokens is None:
        if compression_ratio is not None:
            target_tokens = int(original_tokens * compression_ratio)
            click.echo(
                f"Using profile compression ratio {compression_ratio:.0%} "
                f"→ target: {target_tokens:,} tokens",
                err=True,
            )
        else:
            raise click.UsageError("Provide --target-tokens or --profile with an optimal_compression_ratio")

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
@click.argument("input_file", type=click.Path(exists=True), required=True)
@click.option("--profile", "-p", required=True, help="Model profile name (e.g., gpt-4o, claude-sonnet)")
@click.option("--target-tokens", "-t", type=int, default=None, help="Override target token count")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def assemble(input_file: str, profile: str, target_tokens: int | None, json_output: bool) -> None:
    """Assemble text using a model profile's optimal settings.

    Loads the profile YAML, applies the optimal compression ratio,
    and assembles the context within the configured token budget.
    """
    profile_data = _load_profile(profile)
    model_name = profile_data.get("model_id", "gpt-4o")
    compression_ratio = profile_data.get("optimal_compression_ratio", 0.45)
    context_window = profile_data.get("context_window", 128000)

    model_family = _resolve_model(model_name)
    counter = TokenCounter(model_family)
    summarizer = ExtractiveSummarizer(model=model_family)

    content = Path(input_file).read_text()
    original_tokens = counter.count(content).token_count

    if target_tokens is None:
        target_tokens = int(original_tokens * compression_ratio)

    compressed = summarizer.compress(content, target_tokens)
    compressed_tokens = counter.count(compressed).token_count

    if json_output:
        click.echo(json.dumps({
            "profile": profile,
            "model": model_name,
            "context_window": context_window,
            "optimal_compression_ratio": compression_ratio,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "actual_ratio": round(compressed_tokens / original_tokens, 4) if original_tokens > 0 else 1.0,
            "fits_in_window": compressed_tokens <= context_window,
            "text": compressed,
        }, indent=2))
    else:
        click.echo(f"Profile:     {profile} ({model_name})", err=True)
        click.echo(f"Window:      {context_window:,} tokens", err=True)
        click.echo(f"Original:    {original_tokens:,} tokens", err=True)
        click.echo(f"Compressed:  {compressed_tokens:,} tokens", err=True)
        click.echo(f"Ratio:       {compressed_tokens/original_tokens:.1%}" if original_tokens else "N/A", err=True)
        click.echo(compressed)


@main.command()
@click.option("--volume", "-v", type=int, required=True, help="Monthly request volume")
@click.option("--tokens-per-doc", "-t", type=int, required=True, help="Average document token count per request")
@click.option("--profile", "-p", default="claude-sonnet", help="Model profile name (e.g., gpt-4o, claude-sonnet)")
@click.option("--output-tokens", type=int, default=500, help="Average output tokens per request (default: 500)")
@click.option("--compression-ratio", type=float, default=None, help="Override compression ratio (e.g., 0.4)")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def cost(
    volume: int,
    tokens_per_doc: int,
    profile: str,
    output_tokens: int,
    compression_ratio: float | None,
    json_output: bool,
) -> None:
    """Calculate cost savings from context optimization.

    Compares monthly cost under naive (uncompressed) vs optimized (compressed)
    context engineering using 2026 model pricing.

    \b
    Example:
        ctx-toolkit cost --volume 100000 --tokens-per-doc 8000 --profile claude-sonnet

    This outputs:
        - Tokens saved per request
        - Monthly cost under naive vs. optimized
        - ROI percentage
    """
    # Load profile to get compression ratio if not overridden
    profile_data: dict[str, Any] = {}
    try:
        profile_data = _load_profile(profile)
    except click.BadParameter:
        # Profile not found: use defaults from PRICING_2026
        pass

    # Determine compression ratio
    if compression_ratio is None:
        compression_ratio = profile_data.get("optimal_compression_ratio", 0.45)

    # Get pricing (from profile or fallback to hardcoded 2026 rates)
    pricing = PRICING_2026.get(profile, PRICING_2026.get("claude-sonnet"))
    if pricing is None:
        pricing = {"input": 3.00, "output": 15.00, "cached_input": 0.30}

    # Also use profile pricing if available
    if "pricing_2026" in profile_data:
        profile_pricing = profile_data["pricing_2026"]
        pricing = {
            "input": profile_pricing.get("input_per_million", pricing["input"]),
            "output": profile_pricing.get("output_per_million", pricing["output"]),
            "cached_input": profile_pricing.get("cached_input_per_million", pricing.get("cached_input", pricing["input"] * 0.5)),
        }

    input_rate = pricing["input"]  # USD per million input tokens
    output_rate = pricing["output"]  # USD per million output tokens

    # Naive (uncompressed) scenario
    naive_input_tokens_per_request = tokens_per_doc
    naive_total_input_tokens = naive_input_tokens_per_request * volume
    naive_total_output_tokens = output_tokens * volume

    naive_input_cost = (naive_total_input_tokens / 1_000_000) * input_rate
    naive_output_cost = (naive_total_output_tokens / 1_000_000) * output_rate
    naive_total_cost = naive_input_cost + naive_output_cost

    # Optimized (compressed) scenario
    optimized_input_tokens_per_request = int(tokens_per_doc * compression_ratio)
    tokens_saved_per_request = naive_input_tokens_per_request - optimized_input_tokens_per_request
    total_tokens_saved = tokens_saved_per_request * volume

    optimized_total_input_tokens = optimized_input_tokens_per_request * volume
    optimized_input_cost = (optimized_total_input_tokens / 1_000_000) * input_rate
    optimized_output_cost = naive_output_cost  # Output tokens don't change
    optimized_total_cost = optimized_input_cost + optimized_output_cost

    monthly_savings = naive_total_cost - optimized_total_cost
    roi_pct = (monthly_savings / naive_total_cost * 100) if naive_total_cost > 0 else 0.0

    if json_output:
        click.echo(json.dumps({
            "profile": profile,
            "model_pricing_source": "2026 pricing (see profiles/*.yaml)",
            "inputs": {
                "monthly_volume": volume,
                "tokens_per_doc": tokens_per_doc,
                "output_tokens_per_request": output_tokens,
                "compression_ratio": compression_ratio,
            },
            "pricing": {
                "input_per_million_usd": input_rate,
                "output_per_million_usd": output_rate,
            },
            "tokens_saved_per_request": tokens_saved_per_request,
            "monthly_naive_cost_usd": round(naive_total_cost, 2),
            "monthly_optimized_cost_usd": round(optimized_total_cost, 2),
            "monthly_savings_usd": round(monthly_savings, 2),
            "roi_percentage": round(roi_pct, 1),
            "annual_savings_usd": round(monthly_savings * 12, 2),
            "total_tokens_saved_monthly": total_tokens_saved,
        }, indent=2))
    else:
        click.echo(f"\n{'='*55}")
        click.echo(f"Cost Savings Calculator — 2026 Pricing")
        click.echo(f"{'='*55}")
        click.echo(f"\nConfiguration:")
        click.echo(f"  Profile:             {profile}")
        click.echo(f"  Monthly volume:      {volume:,} requests")
        click.echo(f"  Tokens per document: {tokens_per_doc:,}")
        click.echo(f"  Output tokens/req:   {output_tokens:,}")
        click.echo(f"  Compression ratio:   {compression_ratio:.0%}")
        click.echo(f"\nPricing (2026):")
        click.echo(f"  Input:   ${input_rate:.4f}/million tokens")
        click.echo(f"  Output:  ${output_rate:.4f}/million tokens")
        click.echo(f"\n{'─'*55}")
        click.echo(f"Tokens saved per request: {tokens_saved_per_request:,}")
        click.echo(f"{'─'*55}")
        click.echo(f"\n  NAIVE implementation:")
        click.echo(f"    Input tokens:   {naive_total_input_tokens:,}/month")
        click.echo(f"    Monthly cost:   ${naive_total_cost:,.2f}")
        click.echo(f"\n  OPTIMIZED implementation:")
        click.echo(f"    Input tokens:   {optimized_total_input_tokens:,}/month")
        click.echo(f"    Monthly cost:   ${optimized_total_cost:,.2f}")
        click.echo(f"\n{'─'*55}")
        click.echo(f"  Monthly savings:  ${monthly_savings:,.2f}")
        click.echo(f"  Annual savings:   ${monthly_savings*12:,.2f}")
        click.echo(f"  ROI:              {roi_pct:.1f}%")
        click.echo(f"{'='*55}\n")


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
