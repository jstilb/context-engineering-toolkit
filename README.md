# Context Engineering Toolkit

[![Tests](https://github.com/jstilb/context-engineering-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/jstilb/context-engineering-toolkit/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Context window optimization library for LLM applications. Compression, prioritization, and benchmarking tools to maximize the value of every token.

## Why I Built This

Context windows are the critical bottleneck in every LLM application. Whether you are building RAG, agents, or chat systems, you are constantly fighting the trade-off between providing enough context for good answers and staying within token limits. Most teams solve this with naive truncation -- cutting text at an arbitrary character count and hoping for the best.

This toolkit provides principled approaches:
- **Extractive compression** that preserves information density by selecting the most important sentences
- **Token-aware truncation** that respects token boundaries instead of splitting mid-word
- **Priority-based assembly** that ensures the most relevant information fits first
- **Retention benchmarks** that measure whether your context strategy actually preserves key information

The result: smaller context windows with higher information density, lower API costs, and better LLM outputs.

## Features

| Feature | Description |
|---------|-------------|
| Multi-model token counting | Accurate counts for GPT-4, GPT-4o, Claude, Llama |
| Cost estimation | Per-request cost calculation by model |
| Extractive compression | TF-IDF sentence scoring with position bias |
| Smart truncation | Token-aware head/tail/middle-out strategies |
| Priority assembly | REQUIRED > HIGH > MEDIUM > LOW context ordering |
| Token budgeting | Section-based allocation with rebalancing |
| Retention benchmarks | Key term, entity, numeric, sentence coverage metrics |
| CLI interface | Count, compress, benchmark from the command line |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run the demo
python -m src.cli demo

# Count tokens
echo "Your text here" | python -m src.cli count

# Compress a document to 500 tokens
python -m src.cli compress --file document.txt --target-tokens 500

# Benchmark compression quality
python -m src.cli benchmark original.txt compressed.txt
```

## Architecture

```
src/
  tokens/
    counter.py           # Multi-model token counting with tiktoken
    budget.py            # Section-based token budget management
  compression/
    extractive.py        # TF-IDF sentence scoring and selection
    truncation.py        # Token-aware head/tail/middle truncation
  assembly/
    priority.py          # Priority-based context window assembly
  benchmarks/
    retention.py         # Information retention measurement
  cli.py                 # Click-based CLI interface
```

See [docs/architecture.md](docs/architecture.md) for detailed Mermaid diagrams.

## Usage

### Token Counting

```python
from src.tokens.counter import TokenCounter, ModelFamily

counter = TokenCounter(ModelFamily.GPT4O)
result = counter.count("Your text here")

print(f"Tokens: {result.token_count}")
print(f"Cost: ${result.estimated_input_cost_usd:.6f}")
print(f"Window usage: {result.utilization:.2%}")
print(f"Remaining: {result.remaining_tokens:,}")
```

### Extractive Compression

```python
from src.compression.extractive import ExtractiveSummarizer

summarizer = ExtractiveSummarizer(model=ModelFamily.GPT4O)

# Compress to target token count
compressed = summarizer.compress(long_text, target_tokens=500)

# Or compress by ratio
compressed = summarizer.compress_with_ratio(long_text, ratio=0.3)  # 30% of original
```

### Priority-Based Context Assembly

```python
from src.assembly.priority import PriorityAssembler, ContextItem, ContextPriority

assembler = PriorityAssembler(budget_tokens=4000, model=ModelFamily.GPT4O)

# System prompt always included
assembler.add(ContextItem(
    content="You are a helpful assistant.",
    priority=ContextPriority.REQUIRED,
))

# RAG results, ordered by relevance
assembler.add(ContextItem(
    content=rag_chunk_1,
    priority=ContextPriority.HIGH,
    relevance_score=0.95,
    category="retrieved_context",
))

# Chat history as supporting context
assembler.add(ContextItem(
    content=chat_history,
    priority=ContextPriority.MEDIUM,
    category="chat_history",
))

result = assembler.assemble()
print(f"Included {len(result.included_items)} items, excluded {len(result.excluded_items)}")
print(f"Token utilization: {result.utilization:.1%}")
```

### Token Budget Planning

```python
from src.tokens.budget import TokenBudget, BudgetPriority

budget = TokenBudget(total_budget=8000, response_reserve=2000)
budget.add_section("system", system_prompt, 100, priority=BudgetPriority.CRITICAL)
budget.add_section("context", rag_results, 3000, priority=BudgetPriority.HIGH)
budget.add_section("history", chat_history, 2000, priority=BudgetPriority.MEDIUM)

report = budget.allocate()
print(report.summary())

# Rebalance if sections are uneven
rebalanced = budget.rebalance(report)
```

### Retention Benchmarking

```python
from src.benchmarks.retention import RetentionBenchmark

benchmark = RetentionBenchmark()
result = benchmark.evaluate(original_text, compressed_text)

print(f"Overall retention: {result.overall_score:.1%}")
print(f"Key terms: {result.key_term_retention:.1%}")
print(f"Entities: {result.entity_retention:.1%}")
print(f"Numbers: {result.numeric_retention:.1%}")
print(f"Compression ratio: {result.compression_ratio:.1%}")
```

## Design Decisions

- [ADR-001: Extractive Over Abstractive Default](docs/decisions/001-extractive-over-abstractive-default.md)
- [ADR-002: Multi-Model Tokenization Strategy](docs/decisions/002-multi-model-tokenization.md)

## Development

```bash
pip install -e ".[dev]"
make test          # Run tests with coverage
make lint          # Lint with ruff
make typecheck     # Type check with mypy
make demo          # Run interactive demo
```

## License

MIT
