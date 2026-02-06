# ADR-001: Extractive Compression as Default

## Status
Accepted

## Context
We need a default compression strategy for reducing context window usage. Two main approaches exist:

1. **Extractive**: Select the most important sentences from the original text
2. **Abstractive**: Use an LLM to generate a summary

## Decision
Extractive compression is the default strategy.

## Rationale

### Why extractive wins as default:
- **No LLM dependency**: Extractive works without API calls, making it fast and free
- **Deterministic**: Same input always produces same output (important for testing)
- **Information preservation**: Selected sentences are verbatim from the source, reducing hallucination risk
- **Latency**: Runs in milliseconds vs seconds for LLM-based summarization
- **Cost**: Zero additional cost vs LLM inference cost for abstractive

### When abstractive is better:
- When maximum compression is needed (abstractive can synthesize multiple sentences into one)
- When the output needs to be coherent prose rather than selected fragments
- When the compression ratio exceeds 5:1 (extractive becomes incoherent at extreme ratios)

### Trade-offs accepted:
- Extractive may not produce the most readable output
- At high compression ratios (>80% reduction), extractive loses coherence
- Cannot synthesize information across sentences

## Consequences
- Users get fast, free compression out of the box
- Abstractive compression is available as an opt-in for users with LLM access
- Benchmark comparisons include both strategies
