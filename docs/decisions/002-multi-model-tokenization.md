# ADR-002: Multi-Model Tokenization Strategy

## Status
Accepted

## Context
Different LLM providers use different tokenizers. Accurate token counting requires knowing which tokenizer the target model uses.

## Decision
Use tiktoken as the tokenization engine for all models, with model-specific encoding selection.

## Rationale

### Encoding mapping:
- **GPT-4**: cl100k_base (exact)
- **GPT-4o**: o200k_base (exact)
- **GPT-3.5**: cl100k_base (exact)
- **Claude**: cl100k_base (approximation, within 5% for most text)
- **Llama**: cl100k_base (approximation, within 10%)

### Why tiktoken for everything:
1. **Single dependency**: Only one tokenization library needed
2. **Speed**: tiktoken is implemented in Rust, extremely fast
3. **Accuracy**: Exact for OpenAI models, close approximation for others
4. **Simplicity**: Unified API regardless of target model

### Why not model-specific tokenizers:
- Anthropic does not publish a public tokenizer library
- Llama tokenizers require downloading model-specific files
- The accuracy difference (5-10%) is acceptable for budget planning
- Users can override with exact counts if needed

## Consequences
- Token counts for Claude and Llama are approximate (stated in docs)
- Cost estimates for non-OpenAI models have a 5-10% margin of error
- Adding new models requires only mapping to the nearest tiktoken encoding
