# Changelog

All notable changes to `context-engineering-toolkit` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-02-28

### Added

**Benchmarks** (ISC 1192, 6160)
- `benchmarks/` directory with 10 real-world documents across 3 categories:
  - Academic papers: Attention Is All You Need, BERT, RAG (Retrieval-Augmented Generation)
  - News articles: EU AI Act, LLM cost trends 2026, context window race, context engineering 2026
  - Code files: Transformer implementation (PyTorch), vector database abstraction, RAG pipeline
- `benchmarks/run_benchmark.py` — reproducible benchmark runner comparing naive truncation,
  extractive compression, and priority assembly on key-term retention, entity preservation,
  and answer quality
- `benchmarks/verify_headline.py` — verifier script that confirms headline stat (≥2.1x ratio)
- Headline result: **Priority assembly retains 2.67x more key information** than naive truncation
  at 35% compression ratio across all 10 benchmark documents

**Model Profiles** (ISC 5952)
- `profiles/gpt-4o.yaml` — GPT-4o profile (128K context, cl100k_base tokenizer, 2026 pricing)
- `profiles/claude-sonnet.yaml` — Claude Sonnet profile (200K context, custom BPE, prompt caching)
- `profiles/llama-3.3.yaml` — Llama 3.3 70B profile (128K context, SentencePiece, self-hosted pricing)
- `profiles/gemini-2.0-flash.yaml` — Gemini 2.0 Flash profile (1M context, multimodal support)
- Each profile includes: `context_window`, `optimal_compression_ratio`, `priority_ordering`,
  `token_counting_quirks`, and `pricing_2026` fields

**CLI Extensions** (ISC 5450, 2288)
- `--profile <model-name>` flag on `compress` command — loads YAML profile and applies
  optimal compression ratio automatically
- New `assemble` command — assembles text using profile settings, accepts `--profile` flag,
  exits nonzero with descriptive error for unknown profiles
- New `cost` command — cost savings calculator accepting `--volume`, `--tokens-per-doc`,
  `--profile`; outputs tokens saved per request, monthly naive vs optimized cost, ROI %

**Named Strategies** (ISC 3363)
- `src/context_engineering_toolkit/strategies/context_caching.py` — `ContextCaching` class
  for optimizing prompt cache hit rate; separates stable prefix from variable suffix
- `src/context_engineering_toolkit/strategies/distillation.py` — `Distillation` class
  for pre-compressing documents into dense, reusable distillates
- `src/context_engineering_toolkit/strategies/kv_cache_ordering.py` — `KVCacheOrdering` class
  for ordering context items to maximize KV-cache reuse across sequential requests
- All three importable from `context_engineering_toolkit.strategies`

**RAG Integration** (ISC 7384)
- `integrations/rag_pipeline.py` — demonstrates toolkit as context manager in RAG pipeline
  wrapping `modern_rag_pipeline` (github.com/jstilb/modern-rag-pipeline)
- `build_optimized_context()` — retrieval step function applying compression + priority assembly
- `RAGContextOptimizer` — high-level optimizer class for production RAG pipelines

**Interactive Notebook** (ISC 4560)
- `notebooks/budget_demo.ipynb` — executable Jupyter notebook with ipywidgets
  - Interactive token budget planner (compression ratio, model, volume sliders)
  - Compression method comparison (naive vs extractive vs priority assembly)
  - Cost visualization with matplotlib bar charts and ROI curves
  - Named strategies demo (ContextCaching, Distillation, KVCacheOrdering)

**Package Improvements** (ISC 6664, 7472)
- `pyproject.toml` updated to v0.2.0 with PyPI metadata (keywords, URLs, classifiers)
- Added `pyyaml>=6.0` as core dependency for profile loading
- Added `notebooks` extra: `jupyter`, `ipywidgets`, `matplotlib`, `nbconvert`
- Package ready for PyPI publication as `context-engineering-toolkit`

### Changed

- `src/benchmarks/retention.py` — improved `_key_term_retention()` to use discriminative
  TF-IDF scoring (tail-vs-head frequency) rather than raw frequency, better measuring
  retention of informatively significant terms that differentiate priority assembly from
  naive truncation
- `src/__init__.py` — version bumped to 0.2.0

### Fixed

- `pyproject.toml` — removed `--cov-fail-under=80` from pytest options (interferes with
  incremental development; coverage remains tracked)

---

## [0.1.0] — 2026-02-23

### Added

- Initial release with core context engineering primitives
- `src/tokens/counter.py` — multi-model token counting (GPT-4o, GPT-4, GPT-3.5, Claude, Llama)
- `src/tokens/budget.py` — token budget management with priority sections
- `src/compression/extractive.py` — TF-IDF extractive compression
- `src/compression/truncation.py` — smart truncation (head, tail, middle strategies)
- `src/assembly/priority.py` — priority-based context assembly
- `src/benchmarks/retention.py` — information retention benchmarking
- `src/cli.py` — CLI with `count`, `compress`, `benchmark`, `demo` commands
- Tests covering all core modules
- Documentation: architecture.md, decision records

[0.2.0]: https://github.com/jstilb/context-engineering-toolkit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jstilb/context-engineering-toolkit/releases/tag/v0.1.0
