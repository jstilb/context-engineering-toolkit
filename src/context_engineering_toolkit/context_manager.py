"""Context manager wrapper for the Context Engineering Toolkit.

Provides a context manager interface for use in RAG pipelines and
other integrations that need clean setup/teardown semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.tokens.counter import ModelFamily


class ContextEngineeringToolkit:
    """Context manager for the Context Engineering Toolkit.

    Provides a clean interface for using the toolkit as a Python context
    manager in RAG pipelines and other integrations. Handles initialization,
    configuration, and cleanup automatically.

    Example:
        with ContextEngineeringToolkit(model="claude-sonnet", budget=8000) as ctx:
            compressed = ctx.compress(document_text)
            context = ctx.assemble([system_prompt, compressed, user_query])

    Args:
        model: Model profile to use (e.g., "gpt-4o", "claude-sonnet").
        budget: Maximum token budget for assembled context.
        profile_dir: Directory containing YAML model profiles.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        budget: int = 4096,
        profile_dir: str | None = None,
    ) -> None:
        self.model = model
        self.budget = budget
        self.profile_dir = profile_dir
        self._profile: dict[str, Any] | None = None
        self._active = False

    def __enter__(self) -> ContextEngineeringToolkit:
        """Initialize toolkit and load model profile."""
        self._load_profile()
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up toolkit resources."""
        self._active = False
        self._profile = None

    def _load_profile(self) -> None:
        """Load model profile from YAML if profile_dir is set."""
        if self.profile_dir is None:
            return

        from pathlib import Path

        import yaml

        profile_path = Path(self.profile_dir) / f"{self.model}.yaml"
        if profile_path.exists():
            self._profile = yaml.safe_load(profile_path.read_text())

    def compress(self, text: str, target_tokens: int | None = None) -> str:
        """Compress text using the configured model's optimal compression ratio.

        Args:
            text: Text to compress.
            target_tokens: Override target token count. If None, uses
                           budget * optimal_compression_ratio from profile.

        Returns:
            Compressed text.
        """
        from src.compression.extractive import ExtractiveSummarizer
        from src.tokens.counter import TokenCounter

        model_family = self._resolve_model_family()
        counter = TokenCounter(model_family)
        summarizer = ExtractiveSummarizer(model=model_family)

        if target_tokens is None:
            ratio = 0.45  # default
            if self._profile:
                ratio = self._profile.get("optimal_compression_ratio", 0.45)
            original_tokens = counter.count(text).token_count
            target_tokens = int(original_tokens * ratio)

        return summarizer.compress(text, target_tokens)

    def assemble(self, items: list[str]) -> str:
        """Assemble context items within the token budget.

        Args:
            items: List of text items in priority order (first = highest priority).

        Returns:
            Assembled context string within budget.
        """
        from src.assembly.priority import ContextItem, ContextPriority, PriorityAssembler

        model_family = self._resolve_model_family()
        assembler = PriorityAssembler(budget_tokens=self.budget, model=model_family)

        priorities = [
            ContextPriority.REQUIRED,
            ContextPriority.HIGH,
            ContextPriority.MEDIUM,
            ContextPriority.LOW,
            ContextPriority.OPTIONAL,
        ]

        for i, item in enumerate(items):
            priority = priorities[min(i, len(priorities) - 1)]
            assembler.add(ContextItem(content=item, priority=priority))

        result = assembler.assemble()
        return result.assembled_text

    def _resolve_model_family(self) -> ModelFamily:
        """Resolve model name to ModelFamily enum."""
        from src.tokens.counter import ModelFamily

        mapping = {
            "gpt-4o": ModelFamily.GPT4O,
            "gpt-4": ModelFamily.GPT4,
            "gpt-3.5-turbo": ModelFamily.GPT35,
            "claude-sonnet": ModelFamily.CLAUDE,
            "claude": ModelFamily.CLAUDE,
            "llama-3.3": ModelFamily.LLAMA,
            "llama": ModelFamily.LLAMA,
        }
        return mapping.get(self.model, ModelFamily.GPT4O)
