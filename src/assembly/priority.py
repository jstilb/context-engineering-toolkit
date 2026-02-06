"""Priority-based context assembly for LLM prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.tokens.counter import ModelFamily, TokenCounter


class ContextPriority(Enum):
    """Priority levels for context items."""

    REQUIRED = 0     # Must be included (system prompt, instructions)
    HIGH = 1         # Strongly preferred (most relevant results)
    MEDIUM = 2       # Helpful context (supporting info)
    LOW = 3          # Nice to have (background, examples)
    OPTIONAL = 4     # Include only if space remains


@dataclass(frozen=True)
class ContextItem:
    """A single piece of context with metadata for assembly decisions.

    Attributes:
        content: The text content.
        priority: How important this context is.
        source: Where this content came from (for attribution).
        relevance_score: Optional score from retrieval (0.0 to 1.0).
        category: Grouping category (e.g., "rag_results", "chat_history").
    """

    content: str
    priority: ContextPriority = ContextPriority.MEDIUM
    source: str = ""
    relevance_score: float = 0.0
    category: str = "general"


@dataclass
class AssemblyResult:
    """Result of context assembly."""

    assembled_text: str
    included_items: list[ContextItem]
    excluded_items: list[ContextItem]
    total_tokens: int
    budget_tokens: int

    @property
    def inclusion_rate(self) -> float:
        """Fraction of items that were included."""
        total = len(self.included_items) + len(self.excluded_items)
        if total == 0:
            return 1.0
        return len(self.included_items) / total

    @property
    def utilization(self) -> float:
        """Fraction of budget used."""
        if self.budget_tokens == 0:
            return 0.0
        return self.total_tokens / self.budget_tokens


class PriorityAssembler:
    """Assemble context items into a prompt, prioritizing by importance.

    Implements a greedy algorithm that:
    1. Always includes REQUIRED items (errors if they exceed budget)
    2. Sorts remaining items by priority, then relevance score
    3. Greedily adds items until budget is exhausted
    4. Optionally groups items by category with headers

    Example:
        assembler = PriorityAssembler(budget_tokens=4000, model=ModelFamily.GPT4O)
        assembler.add(ContextItem("System: You are...", ContextPriority.REQUIRED))
        assembler.add(ContextItem("Doc chunk 1...", ContextPriority.HIGH, relevance_score=0.95))
        assembler.add(ContextItem("Doc chunk 2...", ContextPriority.MEDIUM, relevance_score=0.72))
        result = assembler.assemble()
    """

    def __init__(
        self,
        budget_tokens: int,
        model: ModelFamily = ModelFamily.GPT4O,
        separator: str = "\n\n---\n\n",
        category_headers: bool = True,
    ) -> None:
        """Initialize assembler.

        Args:
            budget_tokens: Maximum tokens in assembled context.
            model: Model family for token counting.
            separator: Text between context items.
            category_headers: Whether to add category headers.
        """
        self.budget_tokens = budget_tokens
        self._counter = TokenCounter(model)
        self.separator = separator
        self.category_headers = category_headers
        self._items: list[ContextItem] = []

    def add(self, item: ContextItem) -> None:
        """Add a context item for assembly."""
        self._items.append(item)

    def add_many(self, items: list[ContextItem]) -> None:
        """Add multiple context items."""
        self._items.extend(items)

    def clear(self) -> None:
        """Remove all items."""
        self._items.clear()

    def assemble(self) -> AssemblyResult:
        """Assemble context items into a single text within budget.

        Returns:
            AssemblyResult with assembled text and metadata.

        Raises:
            ValueError: If REQUIRED items exceed the budget.
        """
        if not self._items:
            return AssemblyResult(
                assembled_text="",
                included_items=[],
                excluded_items=[],
                total_tokens=0,
                budget_tokens=self.budget_tokens,
            )

        sep_tokens = self._counter.count(self.separator).token_count

        # Phase 1: Include all REQUIRED items
        required = [i for i in self._items if i.priority == ContextPriority.REQUIRED]
        optional = [i for i in self._items if i.priority != ContextPriority.REQUIRED]

        included: list[ContextItem] = []
        excluded: list[ContextItem] = []
        used_tokens = 0

        for item in required:
            item_tokens = self._counter.count(item.content).token_count
            cost = item_tokens + (sep_tokens if included else 0)
            if used_tokens + cost > self.budget_tokens:
                raise ValueError(
                    f"REQUIRED items exceed budget: {used_tokens + cost} > {self.budget_tokens}. "
                    f"Cannot assemble context — increase budget or reduce required content."
                )
            used_tokens += cost
            included.append(item)

        # Phase 2: Sort optional items by priority, then relevance
        optional.sort(
            key=lambda i: (i.priority.value, -i.relevance_score)
        )

        for item in optional:
            item_tokens = self._counter.count(item.content).token_count
            cost = item_tokens + sep_tokens
            if used_tokens + cost <= self.budget_tokens:
                used_tokens += cost
                included.append(item)
            else:
                excluded.append(item)

        # Build assembled text
        if self.category_headers:
            assembled_text = self._assemble_with_categories(included)
        else:
            assembled_text = self.separator.join(item.content for item in included)

        actual_tokens = self._counter.count(assembled_text).token_count

        return AssemblyResult(
            assembled_text=assembled_text,
            included_items=included,
            excluded_items=excluded,
            total_tokens=actual_tokens,
            budget_tokens=self.budget_tokens,
        )

    def _assemble_with_categories(self, items: list[ContextItem]) -> str:
        """Group items by category and add headers."""
        categories: dict[str, list[ContextItem]] = {}
        for item in items:
            categories.setdefault(item.category, []).append(item)

        parts: list[str] = []
        for category, category_items in categories.items():
            if category != "general" and len(categories) > 1:
                parts.append(f"## {category.replace('_', ' ').title()}")
            for item in category_items:
                parts.append(item.content)

        return self.separator.join(parts)
