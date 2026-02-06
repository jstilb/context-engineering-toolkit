"""Token budget management for context window allocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BudgetPriority(Enum):
    """Priority levels for context sections."""

    CRITICAL = 0   # System prompt, tool definitions — always included
    HIGH = 1       # Most relevant context, recent messages
    MEDIUM = 2     # Supporting context, earlier messages
    LOW = 3        # Nice-to-have, background info


@dataclass
class BudgetSection:
    """A named section of the context window with a token budget.

    Attributes:
        name: Human-readable section name.
        content: The text content for this section.
        token_count: Actual token count of content.
        max_tokens: Maximum tokens allocated to this section.
        priority: How important this section is.
        compressible: Whether this section can be compressed to fit.
    """

    name: str
    content: str
    token_count: int
    max_tokens: int
    priority: BudgetPriority = BudgetPriority.MEDIUM
    compressible: bool = True

    @property
    def utilization(self) -> float:
        """Fraction of budget used."""
        if self.max_tokens == 0:
            return 0.0
        return self.token_count / self.max_tokens

    @property
    def over_budget(self) -> bool:
        """Whether this section exceeds its allocation."""
        return self.token_count > self.max_tokens

    @property
    def overflow(self) -> int:
        """How many tokens over budget (0 if within budget)."""
        return max(0, self.token_count - self.max_tokens)


@dataclass
class BudgetReport:
    """Summary of token budget utilization."""

    total_tokens: int
    total_budget: int
    sections: list[BudgetSection]
    overhead_tokens: int = 0  # Formatting, separators, etc.

    @property
    def utilization(self) -> float:
        """Overall context window utilization."""
        if self.total_budget == 0:
            return 0.0
        return (self.total_tokens + self.overhead_tokens) / self.total_budget

    @property
    def remaining(self) -> int:
        """Tokens remaining for response generation."""
        return max(0, self.total_budget - self.total_tokens - self.overhead_tokens)

    @property
    def over_budget_sections(self) -> list[BudgetSection]:
        """Sections that exceed their allocation."""
        return [s for s in self.sections if s.over_budget]

    def summary(self) -> str:
        """Human-readable budget summary."""
        lines = [
            f"Token Budget Report",
            f"  Total: {self.total_tokens:,} / {self.total_budget:,} "
            f"({self.utilization:.1%})",
            f"  Overhead: {self.overhead_tokens:,}",
            f"  Remaining: {self.remaining:,}",
            f"  Sections:",
        ]
        for section in sorted(self.sections, key=lambda s: s.priority.value):
            status = "OVER" if section.over_budget else "OK"
            lines.append(
                f"    [{status}] {section.name}: "
                f"{section.token_count:,} / {section.max_tokens:,} "
                f"({section.utilization:.1%}) "
                f"[{section.priority.name}]"
            )
        return "\n".join(lines)


class TokenBudget:
    """Manages token allocation across context window sections.

    Implements priority-based budgeting where critical sections are
    allocated first, then remaining budget is distributed by priority.

    Example:
        budget = TokenBudget(total_budget=8000, response_reserve=2000)
        budget.add_section("system", system_prompt, 500, priority=BudgetPriority.CRITICAL)
        budget.add_section("context", rag_results, 3000, priority=BudgetPriority.HIGH)
        budget.add_section("history", chat_history, 4000, priority=BudgetPriority.MEDIUM)
        report = budget.allocate()
    """

    def __init__(
        self,
        total_budget: int,
        response_reserve: int = 1000,
        overhead_per_section: int = 20,
    ) -> None:
        """Initialize budget manager.

        Args:
            total_budget: Total context window size in tokens.
            response_reserve: Tokens to reserve for model response.
            overhead_per_section: Estimated formatting overhead per section.
        """
        self.total_budget = total_budget
        self.response_reserve = response_reserve
        self.overhead_per_section = overhead_per_section
        self._sections: list[BudgetSection] = []

    @property
    def available_budget(self) -> int:
        """Budget available for content (total minus response reserve)."""
        return self.total_budget - self.response_reserve

    def add_section(
        self,
        name: str,
        content: str,
        token_count: int,
        max_tokens: int | None = None,
        priority: BudgetPriority = BudgetPriority.MEDIUM,
        compressible: bool = True,
    ) -> None:
        """Add a section to the budget.

        Args:
            name: Section name.
            content: Section text content.
            token_count: Actual token count of content.
            max_tokens: Maximum tokens for this section (defaults to token_count).
            priority: Section priority.
            compressible: Whether content can be compressed.
        """
        if max_tokens is None:
            max_tokens = token_count

        self._sections.append(
            BudgetSection(
                name=name,
                content=content,
                token_count=token_count,
                max_tokens=max_tokens,
                priority=priority,
                compressible=compressible,
            )
        )

    def allocate(self) -> BudgetReport:
        """Allocate budget across sections by priority.

        Critical sections get their full allocation first.
        Remaining budget is distributed to lower-priority sections
        proportionally to their requested size.

        Returns:
            BudgetReport with allocation results.
        """
        # Sort by priority (critical first)
        sorted_sections = sorted(self._sections, key=lambda s: s.priority.value)

        overhead = len(sorted_sections) * self.overhead_per_section
        remaining = self.available_budget - overhead

        allocated: list[BudgetSection] = []

        for section in sorted_sections:
            if section.priority == BudgetPriority.CRITICAL:
                # Critical sections always get their full allocation
                alloc = min(section.token_count, remaining)
                remaining -= alloc
                allocated.append(
                    BudgetSection(
                        name=section.name,
                        content=section.content,
                        token_count=section.token_count,
                        max_tokens=alloc,
                        priority=section.priority,
                        compressible=section.compressible,
                    )
                )
            else:
                # Non-critical sections share remaining budget
                alloc = min(section.token_count, max(0, remaining))
                remaining -= alloc
                allocated.append(
                    BudgetSection(
                        name=section.name,
                        content=section.content,
                        token_count=section.token_count,
                        max_tokens=alloc,
                        priority=section.priority,
                        compressible=section.compressible,
                    )
                )

        total_tokens = sum(s.token_count for s in allocated)

        return BudgetReport(
            total_tokens=total_tokens,
            total_budget=self.total_budget,
            sections=allocated,
            overhead_tokens=overhead,
        )

    def rebalance(self, report: BudgetReport) -> BudgetReport:
        """Rebalance budget by redistributing unused tokens from under-budget sections.

        Takes tokens from sections using less than their allocation
        and gives them to sections that are over budget.

        Args:
            report: Current budget report.

        Returns:
            New BudgetReport with rebalanced allocations.
        """
        # Calculate surplus from under-budget sections
        surplus = 0
        needs = []
        balanced = []

        for section in report.sections:
            if section.token_count < section.max_tokens:
                surplus += section.max_tokens - section.token_count
                balanced.append(
                    BudgetSection(
                        name=section.name,
                        content=section.content,
                        token_count=section.token_count,
                        max_tokens=section.token_count,  # Shrink to actual
                        priority=section.priority,
                        compressible=section.compressible,
                    )
                )
            elif section.over_budget:
                needs.append(section)
            else:
                balanced.append(section)

        # Distribute surplus to over-budget sections by priority
        needs.sort(key=lambda s: s.priority.value)
        for section in needs:
            give = min(section.overflow, surplus)
            surplus -= give
            balanced.append(
                BudgetSection(
                    name=section.name,
                    content=section.content,
                    token_count=section.token_count,
                    max_tokens=section.max_tokens + give,
                    priority=section.priority,
                    compressible=section.compressible,
                )
            )

        total_tokens = sum(s.token_count for s in balanced)
        return BudgetReport(
            total_tokens=total_tokens,
            total_budget=report.total_budget,
            sections=balanced,
            overhead_tokens=report.overhead_tokens,
        )
