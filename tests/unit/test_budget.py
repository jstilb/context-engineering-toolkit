"""Tests for the token budget module."""

from src.tokens.budget import BudgetPriority, BudgetReport, BudgetSection, TokenBudget


class TestBudgetSection:
    """Tests for BudgetSection."""

    def test_utilization_within_budget(self) -> None:
        section = BudgetSection(name="test", content="hello", token_count=50, max_tokens=100)
        assert section.utilization == 0.5

    def test_utilization_at_budget(self) -> None:
        section = BudgetSection(name="test", content="hello", token_count=100, max_tokens=100)
        assert section.utilization == 1.0

    def test_over_budget(self) -> None:
        section = BudgetSection(name="test", content="hello", token_count=150, max_tokens=100)
        assert section.over_budget is True
        assert section.overflow == 50

    def test_not_over_budget(self) -> None:
        section = BudgetSection(name="test", content="hello", token_count=50, max_tokens=100)
        assert section.over_budget is False
        assert section.overflow == 0

    def test_zero_max_tokens(self) -> None:
        section = BudgetSection(name="test", content="", token_count=0, max_tokens=0)
        assert section.utilization == 0.0


class TestBudgetReport:
    """Tests for BudgetReport."""

    def test_utilization(self) -> None:
        report = BudgetReport(
            total_tokens=4000,
            total_budget=8000,
            sections=[],
            overhead_tokens=0,
        )
        assert report.utilization == 0.5

    def test_remaining(self) -> None:
        report = BudgetReport(
            total_tokens=3000,
            total_budget=8000,
            sections=[],
            overhead_tokens=200,
        )
        assert report.remaining == 4800

    def test_over_budget_sections(self) -> None:
        sections = [
            BudgetSection("a", "x", 50, 100),
            BudgetSection("b", "y", 150, 100),  # over budget
            BudgetSection("c", "z", 200, 100),  # over budget
        ]
        report = BudgetReport(total_tokens=400, total_budget=1000, sections=sections)
        assert len(report.over_budget_sections) == 2

    def test_summary_output(self) -> None:
        sections = [
            BudgetSection("system", "sys", 50, 100, BudgetPriority.CRITICAL),
            BudgetSection("context", "ctx", 200, 150, BudgetPriority.HIGH),
        ]
        report = BudgetReport(total_tokens=250, total_budget=1000, sections=sections)
        summary = report.summary()
        assert "Token Budget Report" in summary
        assert "system" in summary
        assert "context" in summary
        assert "OVER" in summary  # context is over budget


class TestTokenBudget:
    """Tests for TokenBudget."""

    def test_available_budget(self) -> None:
        budget = TokenBudget(total_budget=8000, response_reserve=2000)
        assert budget.available_budget == 6000

    def test_allocate_empty(self) -> None:
        budget = TokenBudget(total_budget=8000)
        report = budget.allocate()
        assert report.total_tokens == 0
        assert len(report.sections) == 0

    def test_allocate_critical_first(self) -> None:
        budget = TokenBudget(total_budget=8000, response_reserve=1000, overhead_per_section=0)
        budget.add_section("system", "sys", 100, priority=BudgetPriority.CRITICAL)
        budget.add_section("context", "ctx", 500, priority=BudgetPriority.HIGH)
        report = budget.allocate()
        assert len(report.sections) == 2
        # Critical section should be fully allocated
        critical = [s for s in report.sections if s.priority == BudgetPriority.CRITICAL]
        assert critical[0].max_tokens >= critical[0].token_count

    def test_allocate_respects_budget(self) -> None:
        budget = TokenBudget(total_budget=500, response_reserve=100, overhead_per_section=0)
        budget.add_section("a", "x", 200, priority=BudgetPriority.HIGH)
        budget.add_section("b", "y", 200, priority=BudgetPriority.MEDIUM)
        budget.add_section("c", "z", 200, priority=BudgetPriority.LOW)
        report = budget.allocate()
        # Only 400 available (500 - 100 reserve), so not all fit
        total_allocated = sum(s.max_tokens for s in report.sections)
        assert total_allocated <= 400

    def test_rebalance_redistributes_surplus(self) -> None:
        budget = TokenBudget(total_budget=1000, response_reserve=0, overhead_per_section=0)
        # Section a uses 50 of 200 allocation -> 150 surplus
        # Section b uses 250 of 200 allocation -> 50 overflow
        sections = [
            BudgetSection("a", "x", 50, 200, BudgetPriority.HIGH),
            BudgetSection("b", "y", 250, 200, BudgetPriority.MEDIUM),
        ]
        report = BudgetReport(total_tokens=300, total_budget=1000, sections=sections)
        rebalanced = budget.rebalance(report)
        # Section a should shrink, section b should get more
        section_a = [s for s in rebalanced.sections if s.name == "a"][0]
        section_b = [s for s in rebalanced.sections if s.name == "b"][0]
        assert section_a.max_tokens == 50  # Shrunk to actual
        assert section_b.max_tokens > 200  # Got surplus

    def test_add_section_defaults_max_to_token_count(self) -> None:
        budget = TokenBudget(total_budget=8000, overhead_per_section=0)
        budget.add_section("test", "content", 500)
        report = budget.allocate()
        assert report.sections[0].token_count == 500
