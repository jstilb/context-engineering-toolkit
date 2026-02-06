"""Tests for context assembly module."""

import pytest

from src.assembly.priority import (
    AssemblyResult,
    ContextItem,
    ContextPriority,
    PriorityAssembler,
)
from src.tokens.counter import ModelFamily


class TestContextItem:
    """Tests for ContextItem."""

    def test_defaults(self) -> None:
        item = ContextItem(content="Hello")
        assert item.priority == ContextPriority.MEDIUM
        assert item.source == ""
        assert item.relevance_score == 0.0
        assert item.category == "general"

    def test_custom_values(self) -> None:
        item = ContextItem(
            content="Test",
            priority=ContextPriority.HIGH,
            source="rag",
            relevance_score=0.95,
            category="search_results",
        )
        assert item.priority == ContextPriority.HIGH
        assert item.relevance_score == 0.95


class TestPriorityAssembler:
    """Tests for PriorityAssembler."""

    def test_empty_assembly(self) -> None:
        assembler = PriorityAssembler(budget_tokens=4000)
        result = assembler.assemble()
        assert result.assembled_text == ""
        assert result.total_tokens == 0
        assert result.inclusion_rate == 1.0

    def test_single_item_fits(self) -> None:
        assembler = PriorityAssembler(budget_tokens=4000)
        assembler.add(ContextItem(content="Hello, world!"))
        result = assembler.assemble()
        assert "Hello, world!" in result.assembled_text
        assert len(result.included_items) == 1
        assert len(result.excluded_items) == 0

    def test_required_items_always_included(self) -> None:
        assembler = PriorityAssembler(budget_tokens=4000)
        assembler.add(ContextItem(
            content="System prompt: You are helpful.",
            priority=ContextPriority.REQUIRED,
        ))
        assembler.add(ContextItem(content="Optional context.", priority=ContextPriority.LOW))
        result = assembler.assemble()
        required = [i for i in result.included_items if i.priority == ContextPriority.REQUIRED]
        assert len(required) == 1

    def test_required_exceeding_budget_raises(self) -> None:
        assembler = PriorityAssembler(budget_tokens=5)  # Very small budget
        assembler.add(ContextItem(
            content="This is a very long system prompt that exceeds the tiny budget. " * 10,
            priority=ContextPriority.REQUIRED,
        ))
        with pytest.raises(ValueError, match="REQUIRED items exceed budget"):
            assembler.assemble()

    def test_priority_ordering(self) -> None:
        assembler = PriorityAssembler(budget_tokens=200, category_headers=False)
        assembler.add(ContextItem(content="Low priority item", priority=ContextPriority.LOW))
        assembler.add(ContextItem(content="High priority item", priority=ContextPriority.HIGH))
        result = assembler.assemble()
        # High priority should be included before low
        included_priorities = [i.priority for i in result.included_items]
        if len(included_priorities) >= 2:
            assert included_priorities[0].value <= included_priorities[1].value

    def test_budget_enforcement(self) -> None:
        assembler = PriorityAssembler(budget_tokens=50)
        for i in range(20):
            assembler.add(ContextItem(
                content=f"This is context item number {i} with some extra text to use tokens.",
                priority=ContextPriority.MEDIUM,
            ))
        result = assembler.assemble()
        assert result.total_tokens <= 50
        assert len(result.excluded_items) > 0

    def test_relevance_score_tiebreaker(self) -> None:
        assembler = PriorityAssembler(budget_tokens=200, category_headers=False)
        assembler.add(ContextItem(
            content="Less relevant", priority=ContextPriority.HIGH, relevance_score=0.5
        ))
        assembler.add(ContextItem(
            content="More relevant", priority=ContextPriority.HIGH, relevance_score=0.9
        ))
        result = assembler.assemble()
        if len(result.included_items) >= 2:
            # Higher relevance should come first among same priority
            assert result.included_items[0].relevance_score >= result.included_items[1].relevance_score

    def test_add_many(self) -> None:
        assembler = PriorityAssembler(budget_tokens=4000)
        items = [
            ContextItem(content=f"Item {i}") for i in range(5)
        ]
        assembler.add_many(items)
        result = assembler.assemble()
        assert len(result.included_items) == 5

    def test_clear(self) -> None:
        assembler = PriorityAssembler(budget_tokens=4000)
        assembler.add(ContextItem(content="Test"))
        assembler.clear()
        result = assembler.assemble()
        assert len(result.included_items) == 0

    def test_category_headers(self) -> None:
        assembler = PriorityAssembler(budget_tokens=4000, category_headers=True)
        assembler.add(ContextItem(content="Search result 1", category="search_results"))
        assembler.add(ContextItem(content="Chat message 1", category="chat_history"))
        result = assembler.assemble()
        assert "Search Results" in result.assembled_text
        assert "Chat History" in result.assembled_text

    def test_utilization(self) -> None:
        assembler = PriorityAssembler(budget_tokens=10000)
        assembler.add(ContextItem(content="Short text"))
        result = assembler.assemble()
        assert 0.0 < result.utilization < 0.1  # Small text in large budget

    def test_inclusion_rate(self) -> None:
        assembler = PriorityAssembler(budget_tokens=50)
        for i in range(10):
            assembler.add(ContextItem(
                content=f"Item {i} with some padding text to consume tokens."
            ))
        result = assembler.assemble()
        assert 0.0 < result.inclusion_rate < 1.0
