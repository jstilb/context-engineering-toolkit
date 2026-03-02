"""Token-aware smart truncation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.tokens.counter import ModelFamily, TokenCounter


class TruncationStrategy(Enum):
    """Where to truncate content."""

    HEAD = "head"  # Keep beginning, cut end
    TAIL = "tail"  # Keep end, cut beginning
    MIDDLE = "middle"  # Keep beginning and end, cut middle


@dataclass(frozen=True)
class TruncationResult:
    """Result of a truncation operation."""

    text: str
    original_tokens: int
    truncated_tokens: int
    strategy: TruncationStrategy
    was_truncated: bool

    @property
    def compression_ratio(self) -> float:
        """Ratio of truncated to original tokens."""
        if self.original_tokens == 0:
            return 1.0
        return self.truncated_tokens / self.original_tokens


class SmartTruncator:
    """Token-aware truncation that respects token and sentence boundaries.

    Unlike character-level truncation, this ensures:
    1. Token boundaries are respected (no split tokens)
    2. Sentence boundaries are preferred (no mid-sentence cuts)
    3. Multiple strategies (head, tail, middle-out)

    Example:
        truncator = SmartTruncator(model=ModelFamily.GPT4O)
        result = truncator.truncate(long_text, max_tokens=500)
        print(f"Kept {result.truncated_tokens} of {result.original_tokens} tokens")
    """

    ELLIPSIS = " [...] "

    def __init__(self, model: ModelFamily = ModelFamily.GPT4O) -> None:
        self._counter = TokenCounter(model)

    def truncate(
        self,
        text: str,
        max_tokens: int,
        strategy: TruncationStrategy = TruncationStrategy.HEAD,
    ) -> TruncationResult:
        """Truncate text to fit within max_tokens.

        Args:
            text: Input text.
            max_tokens: Maximum tokens in output.
            strategy: Where to cut (head, tail, middle).

        Returns:
            TruncationResult with truncated text and metadata.
        """
        original_count = self._counter.count(text).token_count

        if original_count <= max_tokens:
            return TruncationResult(
                text=text,
                original_tokens=original_count,
                truncated_tokens=original_count,
                strategy=strategy,
                was_truncated=False,
            )

        if strategy == TruncationStrategy.HEAD:
            truncated = self._truncate_head(text, max_tokens)
        elif strategy == TruncationStrategy.TAIL:
            truncated = self._truncate_tail(text, max_tokens)
        else:
            truncated = self._truncate_middle(text, max_tokens)

        truncated_count = self._counter.count(truncated).token_count

        return TruncationResult(
            text=truncated,
            original_tokens=original_count,
            truncated_tokens=truncated_count,
            strategy=strategy,
            was_truncated=True,
        )

    def _truncate_head(self, text: str, max_tokens: int) -> str:
        """Keep the beginning of text."""
        # Reserve tokens for ellipsis
        ellipsis_tokens = self._counter.count(self.ELLIPSIS).token_count
        target = max_tokens - ellipsis_tokens

        if target <= 0:
            return self._counter.truncate_to_tokens(text, max_tokens)

        truncated = self._counter.truncate_to_tokens(text, target)

        # Try to end at a sentence boundary
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")
        boundary = max(last_period, last_newline)

        if boundary > len(truncated) * 0.5:  # Only if we keep > 50% of content
            truncated = truncated[: boundary + 1]

        return truncated + self.ELLIPSIS

    def _truncate_tail(self, text: str, max_tokens: int) -> str:
        """Keep the end of text."""
        ellipsis_tokens = self._counter.count(self.ELLIPSIS).token_count
        target = max_tokens - ellipsis_tokens

        if target <= 0:
            # Just take the last max_tokens
            tokens = self._counter.encode(text)
            return self._counter.decode(tokens[-max_tokens:])

        # Encode, take last N tokens, decode
        tokens = self._counter.encode(text)
        tail_tokens = tokens[-target:]
        tail_text = self._counter.decode(tail_tokens)

        # Try to start at a sentence boundary
        first_period = tail_text.find(".")
        first_newline = tail_text.find("\n")
        candidates = [b for b in [first_period, first_newline] if b >= 0]

        if candidates:
            boundary = min(candidates)
            if boundary < len(tail_text) * 0.3:  # Only if we skip < 30%
                tail_text = tail_text[boundary + 1 :].lstrip()

        return self.ELLIPSIS + tail_text

    def _truncate_middle(self, text: str, max_tokens: int) -> str:
        """Keep beginning and end, cut the middle."""
        ellipsis_tokens = self._counter.count(self.ELLIPSIS).token_count
        target = max_tokens - ellipsis_tokens

        if target <= 0:
            return self._counter.truncate_to_tokens(text, max_tokens)

        # Split budget: 60% head, 40% tail (beginning is usually more important)
        head_budget = int(target * 0.6)
        tail_budget = target - head_budget

        head = self._counter.truncate_to_tokens(text, head_budget)
        tokens = self._counter.encode(text)
        tail_tokens = tokens[-tail_budget:]
        tail = self._counter.decode(tail_tokens)

        return head + self.ELLIPSIS + tail
