#!/usr/bin/env python3
"""
Verify that benchmark results meet the headline requirement:
priority assembly key-term retention ratio >= 2.1x vs naive truncation.

Usage:
    python benchmarks/verify_headline.py results.json

Exit codes:
    0 — Headline stat confirmed (ratio >= 2.1x)
    1 — Headline stat not met (ratio < 2.1x)
    2 — Invalid or missing results file
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MINIMUM_RATIO = 2.1


def verify(results_path: str) -> int:
    """Verify headline stat from benchmark results file.

    Returns:
        0 if ratio >= 2.1x, 1 otherwise, 2 on file/parse error.
    """
    path = Path(results_path)
    if not path.exists():
        print(f"ERROR: Results file not found: {results_path}", file=sys.stderr)
        return 2

    try:
        results = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in results file: {e}", file=sys.stderr)
        return 2

    headline = results.get("headline_stat", {})
    ratio = headline.get("priority_vs_naive_key_term_retention_ratio")

    if ratio is None:
        print("ERROR: 'headline_stat.priority_vs_naive_key_term_retention_ratio' not found in results", file=sys.stderr)
        return 2

    print(f"Priority vs Naive key-term retention ratio: {ratio:.4f}x")
    print(f"Required minimum: {MINIMUM_RATIO:.1f}x")
    print(f"Document count: {results.get('document_count', '?')}")
    print(f"Categories: {', '.join(results.get('categories', []))}")
    print(f"Description: {headline.get('description', '')}")

    if ratio >= MINIMUM_RATIO:
        print(f"\nCONFIRMED: Headline stat met ({ratio:.2f}x >= {MINIMUM_RATIO:.1f}x)")
        return 0
    else:
        print(f"\nFAILED: Headline stat not met ({ratio:.2f}x < {MINIMUM_RATIO:.1f}x)")
        return 1


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results.json>", file=sys.stderr)
        sys.exit(2)

    exit_code = verify(sys.argv[1])
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
