"""Context Engineering Toolkit — optimize LLM context windows.

This package provides tools for token counting, context compression,
priority assembly, benchmarking, and named context engineering strategies.
"""

from __future__ import annotations

__version__ = "0.2.0"

from src.context_engineering_toolkit.context_manager import ContextEngineeringToolkit

__all__ = ["ContextEngineeringToolkit", "__version__"]
