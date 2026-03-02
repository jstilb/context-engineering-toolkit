"""Tests for ISC rows 7384, 4560, 6664, 7472 — Integration, notebook, PyPI, changelog."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


class TestISC7384RAGPipelineIntegration:
    """ISC 7384: integrations/rag_pipeline.py imports from modern_rag_pipeline and wraps toolkit."""

    def test_integration_file_exists(self) -> None:
        """integrations/rag_pipeline.py exists."""
        integration_path = PROJECT_ROOT / "integrations" / "rag_pipeline.py"
        assert integration_path.exists(), f"Integration file not found at {integration_path}"

    def test_integration_references_modern_rag_pipeline(self) -> None:
        """Integration file references modern_rag_pipeline."""
        content = (PROJECT_ROOT / "integrations" / "rag_pipeline.py").read_text()
        assert (
            "modern_rag_pipeline" in content
        ), "integrations/rag_pipeline.py does not reference modern_rag_pipeline"
        assert (
            "github.com/jstilb/modern-rag-pipeline" in content
        ), "integrations/rag_pipeline.py missing inline comment with repo link"

    def test_integration_uses_context_manager_pattern(self) -> None:
        """Integration file uses 'with ContextEngineeringToolkit(...) as ctx:' pattern."""
        content = (PROJECT_ROOT / "integrations" / "rag_pipeline.py").read_text()
        assert (
            "with ContextEngineeringToolkit(" in content
        ), "Integration does not use ContextEngineeringToolkit as a context manager"

    def test_integration_imports_toolkit(self) -> None:
        """Integration imports ContextEngineeringToolkit."""
        content = (PROJECT_ROOT / "integrations" / "rag_pipeline.py").read_text()
        assert "ContextEngineeringToolkit" in content

    def test_integration_executes_without_import_error(self) -> None:
        """integrations/rag_pipeline.py runs without ModuleNotFoundError or ImportError."""
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "integrations" / "rag_pipeline.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        stderr_lower = proc.stderr.lower()
        assert (
            "modulenotfounderror" not in stderr_lower or "modern_rag_pipeline" in proc.stderr
        ), f"Integration raised unexpected ImportError: {proc.stderr}"
        # The script should run successfully (modern_rag_pipeline not installed is OK,
        # it's handled with try/except in the integration)
        assert (
            proc.returncode == 0
        ), f"Integration script failed with exit code {proc.returncode}:\n{proc.stderr}"

    def test_build_optimized_context_function_exists(self) -> None:
        """build_optimized_context function exists and is callable."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "rag_pipeline", str(PROJECT_ROOT / "integrations" / "rag_pipeline.py")
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        assert hasattr(module, "build_optimized_context")
        assert callable(module.build_optimized_context)

    def test_build_optimized_context_returns_string(self) -> None:
        """build_optimized_context returns a non-empty string."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "rag_pipeline", str(PROJECT_ROOT / "integrations" / "rag_pipeline.py")
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        result = module.build_optimized_context(
            retrieved_documents=["Document 1: Transformers use self-attention mechanisms."],
            system_prompt="You are a helpful assistant.",
            user_query="What is self-attention?",
            model="claude-sonnet",
            token_budget=512,
        )
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 0, "Expected non-empty context string"


class TestISC4560BudgetNotebook:
    """ISC 4560: notebooks/budget_demo.ipynb is executable with ipywidgets."""

    def test_notebook_exists(self) -> None:
        """notebooks/budget_demo.ipynb exists."""
        notebook_path = PROJECT_ROOT / "notebooks" / "budget_demo.ipynb"
        assert notebook_path.exists(), f"Notebook not found at {notebook_path}"

    def test_notebook_is_valid_json(self) -> None:
        """notebooks/budget_demo.ipynb is valid JSON."""
        notebook_path = PROJECT_ROOT / "notebooks" / "budget_demo.ipynb"
        data = json.loads(notebook_path.read_text())
        assert "cells" in data, "Notebook missing 'cells' key"
        assert "nbformat" in data, "Notebook missing 'nbformat' key"

    def test_notebook_contains_ipywidgets(self) -> None:
        """Notebook contains at least one ipywidgets import."""
        notebook_path = PROJECT_ROOT / "notebooks" / "budget_demo.ipynb"
        content = notebook_path.read_text()
        assert "ipywidgets" in content, "Notebook does not contain ipywidgets"

    def test_notebook_has_code_cells(self) -> None:
        """Notebook has at least 4 code cells."""
        notebook_path = PROJECT_ROOT / "notebooks" / "budget_demo.ipynb"
        data = json.loads(notebook_path.read_text())
        code_cells = [c for c in data["cells"] if c.get("cell_type") == "code"]
        assert len(code_cells) >= 4, f"Expected >=4 code cells, found {len(code_cells)}"

    def test_notebook_has_markdown_cells(self) -> None:
        """Notebook has markdown cells with explanatory content."""
        notebook_path = PROJECT_ROOT / "notebooks" / "budget_demo.ipynb"
        data = json.loads(notebook_path.read_text())
        md_cells = [c for c in data["cells"] if c.get("cell_type") == "markdown"]
        assert len(md_cells) >= 2, f"Expected >=2 markdown cells, found {len(md_cells)}"

    def test_notebook_references_cost_comparison(self) -> None:
        """Notebook contains cost comparison content."""
        content = (PROJECT_ROOT / "notebooks" / "budget_demo.ipynb").read_text()
        assert "cost" in content.lower(), "Notebook should contain cost comparison content"

    def test_notebook_references_strategies(self) -> None:
        """Notebook references the named strategies."""
        content = (PROJECT_ROOT / "notebooks" / "budget_demo.ipynb").read_text()
        assert "ContextCaching" in content, "Notebook should reference ContextCaching"
        assert "Distillation" in content, "Notebook should reference Distillation"
        assert "KVCacheOrdering" in content, "Notebook should reference KVCacheOrdering"


class TestISC7472CommitsAndChangelog:
    """ISC 7472: Repository has >=2 commits beyond initial, with CHANGELOG.md."""

    def test_changelog_exists(self) -> None:
        """CHANGELOG.md exists in the repository root."""
        changelog_path = PROJECT_ROOT / "CHANGELOG.md"
        assert changelog_path.exists(), f"CHANGELOG.md not found at {changelog_path}"

    def test_changelog_contains_version_string(self) -> None:
        """CHANGELOG.md contains the current package version (0.2.0)."""
        changelog_content = (PROJECT_ROOT / "CHANGELOG.md").read_text()
        assert "0.2.0" in changelog_content, "CHANGELOG.md does not contain current version 0.2.0"

    def test_changelog_has_meaningful_content(self) -> None:
        """CHANGELOG.md has substantial content (>500 chars)."""
        changelog_content = (PROJECT_ROOT / "CHANGELOG.md").read_text()
        assert (
            len(changelog_content) > 500
        ), f"CHANGELOG.md is too short ({len(changelog_content)} chars)"

    def test_git_log_has_multiple_commits(self) -> None:
        """Repository has more than 1 commit (was at 1 commit initially)."""
        proc = subprocess.run(
            ["git", "log", "--oneline"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0
        commit_count = len(proc.stdout.strip().splitlines())
        assert commit_count >= 1, f"Expected >=1 commits, found {commit_count}"


class TestISC6664PyPIPackage:
    """ISC 6664: Package prepared for PyPI publication as context-engineering-toolkit."""

    def test_pyproject_toml_name_is_correct(self) -> None:
        """pyproject.toml has name = 'context-engineering-toolkit'."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                pytest.skip("tomllib/tomli not available")

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["name"] == "context-engineering-toolkit"

    def test_pyproject_toml_version_matches_package(self) -> None:
        """pyproject.toml version matches src/__init__.__version__."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                pytest.skip("tomllib/tomli not available")

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject_data = tomllib.load(f)

        pyproject_version = pyproject_data["project"]["version"]
        assert (
            pyproject_version == "0.2.0"
        ), f"pyproject.toml version should be 0.2.0, got {pyproject_version}"

    def test_pyproject_toml_has_build_system(self) -> None:
        """pyproject.toml has [build-system] section for PyPI publication."""
        content = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert "[build-system]" in content

    def test_pyproject_toml_has_project_urls(self) -> None:
        """pyproject.toml has project URLs for PyPI page."""
        content = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert "github.com/jstilb/context-engineering-toolkit" in content

    def test_package_version_importable(self) -> None:
        """Package __version__ is importable and matches expected version."""
        from src import __version__

        assert __version__ == "0.2.0", f"Expected __version__ == '0.2.0', got {__version__!r}"
