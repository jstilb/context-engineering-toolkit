"""Tests for ISC rows 1192 and 6160 — Benchmark suite."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
DOCUMENTS_DIR = BENCHMARKS_DIR / "documents"


class TestISC1192BenchmarkDocuments:
    """ISC 1192: benchmarks/ directory with 10 real-world documents across >=3 categories."""

    def test_ten_documents_exist(self) -> None:
        """Exactly 10 benchmark documents exist in benchmarks/documents/."""
        assert DOCUMENTS_DIR.exists(), f"benchmarks/documents/ not found at {DOCUMENTS_DIR}"
        docs = sorted(DOCUMENTS_DIR.glob("*.txt"))
        assert (
            len(docs) == 10
        ), f"Expected 10 documents, found {len(docs)}: {[d.name for d in docs]}"

    def test_three_categories_present(self) -> None:
        """Documents span at least 3 categories (paper, news, code)."""
        docs = list(DOCUMENTS_DIR.glob("*.txt"))
        prefixes = {doc.name.split("_")[0] for doc in docs}
        # Expect: paper, news, code
        assert "paper" in prefixes, "No academic papers found (paper_*.txt)"
        assert "news" in prefixes, "No news articles found (news_*.txt)"
        assert "code" in prefixes, "No code files found (code_*.txt)"
        assert len(prefixes) >= 3, f"Only {len(prefixes)} categories: {prefixes}"

    def test_paper_category_count(self) -> None:
        """At least 2 academic papers are present."""
        papers = list(DOCUMENTS_DIR.glob("paper_*.txt"))
        assert len(papers) >= 2, f"Expected >=2 papers, found {len(papers)}"

    def test_news_category_count(self) -> None:
        """At least 2 news articles are present."""
        news = list(DOCUMENTS_DIR.glob("news_*.txt"))
        assert len(news) >= 2, f"Expected >=2 news articles, found {len(news)}"

    def test_code_category_count(self) -> None:
        """At least 1 code file is present."""
        code = list(DOCUMENTS_DIR.glob("code_*.txt"))
        assert len(code) >= 1, f"Expected >=1 code files, found {len(code)}"

    def test_documents_are_not_empty(self) -> None:
        """All benchmark documents have substantial content (>500 chars)."""
        for doc in DOCUMENTS_DIR.glob("*.txt"):
            content = doc.read_text()
            assert len(content) > 500, f"{doc.name} is too short ({len(content)} chars)"

    def test_benchmark_runner_script_exists(self) -> None:
        """benchmarks/run_benchmark.py script exists."""
        assert (BENCHMARKS_DIR / "run_benchmark.py").exists()

    def test_verify_headline_script_exists(self) -> None:
        """benchmarks/verify_headline.py script exists."""
        assert (BENCHMARKS_DIR / "verify_headline.py").exists()


class TestISC6160BenchmarkHeadlineStat:
    """ISC 6160: Benchmark produces >=2.1x priority assembly vs naive truncation ratio."""

    def test_benchmark_runs_without_error(self, tmp_path: Path) -> None:
        """run_benchmark.py executes successfully and writes valid JSON output."""
        result_file = tmp_path / "results.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(BENCHMARKS_DIR / "run_benchmark.py"),
                "--output",
                str(result_file),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Benchmark failed: {proc.stderr}"
        assert result_file.exists(), "Output file was not created"

    def test_headline_ratio_meets_requirement(self, tmp_path: Path) -> None:
        """Priority assembly key-term retention ratio is >= 2.1x vs naive truncation."""
        result_file = tmp_path / "results.json"
        subprocess.run(
            [
                sys.executable,
                str(BENCHMARKS_DIR / "run_benchmark.py"),
                "--output",
                str(result_file),
            ],
            capture_output=True,
        )
        results = json.loads(result_file.read_text())
        ratio = results["headline_stat"]["priority_vs_naive_key_term_retention_ratio"]
        assert ratio >= 2.1, (
            f"Headline ratio {ratio:.2f}x is below required 2.1x minimum. "
            f"Avg naive: {results['headline_stat']['avg_naive_key_term_retention']:.1%}, "
            f"Avg priority: {results['headline_stat']['avg_priority_key_term_retention']:.1%}"
        )

    def test_benchmark_covers_all_10_documents(self, tmp_path: Path) -> None:
        """Benchmark results cover exactly 10 documents."""
        result_file = tmp_path / "results.json"
        subprocess.run(
            [
                sys.executable,
                str(BENCHMARKS_DIR / "run_benchmark.py"),
                "--output",
                str(result_file),
            ],
            capture_output=True,
        )
        results = json.loads(result_file.read_text())
        assert results["document_count"] == 10
        assert len(results["documents"]) == 10

    def test_benchmark_covers_all_3_categories(self, tmp_path: Path) -> None:
        """Benchmark results span >=3 document categories."""
        result_file = tmp_path / "results.json"
        subprocess.run(
            [
                sys.executable,
                str(BENCHMARKS_DIR / "run_benchmark.py"),
                "--output",
                str(result_file),
            ],
            capture_output=True,
        )
        results = json.loads(result_file.read_text())
        assert len(results["categories"]) >= 3

    def test_verify_headline_exits_zero_on_passing_results(self, tmp_path: Path) -> None:
        """verify_headline.py exits 0 when ratio >= 2.1x."""
        result_file = tmp_path / "results.json"
        subprocess.run(
            [
                sys.executable,
                str(BENCHMARKS_DIR / "run_benchmark.py"),
                "--output",
                str(result_file),
            ],
            capture_output=True,
        )
        proc = subprocess.run(
            [sys.executable, str(BENCHMARKS_DIR / "verify_headline.py"), str(result_file)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"verify_headline.py failed: {proc.stdout}\n{proc.stderr}"
        assert "CONFIRMED" in proc.stdout

    def test_each_document_has_three_method_scores(self, tmp_path: Path) -> None:
        """Each benchmark document has scores for all 3 methods and 3 metrics."""
        result_file = tmp_path / "results.json"
        subprocess.run(
            [
                sys.executable,
                str(BENCHMARKS_DIR / "run_benchmark.py"),
                "--output",
                str(result_file),
            ],
            capture_output=True,
        )
        results = json.loads(result_file.read_text())
        required_methods = {"naive_truncation", "extractive_compression", "priority_assembly"}
        required_metrics = {
            "key_term_retention",
            "entity_retention",
            "sentence_coverage",
            "overall_score",
        }

        for doc in results["documents"]:
            methods = set(doc["methods"].keys())
            assert methods == required_methods, f"Missing methods in {doc['filename']}: {methods}"
            for method, scores in doc["methods"].items():
                metric_keys = set(scores.keys())
                assert required_metrics.issubset(
                    metric_keys
                ), f"Missing metrics for {method} in {doc['filename']}"
