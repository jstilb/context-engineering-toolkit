"""Tests for ISC row 2288 — Cost savings calculator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


class TestISC2288CostCalculator:
    """ISC 2288: Cost calculator accepts volume + token count, outputs savings metrics."""

    def test_cost_command_exits_zero(self) -> None:
        """cost command exits 0 with valid inputs."""
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "cost",
                "--volume",
                "100000",
                "--tokens-per-doc",
                "8000",
                "--profile",
                "claude-sonnet",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0, f"cost command failed: {proc.stderr}\n{proc.stdout}"

    def test_cost_command_json_output_has_required_fields(self) -> None:
        """cost --json-output includes tokens_saved_per_request, monthly costs, and ROI."""
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "cost",
                "--volume",
                "100000",
                "--tokens-per-doc",
                "8000",
                "--profile",
                "claude-sonnet",
                "--json-output",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0, f"cost failed: {proc.stderr}"
        data = json.loads(proc.stdout)

        assert "tokens_saved_per_request" in data, "Missing tokens_saved_per_request"
        assert "monthly_naive_cost_usd" in data, "Missing monthly_naive_cost_usd"
        assert "monthly_optimized_cost_usd" in data, "Missing monthly_optimized_cost_usd"
        assert "roi_percentage" in data, "Missing roi_percentage"

    def test_cost_tokens_saved_is_positive(self) -> None:
        """tokens_saved_per_request is positive (optimized < naive)."""
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "cost",
                "--volume",
                "50000",
                "--tokens-per-doc",
                "5000",
                "--profile",
                "gpt-4o",
                "--json-output",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert (
            data["tokens_saved_per_request"] > 0
        ), f"tokens_saved_per_request should be positive, got {data['tokens_saved_per_request']}"

    def test_cost_optimized_less_than_naive(self) -> None:
        """Optimized monthly cost is strictly less than naive cost."""
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "cost",
                "--volume",
                "100000",
                "--tokens-per-doc",
                "8000",
                "--profile",
                "claude-sonnet",
                "--json-output",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert (
            data["monthly_optimized_cost_usd"] < data["monthly_naive_cost_usd"]
        ), "Optimized cost should be less than naive cost"

    def test_cost_roi_is_positive(self) -> None:
        """ROI percentage is positive."""
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "cost",
                "--volume",
                "200000",
                "--tokens-per-doc",
                "10000",
                "--profile",
                "claude-sonnet",
                "--json-output",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["roi_percentage"] > 0.0, f"ROI should be positive, got {data['roi_percentage']}"

    def test_cost_pricing_references_2026(self) -> None:
        """JSON output references 2026 pricing source."""
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "cost",
                "--volume",
                "100000",
                "--tokens-per-doc",
                "8000",
                "--profile",
                "claude-sonnet",
                "--json-output",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        pricing_source = data.get("model_pricing_source", "")
        assert (
            "2026" in pricing_source
        ), f"Pricing source should reference 2026, got: {pricing_source}"

    def test_cost_gemini_profile_works(self) -> None:
        """cost command works with gemini-2.0-flash profile."""
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "cost",
                "--volume",
                "1000000",
                "--tokens-per-doc",
                "50000",
                "--profile",
                "gemini-2.0-flash",
                "--json-output",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0, f"gemini cost failed: {proc.stderr}"
        data = json.loads(proc.stdout)
        assert data["monthly_naive_cost_usd"] > 0

    def test_cost_math_is_correct(self) -> None:
        """Verify cost calculations are mathematically correct."""
        volume = 100000
        tokens_per_doc = 8000
        # claude-sonnet: $3.00/million input tokens, 35% compression ratio
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "cost",
                "--volume",
                str(volume),
                "--tokens-per-doc",
                str(tokens_per_doc),
                "--profile",
                "claude-sonnet",
                "--json-output",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0
        data = json.loads(proc.stdout)

        # Verify tokens_saved is consistent with compression_ratio
        compression_ratio = data["inputs"]["compression_ratio"]
        expected_saved = tokens_per_doc - int(tokens_per_doc * compression_ratio)
        assert data["tokens_saved_per_request"] == expected_saved, (
            f"tokens_saved_per_request calculation wrong: "
            f"expected {expected_saved}, got {data['tokens_saved_per_request']}"
        )
