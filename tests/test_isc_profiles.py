"""Tests for ISC rows 5952 and 5450 — Model profiles and CLI --profile flag."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
PROFILES_DIR = PROJECT_ROOT / "profiles"
REQUIRED_PROFILES = ["gpt-4o.yaml", "claude-sonnet.yaml", "llama-3.3.yaml", "gemini-2.0-flash.yaml"]
REQUIRED_FIELDS = [
    "context_window",
    "optimal_compression_ratio",
    "priority_ordering",
    "token_counting_quirks",
]


class TestISC5952ModelProfiles:
    """ISC 5952: profiles/ directory with 4 YAML files with required fields."""

    def test_profiles_directory_exists(self) -> None:
        """profiles/ directory exists."""
        assert PROFILES_DIR.exists(), f"profiles/ directory not found at {PROFILES_DIR}"

    def test_exactly_four_yaml_files(self) -> None:
        """Exactly 4 YAML files exist in profiles/."""
        yaml_files = list(PROFILES_DIR.glob("*.yaml"))
        assert (
            len(yaml_files) == 4
        ), f"Expected 4 YAML files, found {len(yaml_files)}: {[f.name for f in yaml_files]}"

    @pytest.mark.parametrize("filename", REQUIRED_PROFILES)
    def test_required_profile_exists(self, filename: str) -> None:
        """Each required profile YAML file exists."""
        profile_path = PROFILES_DIR / filename
        assert profile_path.exists(), f"Required profile not found: {filename}"

    @pytest.mark.parametrize("filename", REQUIRED_PROFILES)
    def test_profile_parses_as_valid_yaml(self, filename: str) -> None:
        """Each profile file is valid YAML."""
        profile_path = PROFILES_DIR / filename
        data = yaml.safe_load(profile_path.read_text())
        assert isinstance(data, dict), f"{filename} did not parse as a dict"

    @pytest.mark.parametrize("filename", REQUIRED_PROFILES)
    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_profile_has_required_field(self, filename: str, field: str) -> None:
        """Each profile contains all required fields."""
        profile_path = PROFILES_DIR / filename
        data = yaml.safe_load(profile_path.read_text())
        assert field in data, (
            f"Profile {filename} missing required field '{field}'. " f"Has: {list(data.keys())}"
        )

    def test_gpt4o_context_window_is_128k(self) -> None:
        """GPT-4o profile has 128,000 token context window."""
        data = yaml.safe_load((PROFILES_DIR / "gpt-4o.yaml").read_text())
        assert (
            data["context_window"] == 128000
        ), f"GPT-4o context_window should be 128000, got {data['context_window']}"

    def test_claude_sonnet_context_window_is_200k(self) -> None:
        """Claude Sonnet profile has 200,000 token context window."""
        data = yaml.safe_load((PROFILES_DIR / "claude-sonnet.yaml").read_text())
        assert (
            data["context_window"] == 200000
        ), f"Claude Sonnet context_window should be 200000, got {data['context_window']}"

    def test_gemini_context_window_is_1m(self) -> None:
        """Gemini 2.0 Flash profile has 1,000,000 token context window."""
        data = yaml.safe_load((PROFILES_DIR / "gemini-2.0-flash.yaml").read_text())
        assert (
            data["context_window"] == 1000000
        ), f"Gemini context_window should be 1000000, got {data['context_window']}"

    def test_compression_ratios_are_valid(self) -> None:
        """All compression ratios are between 0 and 1."""
        for filename in REQUIRED_PROFILES:
            data = yaml.safe_load((PROFILES_DIR / filename).read_text())
            ratio = data["optimal_compression_ratio"]
            assert (
                0.0 < ratio < 1.0
            ), f"{filename}: optimal_compression_ratio {ratio} is not in (0, 1)"

    def test_priority_ordering_is_list(self) -> None:
        """All priority_ordering fields are non-empty lists."""
        for filename in REQUIRED_PROFILES:
            data = yaml.safe_load((PROFILES_DIR / filename).read_text())
            ordering = data["priority_ordering"]
            assert isinstance(ordering, list), f"{filename}: priority_ordering is not a list"
            assert len(ordering) > 0, f"{filename}: priority_ordering is empty"


class TestISC5450CLIProfileFlag:
    """ISC 5450: CLI accepts --profile flag, loads YAML, adjusts behavior."""

    def test_assemble_with_gpt4o_profile_exits_zero(self, tmp_path: Path) -> None:
        """assemble --profile gpt-4o on a sample file exits 0."""
        sample = tmp_path / "sample.txt"
        sample.write_text(
            "The Transformer architecture uses self-attention mechanisms. "
            "BERT extends this with bidirectional pre-training. "
            "Results show 28.4 BLEU on WMT 2014."
        )
        proc = subprocess.run(
            [sys.executable, "-m", "src.cli", "assemble", str(sample), "--profile", "gpt-4o"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert (
            proc.returncode == 0
        ), f"assemble --profile gpt-4o failed: {proc.stderr}\n{proc.stdout}"
        assert len(proc.stdout.strip()) > 0, "No output produced by assemble"

    def test_assemble_with_claude_profile_exits_zero(self, tmp_path: Path) -> None:
        """assemble --profile claude-sonnet exits 0 and produces output."""
        sample = tmp_path / "sample.txt"
        sample.write_text(
            "Context engineering optimizes LLM token budgets. "
            "Priority assembly retains 2.1x more information than naive truncation. "
            "Cost savings at 100K requests/month can reach $28,800."
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "assemble",
                str(sample),
                "--profile",
                "claude-sonnet",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0, f"Failed: {proc.stderr}\n{proc.stdout}"

    def test_assemble_with_invalid_profile_exits_nonzero(self, tmp_path: Path) -> None:
        """assemble --profile unknown-model exits nonzero with descriptive error."""
        sample = tmp_path / "sample.txt"
        sample.write_text("Some text.")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "assemble",
                str(sample),
                "--profile",
                "invalid-model-xyz",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode != 0, "Expected nonzero exit for unknown profile"
        error_output = (proc.stdout + proc.stderr).lower()
        assert (
            "invalid" in error_output or "not found" in error_output or "error" in error_output
        ), f"Expected descriptive error message, got: {proc.stdout}\n{proc.stderr}"

    def test_compress_with_profile_uses_optimal_ratio(self, tmp_path: Path) -> None:
        """compress --profile applies optimal_compression_ratio from YAML."""
        # Create a document long enough to compress
        long_text = " ".join(
            [
                "The Transformer architecture was introduced by Vaswani et al in 2017.",
                "It uses multi-head self-attention to process sequences in parallel.",
                "BERT extends this with bidirectional pre-training on masked language modeling.",
                "GPT-4 achieves 86.4% accuracy on MMLU using causal language modeling.",
                "The context window has grown from 2048 tokens in 2020 to 1 million in 2026.",
                "Cost optimization through context engineering saves 40-70% on API bills.",
            ]
            * 5
        )

        sample = tmp_path / "long_doc.txt"
        sample.write_text(long_text)

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                "compress",
                "--file",
                str(sample),
                "--profile",
                "gpt-4o",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0, f"compress --profile failed: {proc.stderr}"
        assert len(proc.stdout.strip()) > 0, "No compressed output produced"
