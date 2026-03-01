"""Tests for the CLI interface."""

import json
import os
import tempfile

import pytest
from click.testing import CliRunner

from src.cli import main


class TestCLI:
    """Tests for CLI commands."""

    def setup_method(self) -> None:
        self.runner = CliRunner()

    def test_version(self) -> None:
        result = self.runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.2.0" in result.output

    def test_count_text_argument(self) -> None:
        result = self.runner.invoke(main, ["count", "Hello, world!"])
        assert result.exit_code == 0
        assert "Tokens:" in result.output
        assert "Model:" in result.output

    def test_count_json_output(self) -> None:
        result = self.runner.invoke(main, ["count", "Hello, world!", "-j"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "token_count" in data
        assert "model" in data
        assert data["token_count"] > 0

    def test_count_from_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test text for file reading.")
            f.flush()
            result = self.runner.invoke(main, ["count", "--file", f.name])
        os.unlink(f.name)
        assert result.exit_code == 0
        assert "Tokens:" in result.output

    def test_count_different_model(self) -> None:
        result = self.runner.invoke(main, ["count", "Test", "--model", "claude"])
        assert result.exit_code == 0
        assert "claude" in result.output

    def test_count_invalid_model(self) -> None:
        result = self.runner.invoke(main, ["count", "Test", "--model", "invalid"])
        assert result.exit_code != 0

    def test_compress_extractive(self) -> None:
        long_text = "This is sentence one. This is sentence two. This is sentence three. " * 10
        result = self.runner.invoke(
            main, ["compress", long_text, "--target-tokens", "20", "--method", "extractive"]
        )
        assert result.exit_code == 0
        assert "Original:" in result.stderr_bytes.decode() if result.stderr_bytes else True

    def test_compress_truncate(self) -> None:
        long_text = "This is a test sentence. " * 50
        result = self.runner.invoke(
            main, ["compress", long_text, "--target-tokens", "20", "--method", "truncate"]
        )
        assert result.exit_code == 0

    def test_compress_from_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Long text for compression testing. " * 20)
            f.flush()
            result = self.runner.invoke(
                main, ["compress", "--file", f.name, "--target-tokens", "30"]
            )
        os.unlink(f.name)
        assert result.exit_code == 0

    def test_benchmark_command(self) -> None:
        original_text = (
            "The Transformer architecture was introduced in 2017. "
            "GPT-4 achieves 86.4% on MMLU. OpenAI released it in March 2023."
        )
        compressed_text = "The Transformer architecture was introduced in 2017."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as orig:
            orig.write(original_text)
            orig.flush()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as comp:
                comp.write(compressed_text)
                comp.flush()
                result = self.runner.invoke(
                    main, ["benchmark", orig.name, comp.name]
                )
        os.unlink(orig.name)
        os.unlink(comp.name)
        assert result.exit_code == 0
        assert "Overall retention:" in result.output

    def test_benchmark_json_output(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as orig:
            orig.write("Original text with numbers 42 and names like OpenAI.")
            orig.flush()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as comp:
                comp.write("Text with 42 and OpenAI.")
                comp.flush()
                result = self.runner.invoke(
                    main, ["benchmark", orig.name, comp.name, "-j"]
                )
        os.unlink(orig.name)
        os.unlink(comp.name)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "overall_score" in data

    def test_demo_command(self) -> None:
        result = self.runner.invoke(main, ["demo"])
        assert result.exit_code == 0
        assert "Context Engineering Toolkit - Demo" in result.output
        assert "Token Counting" in result.output
        assert "Extractive Compression" in result.output
        assert "Retention Benchmark" in result.output
        assert "Demo complete" in result.output

    def test_count_from_stdin(self) -> None:
        result = self.runner.invoke(main, ["count"], input="Hello from stdin!")
        assert result.exit_code == 0
        assert "Tokens:" in result.output
