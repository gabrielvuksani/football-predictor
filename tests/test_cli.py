"""Integration tests — CLI command smoke tests.

Tests that CLI commands can be invoked without crashing.
Uses Typer's CliRunner for in-process testing (no subprocess needed).
"""
import pytest
from typer.testing import CliRunner

from footy.cli import app

runner = CliRunner()


class TestCliHelp:
    """Every command must respond to --help without error."""

    _COMMANDS = [
        "go", "refresh", "matchday", "nuke",
        "ai-preview", "ai-value", "ai-review",
        "serve", "ingest", "train", "predict", "metrics",
        "update", "ingest-history", "reset-states", "news",
        "backtest", "train-meta", "ingest-extras",
        "ingest-fixtures-odds", "compute-h2h", "compute-xg",
        "update-odds", "cache-stats", "cache-cleanup",
    ]

    @pytest.mark.parametrize("cmd", _COMMANDS)
    def test_help(self, cmd):
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, f"'{cmd} --help' failed: {result.output}"
        assert cmd.replace("-", "_") in result.output.lower() or "usage" in result.output.lower() \
            or "--help" in result.output.lower() or result.output.strip() != ""
