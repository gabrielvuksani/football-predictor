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

    # Root-level commands
    _ROOT_COMMANDS = [
        "go", "refresh", "matchday", "nuke", "serve", "update", "self-test",
    ]
    # Sub-group commands: (group, sub-command)
    _GROUP_COMMANDS = [
        ("data", "ingest"), ("data", "history"), ("data", "extras"),
        ("data", "fixtures-odds"), ("data", "odds"), ("data", "news"),
        ("data", "h2h"), ("data", "xg"), ("data", "reset"),
        ("data", "cache-stats"), ("data", "cache-cleanup"),
        ("model", "train"), ("model", "predict"), ("model", "metrics"),
        ("model", "backtest"), ("model", "train-meta"),
        ("ai", "preview"), ("ai", "value"), ("ai", "review"),
    ]

    @pytest.mark.parametrize("cmd", _ROOT_COMMANDS)
    def test_root_help(self, cmd):
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, f"'{cmd} --help' failed: {result.output}"

    @pytest.mark.parametrize("group,cmd", _GROUP_COMMANDS)
    def test_group_help(self, group, cmd):
        result = runner.invoke(app, [group, cmd, "--help"])
        assert result.exit_code == 0, f"'{group} {cmd} --help' failed: {result.output}"
