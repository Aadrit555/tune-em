"""Unit tests for CLI commands."""

from typer.testing import CliRunner
from anydecision.cli.main import app

runner = CliRunner()


def test_cli_version():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "anydecision / tune-em" in res.output


def test_cli_ask():
    res = runner.invoke(app, [
        "ask",
        "--question", "Is the system stable?",
        "--choices", "yes",
        "--choices", "no",
        "--model", "mock",
    ])
    assert res.exit_code == 0
    assert "Decision" in res.output


def test_cli_benchmark():
    res = runner.invoke(app, [
        "benchmark",
        "--samples", "4",
        "--model", "mock",
        "--level", "L0",
    ])
    assert res.exit_code == 0
    assert "Benchmark Results" in res.output


def test_cli_doom():
    res = runner.invoke(app, [
        "doom",
        "--episodes", "1",
        "--model", "mock",
    ])
    assert res.exit_code == 0
    assert "DOOM COMBAT BENCHMARK RESULTS" in res.output


def test_cli_compare_von_requires_data_dir():
    # Real shootout needs JevBench data; without --data-dir it must fail loudly.
    res = runner.invoke(app, ["compare-von"])
    assert res.exit_code != 0


def test_cli_compare_von_rejects_test_split():
    # The locked test split is never burnable from the CLI.
    res = runner.invoke(app, [
        "compare-von", "--data-dir", "nonexistent", "--split", "test",
    ])
    assert res.exit_code != 0

