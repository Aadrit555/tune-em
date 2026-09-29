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


def test_showcase_runs_fast_and_reports_receipts():
    """Recordable showcase must run non-interactively with honest accounting."""
    from anydecision.demo.showcase import run_showcase

    res = run_showcase(fast=True, games=False)
    assert res["act1_answer"] in ("SQL Injection", "XSS", "CSRF", "Buffer Overflow")
    assert res["ece_after"] <= res["ece_before"]
    assert res["selected_action"] in ("freeze_account", "approve_transfer", "step_up_2fa")
    assert res["abstained"] is True
    assert res["backend_calls"] <= 16
    assert "+".join(res["compute_path"]) in ("L0", "L0+L1", "L0+L1+L2")
    assert res["artifact_head"] == "temperature_scaling"


def test_cli_showcase_command():
    res = runner.invoke(app, ["showcase", "--fast"])
    assert res.exit_code == 0
    assert "Showcase complete" in res.output

