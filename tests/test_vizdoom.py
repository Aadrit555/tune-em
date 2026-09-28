"""Tests for Farama Foundation ViZDoom AI research platform integration."""

import pytest
from typer.testing import CliRunner

from anydecision.cli.main import app
from anydecision.core.engine import DecisionEngine
from anydecision.games.vizdoom_env import (
    ViZDoomDecisionRunner,
    ViZDoomScoreReport,
    is_vizdoom_available,
)


@pytest.fixture
def check_vizdoom():
    if not is_vizdoom_available():
        pytest.skip("ViZDoom package not installed in environment")


def test_is_vizdoom_available(check_vizdoom):
    assert is_vizdoom_available() is True


def test_vizdoom_basic_simulation(check_vizdoom):
    engine = DecisionEngine(model="mock")
    report = ViZDoomDecisionRunner.run_simulation(
        engine=engine,
        scenario="basic",
        num_episodes=1,
        render_console=False,
    )
    assert isinstance(report, ViZDoomScoreReport)
    assert report.scenario == "basic"
    assert report.episodes == 1
    assert report.total_decisions > 0
    assert report.mean_latency_ms >= 0
    assert report.decisions_per_sec >= 0
    assert len(report.telemetry_log) > 0


def test_vizdoom_cli_command(check_vizdoom):
    runner = CliRunner()
    result = runner.invoke(app, ["vizdoom", "--scenario", "basic", "--episodes", "1"])
    assert result.exit_code == 0
    assert "VIZDOOM" in result.output
    assert "SCORECARD" in result.output
