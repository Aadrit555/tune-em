"""Tests for authentic Ultimate DOOM IWAD parsing and combat decision evaluation."""

import os
import pytest
from typer.testing import CliRunner

from anydecision.cli.main import app
from anydecision.core.engine import DecisionEngine
from anydecision.games.ultimate_doom import (
    DEFAULT_WAD_SEARCH_PATHS,
    RealDoomEvaluator,
    RealDoomMap,
    RealDoomScoreReport,
    RealWadEntity,
    UltimateDoomWadParser,
)


def _find_installed_wad():
    for p in DEFAULT_WAD_SEARCH_PATHS:
        if os.path.exists(p):
            return p
    return None


@pytest.fixture
def wad_path():
    p = _find_installed_wad()
    if not p:
        pytest.skip("Local DOOM.WAD file not found in search paths")
    return p


def test_wad_parser_directory(wad_path):
    parser = UltimateDoomWadParser(wad_path=wad_path)
    maps = parser.list_maps()
    assert "E1M1" in maps
    assert "E1M8" in maps
    assert "E2M8" in maps
    assert len(maps) >= 9


def test_wad_parser_parse_e1m1(wad_path):
    parser = UltimateDoomWadParser(wad_path=wad_path)
    doom_map = parser.parse_map("E1M1", skill_level=3)
    assert isinstance(doom_map, RealDoomMap)
    assert doom_map.map_code == "E1M1"
    assert "Hangar" in doom_map.title
    assert doom_map.total_entities > 0
    assert len(doom_map.monsters) > 0
    assert len(doom_map.pickups) > 0
    assert len(doom_map.hazards) > 0

    first_monster = doom_map.monsters[0]
    assert isinstance(first_monster, RealWadEntity)
    assert first_monster.category == "monster"
    assert first_monster.hp > 0
    assert first_monster.distance_to_player >= 0


def test_wad_parser_missing_path():
    with pytest.raises(FileNotFoundError):
        UltimateDoomWadParser(wad_path=r"C:\NonExistent\Fake\DOOM.WAD")


def test_wad_parser_invalid_map(wad_path):
    parser = UltimateDoomWadParser(wad_path=wad_path)
    with pytest.raises(KeyError):
        parser.parse_map("E9M9")


def test_real_doom_evaluator(wad_path):
    engine = DecisionEngine(model="mock")
    report = RealDoomEvaluator.run_map_evaluation(
        engine=engine,
        map_code="E1M1",
        skill_level=3,
        wad_path=wad_path,
        render_console=False,
    )
    assert isinstance(report, RealDoomScoreReport)
    assert report.map_code == "E1M1"
    assert report.total_demons_spawned > 0
    assert report.demons_slain >= 0
    assert report.total_decisions > 0
    assert report.mean_decision_latency_ms >= 0
    assert report.status in ("OBJECTIVE MET", "EPISODE END")
    assert len(report.telemetry_log) > 0


def test_real_doom_missing_wad_fails_loudly():
    engine = DecisionEngine(model="mock")
    with pytest.raises(FileNotFoundError):
        RealDoomEvaluator.run_map_evaluation(
            engine=engine,
            map_code="E1M1",
            skill_level=3,
            wad_path=r"C:\NonExistent\Fake\DOOM.WAD",
            render_console=False,
        )


def test_real_doom_cli_command(wad_path):
    runner = CliRunner()
    result = runner.invoke(app, ["real-doom", "--map", "E1M1", "--skill", "3", "--wad", wad_path])
    assert result.exit_code == 0
    assert "ULTIMATE DOOM" in result.output
    assert "SCORECARD" in result.output

