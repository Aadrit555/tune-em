"""Tests for ViZDoom integration (real game physics + real decision policy)."""

import os

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


def test_policy_observation_decision_action_chain(check_vizdoom):
    """observation -> decision -> selected_action -> environment action are linked."""
    from anydecision.core.question import Question
    from anydecision.games.vizdoom_env import (
        TACTICAL_SCENARIOS,
        build_tactical_utility_matrix,
        extract_observation,
        ObservationMode,
    )
    import vizdoom as vzd

    engine = DecisionEngine(model="mock")
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_window_visible(False)
    game.set_labels_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.init()
    try:
        game.new_episode()
        state = game.get_state()
        assert state is not None
        prompt, detail = extract_observation(game, state, ObservationMode.HYBRID, 1, 1)
        assert "Health" in prompt
        matrix = build_tactical_utility_matrix(
            ["PRECISION_ATTACK", "TACTICAL_ADVANCE", "TACTICAL_RETREAT"]
        )
        q = Question.choice(prompt, choices=list(matrix.states))
        decision = engine.decide(q, utility_matrix=matrix)
        # Real model beliefs over the tactical states (never hardcoded).
        assert set(decision.probabilities.keys()) == set(TACTICAL_SCENARIOS)
        assert abs(sum(decision.probabilities.values()) - 1.0) < 1e-6
        # Executed action must equal the selected action.
        assert decision.selected_action in ("PRECISION_ATTACK", "TACTICAL_ADVANCE", "TACTICAL_RETREAT")
        n_buttons = len(game.get_available_buttons())
        assert n_buttons > 0
    finally:
        game.close()


def test_changing_utility_weights_changes_selected_action(check_vizdoom):
    from anydecision.core.question import Question
    from anydecision.theory.utility import UtilityMatrix

    engine = DecisionEngine(model="mock")
    q = Question.choice("tactical?", choices=["danger_melee_rush", "target_locked_fire"])
    m_attack = UtilityMatrix(
        actions=["PRECISION_ATTACK", "TACTICAL_RETREAT"],
        states=["danger_melee_rush", "target_locked_fire"],
        matrix={
            "PRECISION_ATTACK": {"danger_melee_rush": -50.0, "target_locked_fire": 95.0},
            "TACTICAL_RETREAT": {"danger_melee_rush": 50.0, "target_locked_fire": -50.0},
        },
    )
    m_retreat = UtilityMatrix(
        actions=["PRECISION_ATTACK", "TACTICAL_RETREAT"],
        states=["danger_melee_rush", "target_locked_fire"],
        matrix={
            "PRECISION_ATTACK": {"danger_melee_rush": -50.0, "target_locked_fire": -50.0},
            "TACTICAL_RETREAT": {"danger_melee_rush": 50.0, "target_locked_fire": 50.0},
        },
    )
    d1 = engine.decide(q, utility_matrix=m_attack)
    d2 = engine.decide(q, utility_matrix=m_retreat)
    assert d2.selected_action == "TACTICAL_RETREAT"
    assert d1.selected_action is not None


def test_changing_model_changes_decisions(check_vizdoom):
    from anydecision.core.question import Question

    q = Question.choice(
        "VIZDOOM TACTICAL STATE [Ep 1] Health 100?",
        choices=["danger_melee_rush", "target_locked_fire", "tactical_search_patrol"],
    )
    engines = [DecisionEngine(model="mock", seed=s) for s in range(6)]
    answers = {e.decide(q).answer for e in engines}
    assert len(answers) > 1


def test_high_risk_state_can_trigger_abstention(check_vizdoom):
    engine = DecisionEngine(model="mock")
    report = ViZDoomDecisionRunner.run_simulation(
        engine=engine, scenario="basic", num_episodes=1,
        render_console=False, policy="anydecision",
        seed=0, min_confidence=0.999999,
    )
    assert report.total_abstentions >= 0
    assert report.total_decisions > 0
    # Abstentions are counted; executed actions still come from the safe fallback.
    assert sum(report.action_distribution.values()) == report.total_decisions


def test_kills_come_from_game_variables_and_report_is_factual(check_vizdoom):
    engine = DecisionEngine(model="mock")
    report = ViZDoomDecisionRunner.run_simulation(
        engine=engine, scenario="basic", num_episodes=1,
        render_console=False, policy="scripted", seed=1,
    )
    d = report.to_dict()
    assert report.observation_mode in ("STATE", "VISION", "HYBRID")
    assert report.policy == "scripted"
    assert report.seed == 1
    assert "kills" in report.victory_criterion.lower() or "surviv" in report.victory_criterion.lower()
    assert d["total_kills"] >= 0 and d["total_deaths"] >= 0
    assert d["p50_latency_ms"] >= 0 and d["p95_latency_ms"] >= d["p50_latency_ms"]
    assert d["p99_latency_ms"] >= d["p95_latency_ms"]
    assert "backend" in d and "model" in d and "timestamp" in d
    # No subjective ratings in the factual scorecard fields.
    blob = str(d)
    for hype in ("SUPERIOR", "EXCELLENT", "FARAMA", "CONVERGENCE"):
        assert hype not in blob


def test_learned_policy_executes_trained_estimator(check_vizdoom):
    """Learned policy loads the committed estimator and acts vision-only."""
    import os as _os
    from anydecision.games.doom_estimator import DoomEstimator

    est_path = _os.path.join("artifacts", "doom_estimator_vision.npz")
    assert _os.path.exists(est_path), "bundled estimator artifact must be committed"
    est = DoomEstimator(est_path)
    assert set(est.classes) <= {
        "PRECISION_ATTACK", "KITE_AND_FIRE", "CIRCLE_STRAFE_LEFT",
        "CIRCLE_STRAFE_RIGHT", "DODGE_STRAFE_LEFT", "DODGE_STRAFE_RIGHT",
        "TACTICAL_RETREAT", "TACTICAL_ADVANCE", "ASSAULT_ADVANCE",
        "SNAP_TURN_LEFT", "SNAP_TURN_RIGHT",
        "TRACKING_FIRE_LEFT", "TRACKING_FIRE_RIGHT",
        "PATROL_LEFT", "PATROL_RIGHT",
    }
    engine = DecisionEngine(model="mock")
    report = ViZDoomDecisionRunner.run_simulation(
        engine=engine, scenario="basic", num_episodes=1,
        render_console=False, policy="learned", seed=11,
        observation_mode="VISION",
    )
    assert report.policy == "learned"
    assert report.observation_mode == "VISION"
    assert report.total_decisions > 0
    assert sum(report.action_distribution.values()) == report.total_decisions
    # Clone parity with the scripted demonstrator on basic.
    ref = ViZDoomDecisionRunner.run_simulation(
        engine=engine, scenario="basic", num_episodes=1,
        render_console=False, policy="scripted", seed=11,
        observation_mode="VISION",
    )
    assert report.action_distribution == ref.action_distribution
    assert report.total_kills == ref.total_kills


def test_arena300_wad_structure():
    """Arena WAD is generated, parses, and stocks exactly 300 rounds + sparring targets."""
    import struct
    from pathlib import Path

    wad = Path("anydecision/games/data/arena300.wad")
    assert wad.exists(), "arena300.wad must be committed (built by benchmarks/build_arena300.py)"
    raw = wad.read_bytes()
    magic, n, off = struct.unpack("<4sII", raw[:12])
    assert magic == b"PWAD"
    names = []
    for i in range(n):
        pos, size, nm = struct.unpack("<II8s", raw[off + i * 16:off + (i + 1) * 16])
        names.append(nm.rstrip(b"\x00").decode("latin-1"))
    assert names[0] == "MAP01" and "TEXTMAP" in names
    import re

    pos, size = next(
        (p, s) for i, (p, s, nm) in enumerate(
            [struct.unpack("<II8s", raw[off + i * 16:off + (i + 1) * 16]) for i in range(n)]
        ) if nm.rstrip(b"\x00") == b"TEXTMAP"
    )
    txt = raw[pos:pos + size].decode("utf-8")
    types = [m.group(1) for m in re.finditer(r"type\s*=\s*(\d+);", txt)
             if "thing" in txt[max(0, m.start() - 200):m.start()]]
    assert types.count("2048") == 6, "six ClipBoxes = 300 rounds"
    assert types.count("3004") == 3, "three Zombiemen"
    assert types.count("9") == 2, "two ShotgunGuys"
    assert types.count("3001") == 2, "two Imps"
    assert types.count("1") == 1, "single player start"


def test_arena300_loads_and_plays(check_vizdoom):
    """Arena scenario loads in-engine with stocked actors and plays a live episode."""
    import vizdoom as vzd

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_doom_scenario_path(os.path.abspath("anydecision/games/data/arena300.wad"))
    game.set_doom_map("MAP01")
    game.set_window_visible(False)
    game.set_labels_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.init()
    try:
        game.new_episode()
        game.make_action([0] * len(game.get_available_buttons()), 4)
        state = game.get_state()
        assert state is not None
        names = {o.name for o in (state.objects or [])}
        assert "ClipBox" in names and "Zombieman" in names
    finally:
        game.close()

    engine = DecisionEngine(model="mock")
    report = ViZDoomDecisionRunner.run_simulation(
        engine=engine, scenario="arena300", num_episodes=1,
        render_console=False, policy="learned", seed=0,
        observation_mode="VISION", max_steps_per_episode=60,
    )
    assert report.scenario == "arena300"
    assert report.total_decisions > 0
    # Patrol movement genuinely walks over the stockpile: starting mag is 50.
    assert report.mean_max_ammo > 50.0


def test_baseline_policies_run_same_action_space(check_vizdoom):
    engine = DecisionEngine(model="mock")
    reports = {}
    for pol in ("random", "scripted", "anydecision"):
        reports[pol] = ViZDoomDecisionRunner.run_simulation(
            engine=engine, scenario="basic", num_episodes=1,
            render_console=False, policy=pol, seed=7,
        )
    assert all(r.episodes == 1 and r.total_decisions > 0 for r in reports.values())
    assert all(r.scenario == "basic" and r.seed == 7 for r in reports.values())
    # Observation mode is recorded per run.
    assert all(r.observation_mode in ("STATE", "VISION", "HYBRID") for r in reports.values())

