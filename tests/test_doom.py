"""Tests for DOOM tactical decision environment and combat benchmark."""

import pytest
from anydecision.core.engine import DecisionEngine
from anydecision.games.doom import (
    DoomCombatBenchmarkRunner,
    DoomEnemy,
    DoomGameState,
    DoomScenarioEnvironment,
    DoomTacticalAgent,
)


def test_doom_environment_generation():
    env = DoomScenarioEnvironment(seed=42)

    for diff in ["easy", "medium", "hard", "boss"]:
        state = env.generate_encounter(difficulty=diff)
        assert isinstance(state, DoomGameState)
        assert len(state.enemies) >= 1
        assert state.health > 0
        assert state.current_weapon != ""
        assert state.hud_status() != ""
        assert len(state.ascii_scene) > 0


def test_doom_environment_action_execution():
    env = DoomScenarioEnvironment(seed=101)
    state = env.generate_encounter(difficulty="easy")
    initial_enemy_hp = state.enemies[0].hp

    new_state, outcome = env.execute_action(state, "shoot_primary")

    assert outcome.action == "shoot_primary"
    assert outcome.damage_dealt > 0
    # Enemy took damage or died
    if new_state.enemies:
        assert new_state.enemies[0].hp < initial_enemy_hp or outcome.enemy_killed
    else:
        assert outcome.enemy_killed


def test_doom_tactical_agent_decision():
    engine = DecisionEngine(model="mock")
    agent = DoomTacticalAgent(engine)
    env = DoomScenarioEnvironment(seed=42)

    state = env.generate_encounter(difficulty="medium")
    action, decision, latency_ms = agent.decide_combat_action(state)

    assert action in ["shoot_primary", "dodge_evade", "take_cover", "grab_pickup", "chainsaw_charge"]
    assert decision.selected_action == action
    assert decision.expected_utilities is not None
    assert latency_ms > 0.0


def test_doom_combat_benchmark_simulation():
    engine = DecisionEngine(model="mock")
    results = DoomCombatBenchmarkRunner.run_simulation(
        engine=engine,
        num_episodes=2,
        max_turns_per_episode=3,
        difficulty="medium",
        render_console=False,
    )

    assert results["num_episodes"] == 2
    assert 0.0 <= results["survival_rate"] <= 1.0
    assert results["total_decisions"] > 0
    assert results["mean_latency_ms"] >= 0.0
    assert results["decisions_per_second"] > 0.0
    assert isinstance(results["action_distribution"], dict)
