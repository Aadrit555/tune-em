"""Tests for universal decision interface and arbitrary domain queries (akin to von).

Validates that anydecision behaves as a universal non-autoregressive decision engine
for any prompt and arbitrary candidate choices a user provides across diverse domains.
"""

import pytest
import anydecision
from anydecision import Decision, DecisionEngine, Question
from anydecision.games.doom import DoomEnemy, DoomScenarioEnvironment, DoomTacticalAgent


def test_top_level_choose_convenience_api():
    """Test top-level anydecision.choose(prompt, choices) just like von.choose."""
    prompt = "Classify this security alert: 'Multiple failed SSH attempts from subnet 192.168.1.0/24 followed by root login.'"
    choices = [
        "brute-force attack",
        "routine database maintenance",
        "css stylesheet error",
        "scheduled backup task",
    ]

    result = anydecision.choose(prompt, choices)

    assert isinstance(result, Decision)
    assert result.answer in choices
    assert len(result.probabilities) == 4
    assert 0.0 <= result.confidence <= 1.0
    assert pytest.approx(sum(result.probabilities.values()), abs=1e-4) == 1.0
    assert result.abstained is False


def test_engine_choose_arbitrary_domains():
    """Test engine.choose across varied domains (medical, code, robotics, legal)."""
    engine = DecisionEngine(model="mock")

    # Domain 1: Code Review / Architecture
    code_prompt = "Which software architecture pattern best describes decoupled event-driven services using a central message broker?"
    code_choices = ["Publish-Subscribe", "Model-View-Controller", "Active Record", "Singleton"]
    res1 = engine.choose(code_prompt, code_choices)
    assert res1.answer in code_choices
    assert res1.level == "L0"

    # Domain 2: Clinical Triage
    med_prompt = "Patient exhibits acute chest pressure, left arm numbness, and diaphoresis."
    med_choices = ["Immediate Cardiac Triage", "Routine Outpatient Followup", "Dental Examination"]
    res2 = engine.choose(med_prompt, med_choices, level="L1")
    assert res2.answer in med_choices
    assert res2.level == "L1"

    # Domain 3: Autonomous Robotics
    robot_prompt = "LIDAR sensor detects an obstacle 0.5 meters ahead at velocity 4 m/s."
    robot_choices = ["Emergency Brake", "Accelerate Forward", "Turn Left 45 Degrees", "Turn Right 45 Degrees"]
    res3 = engine.choose(robot_prompt, robot_choices)
    assert res3.answer in robot_choices


def test_engine_decide_string_prompt_with_choices():
    """Test that engine.decide() accepts either Question object or (str, choices)."""
    engine = DecisionEngine(model="mock")

    # String prompt with choices
    res = engine.decide(
        "Is this transaction high risk?",
        choices=["high risk", "low risk"],
        min_confidence=0.50,
    )
    assert isinstance(res, Decision)
    assert res.answer in ["high risk", "low risk"]


def test_scaling_large_option_sets():
    """Test scaling to dozens of arbitrary candidate choices without arbitrary limits."""
    engine = DecisionEngine(model="mock")

    # 25 alphabetized categories
    many_choices = [f"category_{chr(97 + i)}" for i in range(25)]
    prompt = "Assign this support document to the most specific category."

    res = engine.choose(prompt, many_choices)
    assert len(res.probabilities) == 25
    assert res.answer in many_choices
    assert pytest.approx(sum(res.probabilities.values()), abs=1e-4) == 1.0


def test_custom_doom_encounter_simulation():
    """Test user-defined custom DOOM combat encounters with arbitrary demons and weapons."""
    engine = DecisionEngine(model="mock")
    env = DoomScenarioEnvironment(seed=777)
    agent = DoomTacticalAgent(engine)

    # Custom encounter: 3 Imps in an abandoned decontamination facility
    custom_state = env.create_custom_encounter(
        level_name="E1M2: Nuclear Plant - Decontamination Vault",
        enemies=[
            DoomEnemy(name="Vanguard Imp", hp=45, max_hp=45, damage_per_attack=10, distance_meters=3.5, threat_level="MEDIUM"),
            DoomEnemy(name="Flanking Imp", hp=45, max_hp=45, damage_per_attack=10, distance_meters=8.0, threat_level="MEDIUM"),
        ],
        weapon="Super Shotgun",
        health=90,
        armor=40,
        pickup="Medikit (+25 HP)",
    )

    action, decision, latency_ms = agent.decide_combat_action(custom_state)
    assert action != ""
    assert latency_ms > 0.0

    # Execute action and ensure target takes damage or dies
    next_state, outcome = env.execute_action(custom_state, action)
    assert outcome.action == action
    assert outcome.summary != ""
    if outcome.damage_dealt > 0:
        # Super Shotgun deals > 100 damage, so Vanguard Imp (45 HP) should be obliterated!
        assert outcome.enemy_killed is True
        assert len(next_state.enemies) == 1

