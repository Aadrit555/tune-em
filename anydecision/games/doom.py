"""DOOM Tactical Decision Environment & AI Combat Simulator.

Validates the anydecision engine under fast, high-stakes, uncertain game conditions.
Models real-time tactical combat scenarios from Classic DOOM (id Software):
- Monsters: Zombieman, Imp, Pinky Demon, Cacodemon, Baron of Hell, Cyberdemon
- Weapon Arsenal: Shotgun, Super Shotgun, Chaingun, Plasma Rifle, BFG9000, Chainsaw
- Hazards: Slime pits, explosive barrels, projectile fireballs, ambush traps
- Health/Armor & Doomguy HUD facial states: Healthy [ >:D ], Hurt [ :| ], Bloodied [ D: ], Dead [ X_X ]
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, Field
from rich.box import DOUBLE, HEAVY, ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question
from anydecision.theory.utility import UtilityMatrix

# ASCII Art Scenes for DOOM encounters
ASCII_ENCOUNTERS = {
    "Cyberdemon": r"""
           /\_/\
         =( o.o )=   << CYBERDEMON TOWERING 20FT AHEAD >>
          /     \       [ARM-MOUNTED ROCKET LAUNCHER ARMED]
         (  (X)  )      * THUD * * THUD * * ROAAAR *
        (__)___(__)
    """,
    "Baron of Hell": r"""
         (o)(o)      << BARON OF HELL CHARGING >>
        /  /\  \        [GREEN PLASMA FIREBALLS CHARGING]
       |  |  |  |       * HISS * * CREAK *
        \/    \/
    """,
    "Cacodemon": r"""
        .--------.   << CACODEMON HOVERING IN AIR >>
       /  (O)  (O)\     [ELECTRIC LIGHTNING BOLT SPHERE]
      |     V      |
       \  \====/  /
        '--------'
    """,
    "Imp Swarm": r"""
        /\  /\  /\   << AMBUSH: 3 IMPS HURLING FIREBALLS >>
       (oo)(oo)(oo)     [FLAMES CRACKLING ACROSS CORRIDOR]
        ||  ||  ||
    """,
    "Pinky Demon": r"""
        (\___/)      << PINKY DEMON SPRINTING IN MELEE >>
        / O O \         [RAZOR JAWS SNAPPING AT 3 METERS]
       (   "   )
    """,
}

DOOMGUY_HUD = {
    "healthy": "[ >:D ] GODLIKE (100%)",
    "good":    "[ :D ] HEALTHY (75-99%)",
    "hurt":    "[ :| ] HURT (40-74%)",
    "blood":   "[ D: ] CRITICAL (1-39%)",
    "dead":    "[ X_X ] SLAIN IN BATTLE",
}


class DoomEnemy(BaseModel):
    name: str
    hp: int
    max_hp: int
    damage_per_attack: int
    distance_meters: float
    threat_level: str


class DoomGameState(BaseModel):
    """Snapshot of player condition and combat environment."""
    level_name: str
    turn: int
    health: int
    armor: int
    current_weapon: str
    ammo: Dict[str, int]
    enemies: List[DoomEnemy]
    hazard: Optional[str] = None
    pickup: Optional[str] = None
    ascii_scene: str = ""

    def hud_status(self) -> str:
        if self.health <= 0:
            return DOOMGUY_HUD["dead"]
        elif self.health < 40:
            return DOOMGUY_HUD["blood"]
        elif self.health < 75:
            return DOOMGUY_HUD["hurt"]
        elif self.health < 100:
            return DOOMGUY_HUD["good"]
        return DOOMGUY_HUD["healthy"]


class DoomActionOutcome(BaseModel):
    action: str
    success: bool
    damage_dealt: int
    damage_taken: int
    enemy_killed: bool
    summary: str


class DoomScenarioEnvironment:
    """Simulates real-time tactical combat scenarios for Decision Engine benchmark."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def generate_encounter(self, difficulty: str = "medium") -> DoomGameState:
        """Create a randomized authentic DOOM combat encounter."""
        if difficulty == "boss":
            level = "E2M8: Tower of Babel"
            enemy_type = "Cyberdemon"
            enemies = [DoomEnemy(name="Cyberdemon", hp=400, max_hp=400, damage_per_attack=45, distance_meters=18.0, threat_level="LETHAL")]
            hazard = "Explosive Rocket Blast Radius"
            pickup = "Soul Sphere (+100 HP)" if self.rng.random() > 0.4 else None
            weapon = "BFG9000" if self.rng.random() > 0.5 else "Plasma Rifle"
            health = self.rng.randint(45, 90)
        elif difficulty == "hard":
            level = "E1M8: Phobos Anomaly"
            enemy_type = "Baron of Hell"
            enemies = [
                DoomEnemy(name="Baron of Hell #1", hp=180, max_hp=180, damage_per_attack=25, distance_meters=8.0, threat_level="HIGH"),
                DoomEnemy(name="Baron of Hell #2", hp=180, max_hp=180, damage_per_attack=25, distance_meters=14.0, threat_level="HIGH"),
            ]
            hazard = "Acid Floor (-5 HP/sec)"
            pickup = "Medikit (+25 HP)"
            weapon = "Super Shotgun"
            health = self.rng.randint(30, 70)
        else:  # medium
            level = "E1M1: Hangar"
            enemy_type = self.rng.choice(["Imp Swarm", "Pinky Demon", "Cacodemon"])
            if enemy_type == "Imp Swarm":
                enemies = [DoomEnemy(name=f"Imp #{i+1}", hp=60, max_hp=60, damage_per_attack=12, distance_meters=4.0 + i*3, threat_level="MEDIUM") for i in range(2)]
            elif enemy_type == "Pinky Demon":
                enemies = [DoomEnemy(name="Pinky Demon", hp=120, max_hp=120, damage_per_attack=18, distance_meters=3.2, threat_level="MEDIUM")]
            else:
                enemies = [DoomEnemy(name="Cacodemon", hp=150, max_hp=150, damage_per_attack=20, distance_meters=9.0, threat_level="HIGH")]

            hazard = "Explosive Toxic Barrel" if self.rng.random() > 0.6 else None
            pickup = "Box of Shells (+20)" if self.rng.random() > 0.5 else None
            weapon = "Shotgun"
            health = self.rng.randint(25, 85)

        ascii_art = ASCII_ENCOUNTERS.get(enemy_type, ASCII_ENCOUNTERS["Baron of Hell"])

        return DoomGameState(
            level_name=level,
            turn=1,
            health=health,
            armor=self.rng.randint(0, 50),
            current_weapon=weapon,
            ammo={"shells": 16, "cells": 80, "rockets": 6, "bullets": 50},
            enemies=enemies,
            hazard=hazard,
            pickup=pickup,
            ascii_scene=ascii_art,
        )

    def execute_action(self, state: DoomGameState, action: str) -> Tuple[DoomGameState, DoomActionOutcome]:
        """Step simulation environment according to the AI's tactical action."""
        new_state = state.model_copy(deep=True)
        new_state.turn += 1

        damage_dealt = 0
        damage_taken = 0
        killed = False
        summary_msg = ""

        primary_target = new_state.enemies[0] if new_state.enemies else None

        if action.startswith("shoot_") or action == "attack":
            # Fire active weapon
            base_dmg = 75 if "bfg" in action or new_state.current_weapon == "BFG9000" else 45
            damage_dealt = int(base_dmg * self.rng.uniform(0.85, 1.35))
            if primary_target:
                primary_target.hp -= damage_dealt
                if primary_target.hp <= 0:
                    killed = True
                    new_state.enemies.pop(0)
                    summary_msg = f"DIRECT HIT: {damage_dealt} damage! {primary_target.name} blown to gibs!"
                else:
                    summary_msg = f"HIT: Dealt {damage_dealt} damage to {primary_target.name} ({primary_target.hp} HP remaining)."

            # Counter-attack by remaining enemies
            if new_state.enemies:
                threat = new_state.enemies[0]
                damage_taken = int(threat.damage_per_attack * self.rng.uniform(0.6, 1.1))

        elif action.startswith("dodge_") or action == "strafe":
            # Tactical evasion
            summary_msg = "EVASIVE MANEUVER: Strafed sideways, projectile missed completely!"
            damage_taken = int(self.rng.choice([0, 0, 5]))  # Mostly safe dodge
            if primary_target:
                # Enemy advances or stays at bay
                primary_target.distance_meters = max(2.0, primary_target.distance_meters - 1.0)

        elif action == "take_cover":
            summary_msg = "COVER SECURED: Ducked behind blast-resistant bulkhead. Enemy lost line of sight."
            damage_taken = 0

        elif action.startswith("grab_") or action == "collect_pickup":
            if new_state.pickup and "Health" in new_state.pickup or "Medikit" in str(new_state.pickup):
                heal_amt = 25
                new_state.health = min(100, new_state.health + heal_amt)
                summary_msg = f"HEALED: Snatched Medikit! Restored +{heal_amt} HP (Current: {new_state.health}%)."
            elif new_state.pickup and "Soul Sphere" in str(new_state.pickup):
                new_state.health = min(200, new_state.health + 100)
                summary_msg = f"SOUL SPHERE CONSUMED: Supercharged to {new_state.health}% HP!"
            else:
                summary_msg = "SUPPLY RUN: Grabbed spare ammunition."
            damage_taken = int(self.rng.randint(0, 10))

        elif action == "chainsaw_charge":
            if primary_target and primary_target.name in ("Cyberdemon", "Baron of Hell"):
                damage_taken = 60
                summary_msg = f"FATAL ERROR: Rushed {primary_target.name} with chainsaw! Crushed by demon fist!"
            else:
                damage_dealt = 90
                killed = True
                if new_state.enemies:
                    new_state.enemies.pop(0)
                summary_msg = "RIP AND TEAR: Chainsaw sliced demon in half!"

        # Apply damage to armor first, then health
        if damage_taken > 0:
            if new_state.armor > 0:
                armor_absorbed = min(new_state.armor, damage_taken // 2)
                new_state.armor -= armor_absorbed
                damage_taken -= armor_absorbed
            new_state.health = max(0, new_state.health - damage_taken)

        outcome = DoomActionOutcome(
            action=action,
            success=new_state.health > 0,
            damage_dealt=damage_dealt,
            damage_taken=damage_taken,
            enemy_killed=killed,
            summary=summary_msg,
        )

        return new_state, outcome


class DoomTacticalAgent:
    """Autonomous AI Agent driven by anydecision Expected Utility runtime."""

    def __init__(self, engine: DecisionEngine) -> None:
        self.engine = engine

    def decide_combat_action(self, state: DoomGameState) -> Tuple[str, Any, float]:
        """Convert game state into typed decision question and compute optimal action via Expected Utility."""
        enemy_desc = ", ".join(f"{e.name} ({e.threat_level} threat, {e.distance_meters:.1f}m away)" for e in state.enemies) if state.enemies else "Area clear"
        prompt_text = (
            f"DOOM COMBAT DECISION [Turn {state.turn} | {state.level_name}]\n"
            f"Doomguy Vitality: {state.health}% HP | Armor: {state.armor}%\n"
            f"Equipped Weapon: {state.current_weapon}\n"
            f"Hostile Contacts: {enemy_desc}\n"
            f"Hazard: {state.hazard or 'None'} | Pickup: {state.pickup or 'None'}\n"
            f"Question: What is the highest-utility tactical action for survival and elimination?"
        )

        # Build candidate choices
        choices = ["shoot_primary", "dodge_evade", "take_cover"]
        if state.pickup:
            choices.append("grab_pickup")
        if state.current_weapon == "Chainsaw" or state.health > 80:
            choices.append("chainsaw_charge")

        # Construct Decision-Theoretic Expected Utility Matrix
        # Actions vs Game State Scenarios (e.g. enemy_attack, incoming_projectile, safe_window)
        utility_grid = {}
        for action in choices:
            utility_grid[action] = {}
            for scenario in ["incoming_attack", "enemy_advancing", "vulnerable_target"]:
                if action == "shoot_primary":
                    # Shooting is high reward if target is vulnerable, risky if low health
                    u = 25.0 if scenario == "vulnerable_target" else (10.0 if state.health > 50 else -15.0)
                elif action == "dodge_evade":
                    # Dodge is highest utility when attack is incoming
                    u = 35.0 if scenario == "incoming_attack" else 5.0
                elif action == "take_cover":
                    u = 20.0 if scenario == "incoming_attack" else 0.0
                elif action == "grab_pickup":
                    # Grab pickup is lifesaving when health is low
                    u = 50.0 if state.health < 40 else 15.0
                elif action == "chainsaw_charge":
                    u = 40.0 if scenario == "vulnerable_target" and state.health > 60 else -60.0
                else:
                    u = 0.0
                utility_grid[action][scenario] = u

        scenarios = ["incoming_attack", "enemy_advancing", "vulnerable_target"]
        q = Question.choice(
            text=prompt_text,
            choices=scenarios,
        )

        matrix = UtilityMatrix(
            actions=choices,
            states=scenarios,
            matrix=utility_grid,
        )

        t0 = time.perf_counter()
        # Execute decision with adaptive early-exit routing for low latency
        decision = self.engine.decide_adaptive(
            question=q,
            utility_matrix=matrix,
            track_layer_trajectory=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        chosen_action = decision.selected_action or "shoot_primary"
        return chosen_action, decision, latency_ms


class DoomCombatBenchmarkRunner:
    """Runs automated DOOM combat simulation episodes and logs decision metrics."""

    @staticmethod
    def run_simulation(
        engine: DecisionEngine,
        num_episodes: int = 5,
        max_turns_per_episode: int = 6,
        difficulty: str = "medium",
        render_console: bool = False,
    ) -> Dict[str, Any]:
        env = DoomScenarioEnvironment()
        agent = DoomTacticalAgent(engine)
        console = Console() if render_console else None

        episodes_survived = 0
        total_kills = 0
        total_damage_dealt = 0
        total_decisions = 0
        latencies = []
        action_counts: Dict[str, int] = {}

        for ep in range(1, num_episodes + 1):
            state = env.generate_encounter(difficulty=difficulty)
            if console:
                console.print(f"\n[bold red]=== DOOM COMBAT EPISODE #{ep}: {state.level_name} ===[/bold red]")
                console.print(f"[bold cyan]{state.ascii_scene}[/bold cyan]")

            for turn in range(1, max_turns_per_episode + 1):
                if state.health <= 0 or not state.enemies:
                    break

                action, dec, lat_ms = agent.decide_combat_action(state)
                latencies.append(lat_ms)
                total_decisions += 1
                action_counts[action] = action_counts.get(action, 0) + 1

                new_state, outcome = env.execute_action(state, action)

                if console:
                    conf = dec.confidence * 100
                    emergence = dec.decision_emergence_layer or 24
                    console.print(
                        f"  Turn {turn:02d} | HUD: {state.hud_status()} | "
                        f"Action: [bold yellow]{action.upper()}[/bold yellow] "
                        f"(EU: {dec.optimal_action_utility:+.1f} | Conf: {conf:.1f}% | L{emergence}) | "
                        f"{outcome.summary}"
                    )

                state = new_state
                if outcome.enemy_killed:
                    total_kills += 1
                total_damage_dealt += outcome.damage_dealt

            if state.health > 0:
                episodes_survived += 1
                if console:
                    console.print(f"[bold green]>> EPISODE #{ep} CLEARED! Doomguy survived with {state.health}% HP! <<[/bold green]")
            else:
                if console:
                    console.print(f"[bold red]>> EPISODE #{ep} FAILED: Doomguy was slain. <<[/bold red]")

        survival_rate = episodes_survived / max(1, num_episodes)
        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        decisions_per_sec = float(1000.0 / max(1e-6, mean_lat))

        return {
            "num_episodes": num_episodes,
            "survival_rate": survival_rate,
            "total_kills": total_kills,
            "total_decisions": total_decisions,
            "mean_latency_ms": mean_lat,
            "decisions_per_second": decisions_per_sec,
            "action_distribution": action_counts,
        }
