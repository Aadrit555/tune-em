"""ViZDoom integration: real game physics + real anydecision policy control.

Real pipeline (no hardcoded beliefs, no ignored Decisions):

    REAL ViZDoom STATE -> observation extraction (STATE / VISION / HYBRID)
    -> typed DecisionContext prompt -> anydecision candidate scoring
    -> UtilityMatrix -> Decision.selected_action -> REAL ViZDoom action
    -> REAL game transition -> authoritative game variables
    -> factual telemetry + machine-readable artifact.

Observation modes:
    STATE:  privileged game variables + object world coordinates.
    VISION: rendered screen + labels-buffer bounding boxes only
            (no world coordinates, no object names beyond the label buffer).
    HYBRID: both sources.

Policies (identical episodes/seeds/action-space for fair comparison):
    anydecision: model candidate scoring over tactical states + expected utility.
    random:      uniform random action (baseline).
    scripted:    fixed crosshair/bearing heuristic (baseline, clearly labeled).

Metrics use authoritative game variables ONLY:
    kills  = delta KILLCOUNT (never inferred from reward),
    deaths = delta DEATHCOUNT,
    victory is scenario-defined (see VICTORY_CRITERIA); positive reward
    alone is never called a win.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from rich.box import DOUBLE
from rich.console import Console
from rich.table import Table

from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question
from anydecision.games.ultimate_doom import DEFAULT_WAD_SEARCH_PATHS
from anydecision.theory.utility import UtilityMatrix

try:
    import vizdoom as vzd
    VIZDOOM_AVAILABLE = True
except ImportError:
    vzd = None  # type: ignore[assignment]
    VIZDOOM_AVAILABLE = False


def is_vizdoom_available() -> bool:
    """Return True if the vizdoom package is installed and importable."""
    return VIZDOOM_AVAILABLE


class ObservationMode(str, Enum):
    """Which sensor source the observation was built from."""

    STATE = "STATE"  # privileged game variables + world coordinates
    VISION = "VISION"  # screen/labels buffer only, no world coordinates
    HYBRID = "HYBRID"  # both


class PolicyKind(str, Enum):
    """Which policy selected the executed action."""

    ANYDECISION = "anydecision"
    RANDOM = "random"
    SCRIPTED = "scripted"
    LEARNED = "learned"  # trained state-estimator MLP (see doom_estimator.py)


# Scenario-defined victory predicates (explicit per scenario; no universal
# "positive reward = victory" rule). Each entry documents its criterion.
VICTORY_CRITERIA: Dict[str, str] = {
    "basic": "kills > 0 (single-monster elimination objective)",
    "simpler_basic": "kills > 0 (single-monster elimination objective)",
    "rocket_basic": "kills > 0 (single-monster elimination objective)",
    "defend_the_center": "kills > 0 (combat survival objective)",
    "defend_the_line": "kills > 0 (combat survival objective)",
    "deadly_corridor": "kills > 0 (combat traversal objective)",
    "cig": "kills > 0 (combat objective)",
    "deathmatch": "kills > 0 (frag objective)",
    "health_gathering": "survived episode without death (collection objective)",
    "health_gathering_supreme": "survived episode without death (collection objective)",
    "my_way_home": "survived episode without death (navigation objective)",
    "take_cover": "survived episode without death (survival objective)",
}

COMBAT_SCENARIOS = {
    "basic", "simpler_basic", "rocket_basic", "defend_the_center",
    "defend_the_line", "deadly_corridor", "cig", "deathmatch",
}


def victory_criterion_for(scenario: str) -> str:
    """Return the documented victory criterion for a scenario."""
    key = scenario.lower()
    if key in VICTORY_CRITERIA:
        return VICTORY_CRITERIA[key]
    if key.startswith("e1m") or key.startswith("e2m") or key.startswith("e3m") or key.startswith("e4m"):
        return "kills > 0 or survived without death (WAD-map proxy; exit-switch state not tracked)"
    return "kills > 0 or survived without death (default proxy; scenario has no registered criterion)"


def is_victory(scenario: str, kills: int, died: bool) -> bool:
    """Scenario-defined victory predicate over authoritative counters."""
    key = scenario.lower()
    if key in COMBAT_SCENARIOS or key.startswith(("e1m", "e2m", "e3m", "e4m")):
        return kills > 0
    if key in VICTORY_CRITERIA:
        # Survival objectives.
        return not died
    return kills > 0 or not died


class ViZDoomScoreReport(BaseModel):
    """Factual end-of-benchmark report for ViZDoom episodes (no subjective ratings)."""

    scenario: str
    victory_criterion: str
    observation_mode: str = "HYBRID"
    policy: str = "anydecision"
    episodes: int = 0
    episodes_won: int = 0
    win_rate: float = 0.0
    completion_rate: float = 0.0
    total_kills: int = 0
    total_deaths: int = 0
    total_items: int = 0
    total_damage_taken: float = 0.0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    mean_final_health: float = 0.0
    mean_final_armor: float = 0.0
    total_expected_utility: float = 0.0
    total_decisions: int = 0
    total_abstentions: int = 0
    total_backend_calls: int = 0
    total_tokens: int = 0
    tokens_estimated: bool = True
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    decisions_per_sec: float = 0.0
    action_distribution: Dict[str, int] = Field(default_factory=dict)
    compute_path_distribution: Dict[str, int] = Field(default_factory=dict)
    seed: Optional[int] = None
    backend: str = "unknown"
    model: str = "unknown"
    model_revision: str = "main"
    commit: Optional[str] = None
    timestamp: str = ""
    telemetry_log: List[str] = Field(default_factory=list)
    trajectory: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-step sensor/action/reward records (only when record_trajectory=True).",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Machine-readable benchmark artifact dict."""
        return self.model_dump()

    def save_json(self, path: str | Path) -> Path:
        """Write machine-readable artifact to path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=float), q))


def _safe_game_var(game: Any, var: Any, default: float = 0.0) -> float:
    try:
        if var in game.get_available_game_variables():
            return float(game.get_game_variable(var))
    except Exception:
        pass
    return default


TACTICAL_SCENARIOS = [
    "danger_melee_rush",
    "target_locked_fire",
    "target_flank_left",
    "target_flank_right",
    "target_behind",
    "tactical_search_patrol",
]

_FALLBACK_SAFE_ACTION = "TACTICAL_RETREAT"


def build_tactical_utility_matrix(
    action_names: List[str], scenarios: Optional[List[str]] = None
) -> UtilityMatrix:
    """Shared explicit utility grid mapping tactical actions to hypothesis states."""
    states = scenarios or list(TACTICAL_SCENARIOS)
    grid: Dict[str, Dict[str, float]] = {
        "PRECISION_ATTACK": {
            "danger_melee_rush": 20.0, "target_locked_fire": 95.0,
            "target_flank_left": -40.0, "target_flank_right": -40.0,
            "target_behind": -50.0, "tactical_search_patrol": -50.0,
        },
        "KITE_AND_FIRE": {
            "danger_melee_rush": 95.0, "target_locked_fire": 50.0,
            "target_flank_left": 10.0, "target_flank_right": 10.0,
            "target_behind": -30.0, "tactical_search_patrol": -20.0,
        },
        "CIRCLE_STRAFE_LEFT": {
            "danger_melee_rush": 85.0, "target_locked_fire": 45.0,
            "target_flank_left": 20.0, "target_flank_right": 75.0,
            "target_behind": 10.0, "tactical_search_patrol": 10.0,
        },
        "CIRCLE_STRAFE_RIGHT": {
            "danger_melee_rush": 85.0, "target_locked_fire": 45.0,
            "target_flank_left": 75.0, "target_flank_right": 20.0,
            "target_behind": 10.0, "tactical_search_patrol": 10.0,
        },
        "DODGE_STRAFE_LEFT": {
            "danger_melee_rush": 75.0, "target_locked_fire": 10.0,
            "target_flank_left": 30.0, "target_flank_right": 70.0,
            "target_behind": 40.0, "tactical_search_patrol": 20.0,
        },
        "DODGE_STRAFE_RIGHT": {
            "danger_melee_rush": 75.0, "target_locked_fire": 10.0,
            "target_flank_left": 70.0, "target_flank_right": 30.0,
            "target_behind": 40.0, "tactical_search_patrol": 20.0,
        },
        "TACTICAL_RETREAT": {
            "danger_melee_rush": 70.0, "target_locked_fire": 0.0,
            "target_flank_left": 10.0, "target_flank_right": 10.0,
            "target_behind": 0.0, "tactical_search_patrol": 0.0,
        },
        "TACTICAL_ADVANCE": {
            "danger_melee_rush": -40.0, "target_locked_fire": 30.0,
            "target_flank_left": 10.0, "target_flank_right": 10.0,
            "target_behind": 10.0, "tactical_search_patrol": 75.0,
        },
        "ASSAULT_ADVANCE": {
            "danger_melee_rush": -30.0, "target_locked_fire": 75.0,
            "target_flank_left": -20.0, "target_flank_right": -20.0,
            "target_behind": -30.0, "tactical_search_patrol": 15.0,
        },
        "SNAP_TURN_LEFT": {
            "danger_melee_rush": 20.0, "target_locked_fire": -25.0,
            "target_flank_left": 90.0, "target_flank_right": -35.0,
            "target_behind": 80.0, "tactical_search_patrol": 30.0,
        },
        "SNAP_TURN_RIGHT": {
            "danger_melee_rush": 20.0, "target_locked_fire": -25.0,
            "target_flank_left": -35.0, "target_flank_right": 90.0,
            "target_behind": 80.0, "tactical_search_patrol": 50.0,
        },
    }
    matrix = {a: grid.get(a, {s: 0.0 for s in states}) for a in action_names}
    return UtilityMatrix(actions=action_names, states=states, matrix=matrix)


_IGNORED_LABELS = {
    "DoomPlayer", "BulletPuff", "Blood", "TeleportFog", "GreenArmor", "BlueArmor",
    "Medikit", "Stimpack", "HealthBonus", "ArmorBonus", "Clip", "ShellBox", "RocketBox",
}


# Fixed sensor feature order for learned estimators (behavior cloning value).
# All values are numeric and game-engine grounded; VISION-safe subset excludes
# radar geometry (nearest_dist/nearest_rel_deg/radar_hostiles are None there).
FEATURE_ORDER = [
    "health", "armor", "ammo",
    "visible_hostiles", "crosshair_locked", "target_offset_x",
    "radar_hostiles", "nearest_dist", "nearest_rel_deg",
]


def features_from_detail(detail: Dict[str, Any]) -> List[float]:
    """Numeric feature vector in FEATURE_ORDER (booleans -> 0.0/1.0, None -> sentinel)."""
    vals: List[float] = []
    for key in FEATURE_ORDER:
        v = detail.get(key)
        if v is None:
            vals.append(-1.0)
        elif isinstance(v, bool):
            vals.append(1.0 if v else 0.0)
        else:
            vals.append(float(v))
    # Normalize roughly to unit-ish ranges for stable MLP training.
    scales = [100.0, 100.0, 50.0, 4.0, 1.0, 160.0, 4.0, 1000.0, 180.0]
    return [v / s for v, s in zip(vals, scales)]


def extract_observation(
    game: Any,
    state: Any,
    mode: ObservationMode,
    episode: int,
    step: int,
) -> Tuple[str, Dict[str, Any]]:
    """Build a structured observation prompt from the live game state.

    STATE mode uses privileged variables + object world coordinates.
    VISION mode uses the labels buffer (bounding boxes) + game variables
    that a screen-only agent could plausibly read (health/ammo HUD values
    are exposed as numeric HUD state, not world geometry).
    Returns (prompt_text, observation_detail).
    """
    health = _safe_game_var(game, vzd.GameVariable.HEALTH, 100.0)
    armor = _safe_game_var(game, vzd.GameVariable.ARMOR, 0.0)
    ammo = _safe_game_var(game, vzd.GameVariable.SELECTED_WEAPON_AMMO, 50.0)

    screen_width = game.get_screen_width() or 320
    screen_center = screen_width / 2.0
    vis_monsters = [
        lbl for lbl in (state.labels or [])
        if lbl.object_name not in _IGNORED_LABELS
        and not lbl.object_name.startswith("Dead")
    ]
    target_in_crosshair = False
    target_offset_x = 0.0
    vis_target_name = "none visible"
    vis_target_size = 0.0
    if vis_monsters:
        vis_monsters.sort(
            key=lambda m: abs((m.x + m.width / 2.0) - screen_center) - (m.height * 2.0)
        )
        primary = vis_monsters[0]
        vis_target_name = primary.object_name
        vis_target_size = float(primary.width * primary.height)
        target_offset_x = (primary.x + primary.width / 2.0) - screen_center
        target_in_crosshair = abs(target_offset_x) / max(1.0, float(screen_width)) <= 0.055

    detail: Dict[str, Any] = {
        "health": health, "armor": armor, "ammo": ammo,
        "visible_hostiles": len(vis_monsters),
        "crosshair_locked": target_in_crosshair,
        "target_offset_x": target_offset_x,
        "primary_target": vis_target_name,
    }

    if mode in (ObservationMode.STATE, ObservationMode.HYBRID):
        hostiles: List[Dict[str, Any]] = []
        if state.objects:
            players = [o for o in state.objects if o.name == "DoomPlayer"]
            if players:
                player = players[0]
                for o in state.objects:
                    if o.name not in _IGNORED_LABELS and not o.name.startswith("Dead") and o.name != "DoomPlayer":
                        dx = o.position_x - player.position_x
                        dy = o.position_y - player.position_y
                        dist = math.hypot(dx, dy)
                        world_deg = math.degrees(math.atan2(dy, dx))
                        rel_deg = (world_deg - player.angle + 180.0) % 360.0 - 180.0
                        hostiles.append({"name": o.name, "dist": dist, "rel_deg": rel_deg})
        hostiles.sort(key=lambda h: h["dist"])
        detail["radar_hostiles"] = len(hostiles)
        detail["nearest_dist"] = None
        detail["nearest_rel_deg"] = None
        detail["nearest_name"] = None
        if hostiles:
            nearest = hostiles[0]
            detail.update(
                nearest_name=nearest["name"], nearest_dist=nearest["dist"],
                nearest_rel_deg=nearest["rel_deg"],
            )
        prompt = (
            f"VIZDOOM TACTICAL STATE [Ep {episode} | Step {step:02d}]\n"
            f"Health: {health:.0f} | Armor: {armor:.0f} | Ammo: {ammo:.0f} | "
            f"Radar hostiles: {len(hostiles)} | Visible: {len(vis_monsters)}\n"
        )
        if hostiles:
            prompt += (
                f"Nearest threat: {nearest['name']} "
                f"(dist {nearest['dist']:.1f}, rel-angle {nearest['rel_deg']:+.1f} deg; "
                f"crosshair {'LOCKED' if target_in_crosshair else f'{target_offset_x:+.1f}px off'})\n"
            )
        else:
            prompt += f"Primary visual: {vis_target_name} | Crosshair: {'LOCKED' if target_in_crosshair else 'searching'}\n"
        prompt += "Select the regret-minimal tactical combat maneuver."
    else:
        # VISION: no world coordinates, no radar geometry.
        detail["radar_hostiles"] = None
        prompt = (
            f"VIZDOOM VISUAL OBSERVATION [Ep {episode} | Step {step:02d}]\n"
            f"Health: {health:.0f} | Ammo: {ammo:.0f} | Visible hostiles: {len(vis_monsters)}\n"
            f"Primary target: {vis_target_name} "
            f"(bbox area {vis_target_size:.0f}px, offset {target_offset_x:+.1f}px; "
            f"crosshair {'LOCKED' if target_in_crosshair else 'not locked'})\n"
            "Select the regret-minimal tactical combat maneuver."
        )
    return prompt, detail


def scripted_baseline_action(detail: Dict[str, Any]) -> str:
    """Labeled scripted heuristic baseline (not a learned policy)."""
    if detail.get("crosshair_locked"):
        return "PRECISION_ATTACK"
    offset = float(detail.get("target_offset_x", 0.0) or 0.0)
    if offset < 0:
        return "SNAP_TURN_LEFT"
    if offset > 0:
        return "SNAP_TURN_RIGHT"
    if detail.get("radar_hostiles"):
        return "TACTICAL_ADVANCE"
    return "TACTICAL_ADVANCE"


class ViZDoomDecisionRunner:
    """Runs typed decision policies inside live ViZDoom simulations."""

    @staticmethod
    def run_simulation(
        engine: Optional[DecisionEngine] = None,
        scenario: str = "basic",
        num_episodes: int = 3,
        wad_path: Optional[str] = None,
        skill: int = 4,
        max_steps_per_episode: Optional[int] = None,
        frame_skip: int = 4,
        window_visible: bool = False,
        render_console: bool = True,
        observation_mode: str = "HYBRID",
        policy: str = "anydecision",
        seed: Optional[int] = None,
        min_confidence: Optional[float] = None,
        output_path: Optional[str] = None,
        utility_matrix: Optional[UtilityMatrix] = None,
        record_trajectory: bool = False,
        estimator_path: Optional[str] = None,
        hold_open: bool = False,
    ) -> ViZDoomScoreReport:
        if not VIZDOOM_AVAILABLE:
            raise ImportError(
                "ViZDoom is not installed. Install it via 'pip install vizdoom'."
            )
        mode = ObservationMode(str(observation_mode).upper())
        policy_kind = PolicyKind(str(policy).lower())
        if engine is None and policy_kind == PolicyKind.ANYDECISION:
            engine = DecisionEngine(model="mock")
        estimator = None
        if policy_kind == PolicyKind.LEARNED:
            from anydecision.games.doom_estimator import DoomEstimator

            default_est = Path(__file__).resolve().parents[2] / "artifacts" / "doom_estimator_vision.npz"
            estimator = DoomEstimator(estimator_path or str(default_est))
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))

        console = Console() if render_console else None
        game = vzd.DoomGame()

        resolved_wad = None
        if wad_path and os.path.exists(wad_path):
            resolved_wad = wad_path
        else:
            for p in DEFAULT_WAD_SEARCH_PATHS:
                if os.path.exists(p):
                    resolved_wad = p
                    break

        cfg_path = os.path.join(vzd.scenarios_path, f"{scenario}.cfg")
        if scenario.upper().startswith(("E1M", "E2M", "E3M", "E4M")) and resolved_wad:
            scenario = scenario.upper()
            game.set_doom_game_path(resolved_wad)
            game.set_doom_map(scenario)
            game.set_available_buttons([
                vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK,
                vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
                vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,
            ])
            game.set_available_game_variables([
                vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR,
                vzd.GameVariable.SELECTED_WEAPON_AMMO, vzd.GameVariable.KILLCOUNT,
                vzd.GameVariable.DEATHCOUNT, vzd.GameVariable.DAMAGE_TAKEN,
            ])
        elif os.path.exists(cfg_path):
            game.load_config(cfg_path)
        elif resolved_wad:
            game.set_doom_game_path(resolved_wad)
            game.set_doom_map("E1M1")
            scenario = "E1M1"
            game.set_available_buttons([
                vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK,
                vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
                vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,
            ])
            game.set_available_game_variables([
                vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR,
                vzd.GameVariable.SELECTED_WEAPON_AMMO, vzd.GameVariable.KILLCOUNT,
                vzd.GameVariable.DEATHCOUNT, vzd.GameVariable.DAMAGE_TAKEN,
            ])
        else:
            cfg_path = os.path.join(vzd.scenarios_path, "basic.cfg")
            game.load_config(cfg_path)
            scenario = "basic"

        for btn in [
            vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK,
            vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
            vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,
        ]:
            if btn not in game.get_available_buttons():
                game.add_available_button(btn)
        for var in [
            vzd.GameVariable.KILLCOUNT, vzd.GameVariable.HEALTH,
            vzd.GameVariable.ARMOR, vzd.GameVariable.SELECTED_WEAPON_AMMO,
            vzd.GameVariable.DEATHCOUNT, vzd.GameVariable.DAMAGE_TAKEN,
            vzd.GameVariable.ITEMCOUNT,
        ]:
            try:
                if var not in game.get_available_game_variables():
                    game.add_available_game_variable(var)
            except Exception:
                pass

        game.set_doom_skill(skill)
        if window_visible:
            game.set_window_visible(True)
        else:
            game.set_window_visible(False)
        game.set_labels_buffer_enabled(True)
        game.set_objects_info_enabled(True)
        game.init()

        buttons = game.get_available_buttons()
        button_names = [str(b).replace("Button.", "") for b in buttons]

        if console:
            console.print("\n[bold red]========================================================================[/bold red]")
            console.print("[bold red]     VIZDOOM INTEGRATION - TYPED DECISION POLICY EVALUATION            [/bold red]")
            console.print(f"[bold white]     Scenario: {scenario.upper()} | Policy: {policy_kind.value} | Obs: {mode.value}[/bold white]")
            console.print("[bold red]========================================================================[/bold red]\n")

        def make_vec(*active_btns: str) -> List[int]:
            v = [0] * len(buttons)
            for b in active_btns:
                if b in button_names:
                    v[button_names.index(b)] = 1
            return v

        action_map: Dict[str, List[int]] = {
            "PRECISION_ATTACK": make_vec("ATTACK"),
            "KITE_AND_FIRE": make_vec("MOVE_BACKWARD", "ATTACK"),
            "CIRCLE_STRAFE_LEFT": make_vec("MOVE_LEFT", "TURN_RIGHT", "ATTACK"),
            "CIRCLE_STRAFE_RIGHT": make_vec("MOVE_RIGHT", "TURN_LEFT", "ATTACK"),
            "DODGE_STRAFE_LEFT": make_vec("MOVE_LEFT"),
            "DODGE_STRAFE_RIGHT": make_vec("MOVE_RIGHT"),
            "TACTICAL_RETREAT": make_vec("MOVE_BACKWARD"),
            "TACTICAL_ADVANCE": make_vec("MOVE_FORWARD"),
            "ASSAULT_ADVANCE": make_vec("MOVE_FORWARD", "ATTACK"),
            "SNAP_TURN_LEFT": make_vec("TURN_LEFT"),
            "SNAP_TURN_RIGHT": make_vec("TURN_RIGHT"),
        }
        action_names = list(action_map.keys())
        matrix = utility_matrix or build_tactical_utility_matrix(action_names)
        criterion = victory_criterion_for(scenario)

        episodes_won = 0
        episodes_completed = 0
        total_kills = 0
        total_deaths = 0
        total_items = 0
        total_damage_taken = 0.0
        total_rewards: List[float] = []
        all_latencies: List[float] = []
        action_counts: Counter = Counter()
        compute_path_counts: Counter = Counter()
        total_decisions = 0
        total_abstentions = 0
        total_backend_calls = 0
        total_tokens = 0
        total_expected_utility = 0.0
        tokens_estimated_any = False
        trajectory_records: List[Dict[str, Any]] = []
        final_healths: List[float] = []
        final_armors: List[float] = []
        telemetry_logs: List[str] = []
        max_steps_per_ep = max_steps_per_episode if max_steps_per_episode is not None else (1200 if window_visible else 100)

        for ep in range(1, num_episodes + 1):
            if seed is not None:
                try:
                    game.set_seed(seed + ep)
                except Exception:
                    pass
            game.new_episode()
            ep_reward = 0.0
            ep_step = 0
            ep_end_health = 100.0
            ep_end_armor = 0.0
            ep_kills_start = int(_safe_game_var(game, vzd.GameVariable.KILLCOUNT, 0.0))
            ep_deaths_start = int(_safe_game_var(game, vzd.GameVariable.DEATHCOUNT, 0.0))
            ep_damage_start = _safe_game_var(game, vzd.GameVariable.DAMAGE_TAKEN, 0.0)
            ep_items_start = int(_safe_game_var(game, vzd.GameVariable.ITEMCOUNT, 0.0))

            if console:
                console.print(f"[bold cyan]>>> Starting ViZDoom Episode {ep}/{num_episodes}...[/bold cyan]")

            while not game.is_episode_finished() and ep_step < max_steps_per_ep:
                ep_step += 1
                total_decisions += 1
                state = game.get_state()
                if not state:
                    break

                prompt, detail = extract_observation(game, state, mode, ep, ep_step)
                ep_end_health = float(detail.get("health", 0.0))
                ep_end_armor = float(detail.get("armor", 0.0))

                t0 = time.perf_counter()
                if policy_kind == PolicyKind.RANDOM:
                    chosen_act = random.choice(action_names)
                    probs_repr = "{}"
                    abstained = False
                    backend_calls = 0
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                elif policy_kind == PolicyKind.SCRIPTED:
                    chosen_act = scripted_baseline_action(detail)
                    probs_repr = "{}"
                    abstained = False
                    backend_calls = 0
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                elif policy_kind == PolicyKind.LEARNED:
                    assert estimator is not None
                    est_probs = estimator.predict_proba(features_from_detail(detail))
                    chosen_act = max(est_probs, key=est_probs.get)  # type: ignore[arg-type]
                    if chosen_act not in action_map:
                        raise AssertionError(
                            f"Estimator predicted {chosen_act!r} outside the action space."
                        )
                    probs_repr = f"maxP={max(est_probs.values()):.2f}"
                    abstained = False
                    backend_calls = 0
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                else:
                    assert engine is not None
                    q = Question.choice(prompt, choices=list(matrix.states))
                    decision = engine.decide_adaptive(
                        q, utility_matrix=matrix, min_confidence=min_confidence
                    )
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    # Model beliefs: REAL candidate-conditional distribution, never hardcoded.
                    state_probs = dict(decision.probabilities)
                    # The executed action MUST come from Decision.selected_action.
                    if decision.abstained or not decision.selected_action:
                        total_abstentions += 1
                        chosen_act = _FALLBACK_SAFE_ACTION
                        abstention_note = f"abstained({decision.reason})->safe-fallback"
                    else:
                        chosen_act = decision.selected_action
                        abstention_note = "selected"
                        # Internal consistency: executed action must equal selected action.
                        if chosen_act not in action_map:
                            raise AssertionError(
                                f"Selected action {chosen_act!r} not in executed action space."
                            )
                    probs_repr = (
                        f"maxP={max(state_probs.values()) if state_probs else 0.0:.2f}"
                    )
                    backend_calls = decision.backend_calls
                    total_backend_calls += backend_calls
                    if decision.optimal_action_utility is not None:
                        total_expected_utility += float(decision.optimal_action_utility)
                    total_tokens += decision.tokens_processed
                    tokens_estimated_any = tokens_estimated_any or bool(decision.tokens_estimated)
                    compute_path_counts["+".join(decision.compute_path)] += 1
                    detail["decision_answer"] = decision.answer
                    detail["abstention_note"] = abstention_note

                all_latencies.append(lat_ms)
                action_counts[chosen_act] += 1

                # Execute the selected action vector in the ViZDoom engine.
                action_vector = action_map.get(chosen_act, [0] * len(buttons))
                step_reward = game.make_action(action_vector, frame_skip)
                ep_reward += step_reward
                ep_kills_now = int(_safe_game_var(game, vzd.GameVariable.KILLCOUNT, 0.0))

                if record_trajectory:
                    trajectory_records.append({
                        "episode": ep, "step": ep_step,
                        "scenario": scenario, "policy": policy_kind.value,
                        "observation_mode": mode.value, "seed": seed,
                        "features": features_from_detail(detail),
                        "action": chosen_act,
                        "step_reward": float(step_reward),
                        "kills_total": int(ep_kills_now - ep_kills_start),
                    })

                if window_visible:
                    time.sleep(0.028)

                log_line = (
                    f"Ep {ep} | Step {ep_step:02d} | Action: {chosen_act:<19} | "
                    f"Obs: {mode.value} | {probs_repr} | R: {step_reward:+.1f}"
                )
                telemetry_logs.append(log_line)
                if console and (ep_step % 2 == 1 or step_reward > 0):
                    console.print(f"  {log_line}")

            # Authoritative per-episode counters (game variables ONLY).
            ep_kills = max(0, int(_safe_game_var(game, vzd.GameVariable.KILLCOUNT, 0.0)) - ep_kills_start)
            ep_deaths = max(0, int(_safe_game_var(game, vzd.GameVariable.DEATHCOUNT, 0.0)) - ep_deaths_start)
            ep_damage = max(0.0, _safe_game_var(game, vzd.GameVariable.DAMAGE_TAKEN, 0.0) - ep_damage_start)
            ep_items = max(0, int(_safe_game_var(game, vzd.GameVariable.ITEMCOUNT, 0.0)) - ep_items_start)
            total_items += ep_items
            if ep_step >= max_steps_per_ep or game.is_episode_finished():
                episodes_completed += 1
            total_kills += ep_kills
            total_deaths += ep_deaths
            total_damage_taken += ep_damage
            total_rewards.append(ep_reward)
            final_healths.append(ep_end_health)
            final_armors.append(ep_end_armor)
            died = ep_deaths > 0
            won = is_victory(scenario, ep_kills, died)
            if won:
                episodes_won += 1
            if console:
                status = "OBJECTIVE MET" if won else "EPISODE END"
                console.print(
                    f"[bold green]>> Episode {ep} {status}: kills={ep_kills} deaths={ep_deaths} "
                    f"reward={ep_reward:.1f}[/bold green]\n"
                )

        if hold_open and window_visible:
            try:
                if console:
                    console.print("[bold yellow]Holding game window open — press Enter in this terminal to close.[/bold yellow]")
                input()
            except (EOFError, KeyboardInterrupt):
                pass
        game.close()

        mean_lat = float(np.mean(all_latencies)) if all_latencies else 0.0
        throughput = float(1000.0 / max(1e-6, mean_lat))
        mean_rew = float(np.mean(total_rewards)) if total_rewards else 0.0
        win_rate = (episodes_won / max(1, num_episodes)) * 100.0
        completion_rate = (episodes_completed / max(1, num_episodes)) * 100.0
        backend_name = "unknown"
        model_name = "unknown"
        model_rev = "main"
        if engine is not None:
            try:
                backend_name = engine.metadata.backend_name
                model_name = engine.metadata.model_name
                model_rev = engine.metadata.model_revision
            except Exception:
                pass

        report = ViZDoomScoreReport(
            scenario=scenario,
            victory_criterion=criterion,
            observation_mode=mode.value,
            policy=policy_kind.value,
            episodes=num_episodes,
            episodes_won=episodes_won,
            win_rate=win_rate,
            completion_rate=completion_rate,
            total_kills=total_kills,
            total_deaths=total_deaths,
            total_items=total_items,
            mean_final_health=float(np.mean(final_healths)) if final_healths else 0.0,
            mean_final_armor=float(np.mean(final_armors)) if final_armors else 0.0,
            total_expected_utility=float(total_expected_utility),
            total_damage_taken=float(total_damage_taken),
            total_reward=float(sum(total_rewards)),
            mean_reward=mean_rew,
            total_decisions=total_decisions,
            total_abstentions=total_abstentions,
            total_backend_calls=total_backend_calls,
            total_tokens=total_tokens,
            tokens_estimated=tokens_estimated_any if policy_kind == PolicyKind.ANYDECISION else True,
            mean_latency_ms=mean_lat,
            p50_latency_ms=_percentile(all_latencies, 50),
            p95_latency_ms=_percentile(all_latencies, 95),
            p99_latency_ms=_percentile(all_latencies, 99),
            decisions_per_sec=throughput,
            action_distribution=dict(action_counts),
            compute_path_distribution=dict(compute_path_counts),
            seed=seed,
            backend=backend_name,
            model=model_name,
            model_revision=model_rev,
            commit=_git_commit(),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            telemetry_log=telemetry_logs,
            trajectory=trajectory_records,
        )

        if output_path:
            report.save_json(output_path)

        if console:
            console.print("\n[bold red]========================================================================[/bold red]")
            console.print("[bold red]            VIZDOOM EVALUATION SCORECARD (FACTUAL)                        [/bold red]")
            console.print("[bold red]========================================================================[/bold red]")
            table = Table(box=DOUBLE)
            table.add_column("Benchmark Metric", style="bold white")
            table.add_column("Evaluation Result", style="bold yellow")
            table.add_row("SCENARIO", scenario.upper())
            table.add_row("POLICY", policy_kind.value)
            table.add_row("OBSERVATION MODE", mode.value)
            table.add_row("VICTORY CRITERION", criterion)
            table.add_row("EPISODES (won/total)", f"{episodes_won}/{num_episodes} ({win_rate:.1f}%)")
            table.add_row("COMPLETION RATE", f"{completion_rate:.1f}%")
            table.add_row("KILLS (KILLCOUNT)", str(total_kills))
            table.add_row("DEATHS (DEATHCOUNT)", str(total_deaths))
            table.add_row("ITEMS (ITEMCOUNT)", str(total_items))
            table.add_row("DAMAGE TAKEN", f"{total_damage_taken:.1f}")
            table.add_row("MEAN REWARD", f"{mean_rew:+.2f}")
            table.add_row("MEAN LATENCY", f"{mean_lat:.2f} ms")
            table.add_row("P50/P95/P99 LATENCY", f"{report.p50_latency_ms:.1f}/{report.p95_latency_ms:.1f}/{report.p99_latency_ms:.1f} ms")
            table.add_row("DECISIONS", str(total_decisions))
            table.add_row("ABSTENTIONS", str(total_abstentions))
            table.add_row("BACKEND CALLS", str(total_backend_calls))
            table.add_row("TOKENS", f"{total_tokens} ({'estimated' if report.tokens_estimated else 'exact'})")
            table.add_row("BACKEND", f"{backend_name} / {model_name}")
            table.add_row("SEED", str(seed))
            console.print(table)

        return report
