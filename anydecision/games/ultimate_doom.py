"""Real Ultimate DOOM WAD Parser & Authentic Level Scoring Runtime.

Parses authentic IWAD binary data from the official Ultimate DOOM game files (1993/1995 id Software).
Extracts real spatial monster entities, weapons, health pickups, and hazards directly from
the WAD's binary lumps (THINGS, SECTORS) and runs the anydecision Expected Utility runtime
to score real combat decisions.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel
from rich.box import DOUBLE
from rich.console import Console
from rich.table import Table

from anydecision.core.engine import DecisionEngine

# Authentic Doom entity type mappings (id Software Doom Bible / engine source)
DOOM_ENTITY_DEFS: Dict[int, Dict[str, Any]] = {
    # Player spawns
    1: {"name": "Player 1 Start", "category": "player", "hp": 100, "threat": "NONE"},
    # Monsters
    3004: {"name": "Zombieman", "category": "monster", "hp": 20, "threat": "LOW", "dmg": 8},
    9: {"name": "Shotgun Guy", "category": "monster", "hp": 30, "threat": "MEDIUM", "dmg": 18},
    3001: {"name": "Imp", "category": "monster", "hp": 60, "threat": "MEDIUM", "dmg": 12},
    3002: {"name": "Pinky Demon", "category": "monster", "hp": 150, "threat": "MEDIUM", "dmg": 16},
    58: {"name": "Spectre", "category": "monster", "hp": 150, "threat": "MEDIUM", "dmg": 16},
    3006: {"name": "Lost Soul", "category": "monster", "hp": 60, "threat": "LOW", "dmg": 10},
    3005: {"name": "Cacodemon", "category": "monster", "hp": 400, "threat": "HIGH", "dmg": 22},
    3003: {"name": "Baron of Hell", "category": "monster", "hp": 1000, "threat": "EXTREME", "dmg": 30},
    16: {"name": "Cyberdemon", "category": "monster", "hp": 4000, "threat": "LETHAL", "dmg": 50},
    7: {"name": "Spider Mastermind", "category": "monster", "hp": 3000, "threat": "LETHAL", "dmg": 45},
    # Weapons
    2001: {"name": "Shotgun", "category": "weapon", "base_dmg": 80},
    2002: {"name": "Chaingun", "category": "weapon", "base_dmg": 65},
    2003: {"name": "Rocket Launcher", "category": "weapon", "base_dmg": 200},
    2004: {"name": "Plasma Rifle", "category": "weapon", "base_dmg": 120},
    2005: {"name": "Chainsaw", "category": "weapon", "base_dmg": 100},
    2006: {"name": "BFG9000", "category": "weapon", "base_dmg": 450},
    82: {"name": "Super Shotgun", "category": "weapon", "base_dmg": 180},
    # Pickups
    2011: {"name": "Stimpack", "category": "pickup", "heal": 10},
    2012: {"name": "Medikit", "category": "pickup", "heal": 25},
    2013: {"name": "Soul Sphere", "category": "pickup", "heal": 100},
    2018: {"name": "Green Armor", "category": "armor", "armor": 100},
    2019: {"name": "Blue MegaArmor", "category": "armor", "armor": 200},
    2048: {"name": "Box of Bullets", "category": "ammo", "ammo_type": "bullets", "amount": 50},
    2049: {"name": "Box of Shells", "category": "ammo", "ammo_type": "shells", "amount": 20},
    2046: {"name": "Box of Rockets", "category": "ammo", "ammo_type": "rockets", "amount": 5},
    2047: {"name": "Energy Cell Pack", "category": "ammo", "ammo_type": "cells", "amount": 100},
    # Hazards
    2035: {"name": "Explosive Barrel", "category": "hazard", "aoe_dmg": 150},
}

# Standard installation paths for Ultimate DOOM
DEFAULT_WAD_SEARCH_PATHS = [
    r"C:\Users\Aadrit\Downloads\The-Ultimate-DOOM-SteamRIP.com\DOOM\DOOM.WAD",
    r"C:\Program Files (x86)\Steam\steamapps\common\Ultimate Doom\base\DOOM.WAD",
    r"C:\GOG Games\DOOM\DOOM.WAD",
    os.path.expanduser(r"~\Downloads\The-Ultimate-DOOM-SteamRIP.com\DOOM\DOOM.WAD"),
]


class RealWadEntity(BaseModel):
    """A physical entity extracted from a real DOOM.WAD level."""
    type_id: int
    name: str
    category: str
    x: float
    y: float
    distance_to_player: float
    angle: int
    flags: int
    hp: int = 0
    threat: str = "NONE"
    base_dmg: int = 0


class RealDoomMap(BaseModel):
    """A fully parsed real map from Ultimate DOOM."""
    map_code: str
    title: str
    player_spawn: Tuple[float, float]
    monsters: List[RealWadEntity]
    weapons: List[RealWadEntity]
    pickups: List[RealWadEntity]
    hazards: List[RealWadEntity]
    total_entities: int


MAP_TITLES = {
    "E1M1": "Hangar",
    "E1M2": "Nuclear Plant",
    "E1M3": "Toxin Refinery",
    "E1M4": "Command Control",
    "E1M5": "Phobos Lab",
    "E1M6": "Central Processing",
    "E1M7": "Computer Station",
    "E1M8": "Phobos Anomaly",
    "E1M9": "Military Base",
    "E2M1": "Deimos Anomaly",
    "E2M8": "Tower of Babel",
    "E3M1": "Hell Keep",
    "E3M8": "Dis",
    "E4M1": "Hell Beneath",
    "E4M2": "Perfect Hatred",
}


class UltimateDoomWadParser:
    """Extracts genuine level geometry, monster spawns, and items from DOOM.WAD."""

    def __init__(self, wad_path: Optional[str] = None) -> None:
        self.wad_path = self._resolve_wad_path(wad_path)
        self.directory: Dict[str, Tuple[int, int]] = {}
        self._load_wad_directory()

    def _resolve_wad_path(self, path: Optional[str]) -> str:
        if path:
            if os.path.exists(path):
                return path
            raise FileNotFoundError(f"Specified DOOM WAD file does not exist: {path}")
        for p in DEFAULT_WAD_SEARCH_PATHS:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            "Could not locate DOOM.WAD in default search paths. Please specify --wad path."
        )

    def _load_wad_directory(self) -> None:
        """Read the binary lump directory of the IWAD."""
        with open(self.wad_path, "rb") as f:
            magic, num_lumps, dir_offset = struct.unpack("<4sII", f.read(12))
            if magic not in (b"IWAD", b"PWAD"):
                raise ValueError(f"Invalid WAD magic: {magic}")
            f.seek(dir_offset)
            for _ in range(num_lumps):
                pos, size, name = struct.unpack("<II8s", f.read(16))
                clean_name = name.rstrip(b"\x00").decode("latin-1")
                self.directory[clean_name] = (pos, size)

    def list_maps(self) -> List[str]:
        """List all valid maps present in this WAD (e.g. E1M1 through E4M9)."""
        maps = []
        for name in sorted(self.directory.keys()):
            if len(name) == 4 and name.startswith(("E1M", "E2M", "E3M", "E4M")):
                maps.append(name)
        return maps

    def parse_map(self, map_code: str = "E1M1", skill_level: int = 3) -> RealDoomMap:
        """Parse real THINGS lump for the designated map and skill level.

        Skill levels:
          1 = I'm Too Young to Die
          2 = Hey, Not Too Rough
          3 = Hurt Me Plenty (Medium, default)
          4 = Ultra-Violence (Hard)
          5 = Nightmare!
        """
        map_code = map_code.upper()
        if map_code not in self.directory:
            raise KeyError(f"Map '{map_code}' not found in WAD.")

        with open(self.wad_path, "rb") as f:
            magic, num_lumps, dir_offset = struct.unpack("<4sII", f.read(12))
            f.seek(dir_offset)
            lumps = [struct.unpack("<II8s", f.read(16)) for _ in range(num_lumps)]

        # Find the index of the map marker lump
        map_idx = -1
        for i, (_, _, name) in enumerate(lumps):
            if name.rstrip(b"\x00").decode("latin-1") == map_code:
                map_idx = i
                break

        if map_idx == -1:
            raise ValueError(f"Map lump {map_code} missing from lump list.")

        # Next lump in standard DOOM WAD structure is THINGS
        things_pos, things_size, things_name = lumps[map_idx + 1]
        cname = things_name.rstrip(b"\x00").decode("latin-1")
        if cname != "THINGS":
            raise ValueError(f"Expected THINGS lump after {map_code}, found {cname}")

        with open(self.wad_path, "rb") as f:
            f.seek(things_pos)
            data = f.read(things_size)

        num_things = len(data) // 10
        player_x, player_y = 0.0, 0.0

        raw_entities: List[Tuple[int, int, float, float, int]] = []
        for j in range(num_things):
            x, y, angle, type_id, flags = struct.unpack("<hhhhh", data[j * 10 : (j + 1) * 10])

            # Check skill level flags:
            # Bit 0 (1): Skill 1 & 2
            # Bit 1 (2): Skill 3
            # Bit 2 (4): Skill 4 & 5
            skill_bit = 1 if skill_level in (1, 2) else (2 if skill_level == 3 else 4)
            if not (flags & skill_bit) and type_id != 1:
                continue

            if type_id == 1:  # Player 1 Start
                player_x, player_y = float(x), float(y)

            raw_entities.append((type_id, flags, float(x), float(y), angle))

        monsters: List[RealWadEntity] = []
        weapons: List[RealWadEntity] = []
        pickups: List[RealWadEntity] = []
        hazards: List[RealWadEntity] = []

        for tid, flg, x, y, ang in raw_entities:
            dist = math.sqrt((x - player_x) ** 2 + (y - player_y) ** 2) / 32.0  # Convert map units to meters
            info = DOOM_ENTITY_DEFS.get(tid, None)
            if not info:
                continue

            entity = RealWadEntity(
                type_id=int(tid),
                name=str(info["name"]),
                category=str(info["category"]),
                x=float(x),
                y=float(y),
                distance_to_player=float(dist),
                angle=int(ang),
                flags=int(flg),
                hp=int(info.get("hp", 0) or 0),
                threat=str(info.get("threat", "NONE")),
                base_dmg=int(info.get("base_dmg", info.get("dmg", 0)) or 0),
            )

            if entity.category == "monster":
                monsters.append(entity)
            elif entity.category == "weapon":
                weapons.append(entity)
            elif entity.category in ("pickup", "armor", "ammo"):
                pickups.append(entity)
            elif entity.category == "hazard":
                hazards.append(entity)

        # Sort monsters and pickups by distance from the player
        monsters.sort(key=lambda m: m.distance_to_player)
        pickups.sort(key=lambda p: p.distance_to_player)
        hazards.sort(key=lambda h: h.distance_to_player)

        title = f"{map_code}: {MAP_TITLES.get(map_code, 'Unknown Area')}"
        return RealDoomMap(
            map_code=map_code,
            title=title,
            player_spawn=(player_x, player_y),
            monsters=monsters,
            weapons=weapons,
            pickups=pickups,
            hazards=hazards,
            total_entities=num_things,
        )


class RealDoomScoreReport(BaseModel):
    """Factual end-of-level report: WAD spawn census vs live-engine counters.

    demons_slain / pickups_collected are authoritative ViZDoom KILLCOUNT /
    ITEMCOUNT deltas. status is the scenario victory proxy ("OBJECTIVE MET"
    when the engine run met its victory criterion, else "EPISODE END") —
    NOT a claim of full level clearance (exit-switch state is not tracked).
    """
    map_code: str
    map_title: str
    skill_level: str
    total_demons_spawned: int
    demons_slain: int
    kill_percentage: float
    total_pickups_spawned: int
    pickups_collected: int
    item_percentage: float
    final_health: int
    final_armor: int
    total_decisions: int
    mean_decision_latency_ms: float
    throughput_decisions_per_sec: float
    accumulated_expected_utility: float
    status: str
    telemetry_log: List[str]


class RealDoomEvaluator:
    """Executes anydecision against real Ultimate DOOM level entities."""

    @staticmethod
    def run_map_evaluation(
        engine: DecisionEngine,
        map_code: str = "E1M1",
        skill_level: int = 3,
        wad_path: Optional[str] = None,
        render_console: bool = True,
        num_episodes: int = 1,
        max_steps_per_episode: int = 400,
        seed: Optional[int] = None,
        observation_mode: str = "HYBRID",
        policy: str = "anydecision",
        output_path: Optional[str] = None,
    ) -> RealDoomScoreReport:
        console = Console() if render_console else None
        try:
            parser = UltimateDoomWadParser(wad_path)
        except FileNotFoundError as e:
            searched = ", ".join(DEFAULT_WAD_SEARCH_PATHS)
            raise FileNotFoundError(
                f"{e} Searched paths: {searched}. Provide a genuine WAD via "
                "--wad /path/to/DOOM.WAD. Evaluation aborted: no synthetic "
                "substitute is used for real-doom."
            ) from e
        doom_map = parser.parse_map(map_code=map_code, skill_level=skill_level)

        skill_names = {
            1: "I'm Too Young to Die",
            2: "Hey, Not Too Rough",
            3: "Hurt Me Plenty",
            4: "Ultra-Violence",
            5: "Nightmare!",
        }
        skill_str = skill_names.get(skill_level, "Hurt Me Plenty")

        if console:
            console.print("\n[bold red]========================================================================[/bold red]")
            console.print(f"[bold red]   ULTIMATE DOOM IWAD ENGINE - REAL LEVEL EVALUATOR: {doom_map.title.upper()} [/bold red]")
            console.print(f"[bold white]   WAD Source: {parser.wad_path}[/bold white]")
            console.print(f"[bold yellow]   Skill Level: {skill_str} | Total WAD Entities: {doom_map.total_entities}[/bold yellow]")
            console.print(f"[bold white]   Hostiles: {len(doom_map.monsters)} | Pickups: {len(doom_map.pickups)} | Barrels: {len(doom_map.hazards)}[/bold white]")
            console.print("[bold red]========================================================================[/bold red]\n")

        # REAL evaluation: the WAD census above is the genuine initial state.
        # All combat dynamics from here on come from the live ViZDoom engine
        # (real physics, real damage, authoritative KILLCOUNT/DEATHCOUNT/
        # ITEMCOUNT). No damage, kills, or pickups are simulated or fabricated.
        from anydecision.games.vizdoom_env import ViZDoomDecisionRunner, is_vizdoom_available

        if not is_vizdoom_available():
            raise ImportError(
                "real-doom requires the 'vizdoom' package for genuine engine execution. "
                "Install it via 'pip install vizdoom'. Aborting rather than "
                "substituting a synthetic simulator."
            )

        viz_report = ViZDoomDecisionRunner.run_simulation(
            engine=engine,
            scenario=doom_map.map_code,
            num_episodes=num_episodes,
            wad_path=parser.wad_path,
            skill=skill_level,
            max_steps_per_episode=max_steps_per_episode,
            frame_skip=4,
            window_visible=False,
            render_console=False,
            observation_mode=observation_mode,
            policy=policy,
            seed=seed,
        )

        demons_slain = viz_report.total_kills
        pickups_collected = viz_report.total_items
        health = int(round(viz_report.mean_final_health))
        armor = int(round(viz_report.mean_final_armor))
        accumulated_eu = viz_report.total_expected_utility
        latencies = [viz_report.mean_latency_ms] if viz_report.mean_latency_ms else []
        telemetry_logs = list(viz_report.telemetry_log)
        turn = viz_report.total_decisions

        # Compute factual end-of-level scorecard from live engine counters.
        # demons_slain / pickups_collected above are authoritative ViZDoom
        # KILLCOUNT / ITEMCOUNT deltas; no damage was fabricated.
        total_spawned_monsters = len(doom_map.monsters)
        total_spawned_pickups = len(doom_map.pickups)
        kill_pct = (demons_slain / max(1, total_spawned_monsters)) * 100.0
        item_pct = (pickups_collected / max(1, total_spawned_pickups)) * 100.0
        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        throughput = float(1000.0 / max(1e-6, mean_lat))
        # Scenario-defined outcome: WAD-map proxy is kills > 0 or survival.
        # "LEVEL CLEARED" would require exit-switch tracking, which the engine
        # run does not provide; report the proxy honestly.
        status_str = "OBJECTIVE MET" if (viz_report.episodes_won > 0) else "EPISODE END"

        report = RealDoomScoreReport(
            map_code=doom_map.map_code,
            map_title=doom_map.title,
            skill_level=skill_str,
            total_demons_spawned=total_spawned_monsters,
            demons_slain=demons_slain,
            kill_percentage=kill_pct,
            total_pickups_spawned=total_spawned_pickups,
            pickups_collected=pickups_collected,
            item_percentage=item_pct,
            final_health=health,
            final_armor=armor,
            total_decisions=turn,
            mean_decision_latency_ms=mean_lat,
            throughput_decisions_per_sec=throughput,
            accumulated_expected_utility=accumulated_eu,
            status=status_str,
            telemetry_log=telemetry_logs,
        )

        if console:
            console.print("\n[bold red]========================================================================[/bold red]")
            console.print(f"[bold red]             ULTIMATE DOOM END-OF-LEVEL SCORECARD: {doom_map.map_code} [/bold red]")
            console.print("[bold red]========================================================================[/bold red]")

            score_table = Table(box=DOUBLE)
            score_table.add_column("Score Metric", style="bold white")
            score_table.add_column("Level Result", style="bold yellow")

            score_table.add_row(
                "KILLS (engine KILLCOUNT)",
                f"{demons_slain} / {total_spawned_monsters} spawned ({kill_pct:.1f}%)",
            )
            score_table.add_row(
                "ITEMS (engine ITEMCOUNT)",
                f"{pickups_collected} / {total_spawned_pickups} spawned ({item_pct:.1f}%)",
            )
            score_table.add_row(
                "DEATHS (engine DEATHCOUNT)",
                str(viz_report.total_deaths),
            )
            score_table.add_row(
                "FINAL HEALTH / ARMOR (mean)",
                f"{health}% / {armor}%",
            )
            score_table.add_row(
                "MEAN DECISION LATENCY",
                f"{mean_lat:.2f} ms ({throughput:.1f} decisions / sec)",
            )
            score_table.add_row(
                "TOTAL EXPECTED UTILITY",
                f"{accumulated_eu:+.1f} EU",
            )
            score_table.add_row(
                "OUTCOME",
                status_str,
            )
            console.print(score_table)

        if output_path:
            artifact = {
                "experiment": "real_doom",
                "map_code": doom_map.map_code,
                "map_title": doom_map.title,
                "wad_path": parser.wad_path,
                "wad_spawned_monsters": total_spawned_monsters,
                "wad_spawned_pickups": total_spawned_pickups,
                "seed": seed,
                "policy": policy,
                "observation_mode": observation_mode,
                "vizdoom": viz_report.to_dict(),
            }
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

        return report
