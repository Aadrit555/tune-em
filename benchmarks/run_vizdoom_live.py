"""Live Interactive DOOM Gameplay Runner using anydecision Expected Utility Runtime.

Pops up a full 800x600 window with HUD and sound, placing you directly inside
the ViZDoom arena where waves of demons charge in real time.
(ViZDoom integration; not affiliated with or verified by the Farama Foundation.)
"""

from __future__ import annotations

import argparse
from anydecision.games.vizdoom_env import ViZDoomDecisionRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Interactive ViZDoom AI Combat Runner")
    parser.add_argument("scenario", nargs="?", default="defend_the_center", help="ViZDoom scenario name or authentic IWAD map (e.g. defend_the_center, deadly_corridor, basic, E1M1)")
    parser.add_argument("episodes", nargs="?", type=int, default=2, help="Number of episodes to run (default: 2)")
    parser.add_argument("--skill", "-k", type=int, default=4, help="Doom skill level: 1 (Easy) to 5 (Nightmare), default 4 (Ultra-Violence)")
    parser.add_argument("--steps", "-s", type=int, default=1200, help="Max decision steps per episode (default: 1200)")
    parser.add_argument("--wad", "-w", type=str, default=None, help="Optional custom DOOM.WAD path")

    args = parser.parse_args()

    skill_names = {
        1: "I'm Too Young To Die",
        2: "Hey, Not Too Rough",
        3: "Hurt Me Plenty",
        4: "Ultra-Violence",
        5: "Nightmare",
    }
    skill_str = skill_names.get(args.skill, f"Level {args.skill}")

    print("\n=======================================================")
    print("  LAUNCHING LIVE VIZDOOM COMBAT: " + args.scenario.upper())
    print(f"  Skill: {skill_str} (Level {args.skill}) | Max Steps: {args.steps}")
    print("  Audio: ENABLED | Resolution: 800x600 | Engine: ZDoom")
    print("=======================================================\n")

    ViZDoomDecisionRunner.run_simulation(
        scenario=args.scenario,
        num_episodes=args.episodes,
        skill=args.skill,
        max_steps_per_episode=args.steps,
        wad_path=args.wad,
        window_visible=True,
        render_console=True,
    )


if __name__ == "__main__":
    main()
