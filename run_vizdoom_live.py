"""Live Interactive DOOM Gameplay Runner using anydecision Expected Utility Runtime.

Pops up a full 800x600 window with HUD and sound, placing you directly inside
the Farama Foundation ViZDoom arena where waves of demons charge in real time.
"""

import sys
import time
from anydecision.games.vizdoom_env import ViZDoomDecisionRunner

def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "defend_the_center"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    print("\n=======================================================")
    print("  LAUNCHING LIVE VIZDOOM COMBAT: " + scenario.upper())
    print("  Audio: ENABLED | Resolution: 800x600 | Engine: ZDoom")
    print("=======================================================\n")
    
    ViZDoomDecisionRunner.run_simulation(
        scenario=scenario,
        num_episodes=episodes,
        window_visible=True,
        render_console=True,
    )

if __name__ == "__main__":
    main()
