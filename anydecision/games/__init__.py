"""Gaming environments and decision benchmarks.

anydecision.games.doom is a SYNTHETIC toy-combat simulator for deterministic
unit tests and CI (not real DOOM). Genuine evaluation lives in
anydecision.games.vizdoom_env (live ViZDoom engine) and
anydecision.games.ultimate_doom (genuine WAD census + live engine runs).
"""

from anydecision.games.doom import (
    DoomActionOutcome,
    DoomCombatBenchmarkRunner,
    DoomEnemy,
    DoomGameState,
    DoomScenarioEnvironment,
    DoomTacticalAgent,
)

from anydecision.games.ultimate_doom import (
    RealDoomEvaluator,
    RealDoomMap,
    RealDoomScoreReport,
    RealWadEntity,
    UltimateDoomWadParser,
)

from anydecision.games.vizdoom_env import (
    ViZDoomDecisionRunner,
    ViZDoomScoreReport,
    is_vizdoom_available,
)

__all__ = [
    "DoomActionOutcome",
    "DoomCombatBenchmarkRunner",
    "DoomEnemy",
    "DoomGameState",
    "DoomScenarioEnvironment",
    "DoomTacticalAgent",
    "RealDoomEvaluator",
    "RealDoomMap",
    "RealDoomScoreReport",
    "RealWadEntity",
    "UltimateDoomWadParser",
    "ViZDoomDecisionRunner",
    "ViZDoomScoreReport",
    "is_vizdoom_available",
]


