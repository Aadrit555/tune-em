"""Tactical gaming environments and decision benchmarks (DOOM combat simulation)."""

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


