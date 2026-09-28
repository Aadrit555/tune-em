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
]

