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

from anydecision.games.doom_estimator import (
    DoomEstimator,
    label_trajectories,
    save_estimator,
    train_estimator,
)
from anydecision.games.vizdoom_env import (
    FEATURE_ORDER,
    ViZDoomDecisionRunner,
    ViZDoomScoreReport,
    features_from_detail,
    is_vizdoom_available,
)

__all__ = [
    "DoomActionOutcome",
    "DoomCombatBenchmarkRunner",
    "DoomEnemy",
    "DoomEstimator",
    "DoomGameState",
    "DoomScenarioEnvironment",
    "DoomTacticalAgent",
    "FEATURE_ORDER",
    "RealDoomEvaluator",
    "RealDoomMap",
    "RealDoomScoreReport",
    "RealWadEntity",
    "UltimateDoomWadParser",
    "ViZDoomDecisionRunner",
    "ViZDoomScoreReport",
    "features_from_detail",
    "is_vizdoom_available",
    "label_trajectories",
    "save_estimator",
    "train_estimator",
]


