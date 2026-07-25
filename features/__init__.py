"""
features — the data-access / feature-extraction layer.

Bridges the EAV-style SQLite schema (schema.sql) to the plain numpy
arrays the model stage (Dixon-Coles hierarchical Poisson, PyMC) needs:
team indices, division indices, goals, and recency decay weights.

This layer does not know anything about PyMC/Stan and the model layer
does not know anything about SQL — the MatchDataset dataclass in
`features.matches` is the contract between them.
"""

from .matches import MatchDataset, load_training_matches, load_fixtures
from .indices import IndexMap, build_team_index, build_division_index
from .decay import decay_weights
from .validate import check_team_season_consistency
from .db import connect

__all__ = [
    "MatchDataset",
    "load_training_matches",
    "load_fixtures",
    "IndexMap",
    "build_team_index",
    "build_division_index",
    "decay_weights",
    "check_team_season_consistency",
    "connect",
]