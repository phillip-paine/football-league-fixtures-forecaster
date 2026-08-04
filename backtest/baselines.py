"""
Baseline forecasts to compare the model against, per
forecast_engine_design_notes.md ("Benchmarks to beat: naive baselines,
Elo alone, closing bookmaker odds").

Only the naive frequency baseline is implemented here — it's fully
self-contained (nothing but historical outcome counts, no extra data
source). Elo (clubelo.com) and bookmaker-odds baselines both need data
sources that aren't onboarded/read yet — `features/` has no odds-reading
function despite `match_odds` existing in the schema per build_log.md,
and clubelo isn't wired in per the design notes' source table — so both
are stubbed with a clear NotImplementedError rather than guessed at.
Wire these in once there's a real read path for either, rather than
silently faking a "market" comparison against invented numbers.
"""

import numpy as np

from .rps import outcome_idx_from_goals


def naive_frequency_baseline(
        train_home_goals: np.ndarray, train_away_goals: np.ndarray, n_matches_to_predict: int
) -> np.ndarray:
    """
    A constant [P(home), P(draw), P(away)] baseline, tiled for every
    match being scored — the training window's own historical outcome
    frequency, using zero team-specific information. This is the "can
    the model even beat doing nothing" floor.
    """
    outcome_idx = outcome_idx_from_goals(train_home_goals, train_away_goals)
    counts = np.bincount(outcome_idx, minlength=3)
    if counts.sum() == 0:
        raise ValueError("no training matches to compute a naive baseline from")
    freqs = counts / counts.sum()
    return np.tile(freqs, (n_matches_to_predict, 1))


def market_implied_probs(*args, **kwargs):
    raise NotImplementedError(
        "Market-odds baseline needs an odds-reading function against "
        "match_odds, which features/ doesn't expose yet (deferred per "
        "build_log.md's covariate ordering). Wire this in once odds "
        "ingestion has a read path, rather than guessing at the schema "
        "here."
    )


def elo_baseline(*args, **kwargs):
    raise NotImplementedError(
        "Elo baseline needs clubelo.com onboarding, which isn't done yet "
        "per forecast_engine_design_notes.md's data-source table."
    )
