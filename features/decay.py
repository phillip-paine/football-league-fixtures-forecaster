"""
Exponential time-decay weighting (design notes, Phase 1 recency
handling). The half-life itself is a backtesting question and is
never hard-coded here — every call takes it as a parameter so the
walk-forward harness can sweep it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np


def decay_weights(
        match_dates: np.ndarray,
        as_of_date: dt.date,
        half_life_days: float,
) -> np.ndarray:
    """weight = 0.5 ** (age_days / half_life_days), age measured back
    from as_of_date.

    match_dates: array of numpy.datetime64[D] (or castable to it).
    as_of_date: the date the model is being fit "as of" — normally the
      cutoff passed to load_training_matches, not the last match's own
      date, so a slate with no recent matches still decays correctly.
    half_life_days: must be > 0. Larger = slower decay = longer memory.

    Matches on/after as_of_date would produce a non-positive age and
    are almost certainly a walk-forward leak (fitting on data from the
    future), so this raises rather than silently clipping to weight=1.
    """
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be > 0, got {half_life_days}")

    dates = np.asarray(match_dates, dtype="datetime64[D]")
    cutoff = np.datetime64(as_of_date, "D")

    age_days = (cutoff - dates).astype("timedelta64[D]").astype(np.float64)

    if np.any(age_days <= 0):
        n_bad = int(np.sum(age_days <= 0))
        raise ValueError(
            f"{n_bad} match date(s) are on or after as_of_date={as_of_date}. "
            f"decay_weights() expects strictly historical matches — filter "
            f"the training query by as_of_date before computing weights, "
            f"otherwise this is a walk-forward leak."
        )

    return np.exp2(-age_days / half_life_days)