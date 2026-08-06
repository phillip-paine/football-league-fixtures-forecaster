"""
The extraction layer's main deliverable: turn rows in `matches` into
plain numpy arrays a PyMC model can consume directly, using a team/
division index space that stays consistent between fitting and
prediction (see features.indices for why that matters).

Two entry points cover both loop phases in the design notes:

- load_training_matches(): historical, played, strictly before a
  cutoff, decay-weighted. Feeds the weekly MCMC refit.
- load_fixtures(): a date window at/after the cutoff. With unplayed
  matches this is next weekend's slate to predict; with already-played
  matches held out of training, this is exactly what walk-forward
  backtesting scores against. Same function either way — "did this
  match already happen" is a property of the row, not of which mode
  you think you're in, and conflating them is a classic way to
  accidentally leak future data into a backtest.

No covariates yet (weather, rest days, key-player status) — the build
log's own ordering puts those after the minimal Dixon-Coles fit is
proven out, so they're deliberately out of scope here. Every loader
returns a MatchDataset carrying match_id, so a covariate layer can
join its own arrays on afterward without touching this module.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .decay import decay_weights
from .indices import IndexMap, build_division_index, build_team_index
from .promotion import PromotionStatus, build_promotion_status, lookup as lookup_promotion

DateLike = dt.date | dt.datetime | str


def _to_date(d: DateLike) -> dt.date:
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return dt.date.fromisoformat(d)


@dataclass
class MatchDataset:
    """Model-ready arrays. Every array is aligned by position — row i
    of every field describes the same match."""

    match_id: np.ndarray  # int64
    match_date: np.ndarray  # datetime64[D]
    season_id: np.ndarray  # int64
    competition_id: np.ndarray  # int64 (raw db id)
    division_idx: np.ndarray  # int64, 0-based, indexes into division_index
    home_team_id: np.ndarray  # int64 (raw db id)
    away_team_id: np.ndarray  # int64 (raw db id)
    home_idx: np.ndarray  # int64, 0-based, indexes into team_index
    away_idx: np.ndarray  # int64, 0-based, indexes into team_index
    home_goals: np.ndarray  # float64, NaN if not yet played
    away_goals: np.ndarray  # float64, NaN if not yet played
    is_played: np.ndarray  # bool
    decay_weight: np.ndarray | None  # float64, or None (fixtures/backtest windows)

    team_index: IndexMap
    division_index: IndexMap

    # Promoted/relegated covariate flags — see features.promotion. bool,
    # True if that team was promoted/relegated *into* the division it's
    # playing in for this match's season. Optional (default None) so
    # datasets built before this feature existed, or any code path that
    # deliberately opts out, still construct fine; model/dixon_coles.py
    # treats None the same as all-False (no promoted/relegated effect).
    is_promoted_home: np.ndarray | None = None  # bool
    is_promoted_away: np.ndarray | None = None  # bool
    is_relegated_home: np.ndarray | None = None  # bool
    is_relegated_away: np.ndarray | None = None  # bool

    @property
    def n_matches(self) -> int:
        return len(self.match_id)

    @property
    def n_teams(self) -> int:
        return len(self.team_index)

    @property
    def n_divisions(self) -> int:
        return len(self.division_index)

    def to_frame(self) -> pd.DataFrame:
        """Flat pandas view for inspection/debugging — not what gets
        passed into PyMC, which wants the raw arrays/indices above."""
        return pd.DataFrame(
            {
                "match_id": self.match_id,
                "match_date": self.match_date,
                "season_id": self.season_id,
                "competition_id": self.competition_id,
                "division_idx": self.division_idx,
                "home_team_id": self.home_team_id,
                "away_team_id": self.away_team_id,
                "home_idx": self.home_idx,
                "away_idx": self.away_idx,
                "home_goals": self.home_goals,
                "away_goals": self.away_goals,
                "is_played": self.is_played,
                "decay_weight": self.decay_weight
                if self.decay_weight is not None
                else np.nan,
                "is_promoted_home": self.is_promoted_home,
                "is_promoted_away": self.is_promoted_away,
                "is_relegated_home": self.is_relegated_home,
                "is_relegated_away": self.is_relegated_away,
            }
        )

    def __len__(self) -> int:
        return self.n_matches


_BASE_QUERY = """
    SELECT match_id, match_date, season_id, competition_id,
           home_team_id, away_team_id, home_goals, away_goals, status
    FROM matches
"""


def _run_match_query(
        conn: sqlite3.Connection,
        where_clauses: list[str],
        params: list,
        competitions: list[int] | None,
) -> list[sqlite3.Row]:
    clauses = list(where_clauses)
    if competitions:
        placeholders = ",".join("?" * len(competitions))
        clauses.append(f"competition_id IN ({placeholders})")
        params = list(params) + list(competitions)

    query = _BASE_QUERY
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY match_date, match_id"

    return conn.execute(query, params).fetchall()


def _rows_to_dataset(
        rows: list[sqlite3.Row],
        team_index: IndexMap,
        division_index: IndexMap,
        weight_as_of: dt.date | None,
        half_life_days: float | None,
        promotion_status: dict[tuple[int, int], PromotionStatus] | None = None,
) -> MatchDataset:
    n = len(rows)
    match_id = np.empty(n, dtype=np.int64)
    match_date = np.empty(n, dtype="datetime64[D]")
    season_id = np.empty(n, dtype=np.int64)
    competition_id = np.empty(n, dtype=np.int64)
    home_team_id = np.empty(n, dtype=np.int64)
    away_team_id = np.empty(n, dtype=np.int64)
    home_goals = np.full(n, np.nan, dtype=np.float64)
    away_goals = np.full(n, np.nan, dtype=np.float64)
    is_played = np.zeros(n, dtype=bool)
    is_promoted_home = np.zeros(n, dtype=bool)
    is_promoted_away = np.zeros(n, dtype=bool)
    is_relegated_home = np.zeros(n, dtype=bool)
    is_relegated_away = np.zeros(n, dtype=bool)

    for i, r in enumerate(rows):
        match_id[i] = r["match_id"]
        match_date[i] = np.datetime64(r["match_date"], "D")
        season_id[i] = r["season_id"]
        competition_id[i] = r["competition_id"]
        home_team_id[i] = r["home_team_id"]
        away_team_id[i] = r["away_team_id"]
        played = r["home_goals"] is not None and r["away_goals"] is not None
        is_played[i] = played
        if played:
            home_goals[i] = r["home_goals"]
            away_goals[i] = r["away_goals"]

        if promotion_status is not None:
            home_status = lookup_promotion(promotion_status, r["home_team_id"], r["season_id"])
            away_status = lookup_promotion(promotion_status, r["away_team_id"], r["season_id"])
            is_promoted_home[i] = home_status.is_promoted
            is_promoted_away[i] = away_status.is_promoted
            is_relegated_home[i] = home_status.is_relegated
            is_relegated_away[i] = away_status.is_relegated

    home_idx = np.array(team_index.to_idx(home_team_id.tolist()), dtype=np.int64)
    away_idx = np.array(team_index.to_idx(away_team_id.tolist()), dtype=np.int64)
    division_idx = np.array(
        division_index.to_idx(competition_id.tolist()), dtype=np.int64
    )

    weight = None
    if weight_as_of is not None:
        weight = decay_weights(match_date, weight_as_of, half_life_days)

    return MatchDataset(
        match_id=match_id,
        match_date=match_date,
        season_id=season_id,
        competition_id=competition_id,
        division_idx=division_idx,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_idx=home_idx,
        away_idx=away_idx,
        home_goals=home_goals,
        away_goals=away_goals,
        is_played=is_played,
        decay_weight=weight,
        team_index=team_index,
        division_index=division_index,
        is_promoted_home=is_promoted_home if promotion_status is not None else None,
        is_promoted_away=is_promoted_away if promotion_status is not None else None,
        is_relegated_home=is_relegated_home if promotion_status is not None else None,
        is_relegated_away=is_relegated_away if promotion_status is not None else None,
    )


def load_training_matches(
        conn: sqlite3.Connection,
        as_of_date: DateLike,
        half_life_days: float,
        competitions: list[int] | None = None,
        min_date: DateLike | None = None,
        team_index: IndexMap | None = None,
        division_index: IndexMap | None = None,
        promotion_status: dict[tuple[int, int], "PromotionStatus"] | None = None,
        include_promotion_covariate: bool = True,
) -> MatchDataset:
    """Played matches strictly before as_of_date, decay-weighted
    relative to as_of_date. This is what the weekly MCMC refit fits on.

    as_of_date is the walk-forward cutoff, not "today" — pass the
    Saturday morning date for a weekly production run, or a historical
    date when backtesting a past gameweek. Matches on/after it are
    never included, so nothing about the outcome you're about to
    predict can leak into the fit.

    team_index/division_index: pass the same IndexMap objects used
    elsewhere in a run (e.g. the ones load_fixtures() will also use) if
    you want to guarantee identity rather than relying on two builds
    of the dimension tables happening to agree (they will, unless the
    `teams`/`competitions` tables change between calls — but pass them
    explicitly for a backtest loop where that matters).

    promotion_status: same reuse pattern as team_index/division_index —
    pass the same dict (from features.promotion.build_promotion_status)
    used elsewhere in a run. Built automatically if not supplied and
    include_promotion_covariate is True. Set include_promotion_covariate
    =False to skip it entirely (dataset.is_promoted_home etc. stay None)
    — useful for an ablation-test "baseline" run that must not see this
    covariate at all.
    """
    as_of = _to_date(as_of_date)
    where = ["status = 'played'", "home_goals IS NOT NULL", "away_goals IS NOT NULL",
             "match_date < ?"]
    params: list = [as_of.isoformat()]
    if min_date is not None:
        where.append("match_date >= ?")
        params.append(_to_date(min_date).isoformat())

    rows = _run_match_query(conn, where, params, competitions)

    team_index = team_index or build_team_index(conn)
    division_index = division_index or build_division_index(conn)
    if include_promotion_covariate and promotion_status is None:
        promotion_status = build_promotion_status(conn)
    elif not include_promotion_covariate:
        promotion_status = None

    return _rows_to_dataset(
        rows, team_index, division_index, weight_as_of=as_of, half_life_days=half_life_days,
        promotion_status=promotion_status
    )


def load_fixtures(
        conn: sqlite3.Connection,
        start_date: DateLike,
        end_date: DateLike | None = None,
        horizon_days: int | None = None,
        competitions: list[int] | None = None,
        team_index: IndexMap | None = None,
        division_index: IndexMap | None = None,
        promotion_status: dict[tuple[int, int], "PromotionStatus"] | None = None,
        include_promotion_covariate: bool = True,
) -> MatchDataset:
    """Matches in [start_date, end_date). No decay weights (nothing to
    weight — these are being predicted, not fit on).

    Works for both loop phases:
    - Live weekly prediction: start_date = today, matches here are
      status='scheduled' with goals NULL. is_played is False throughout.
    - Walk-forward backtest scoring: start_date = the same cutoff used
      for load_training_matches(), end_date = cutoff + one gameweek.
      Those matches already have status='played' with real goals — the
      dataset carries them (is_played=True, home_goals/away_goals
      populated) so a scoring function can compare predicted vs. actual
      without a second, differently-shaped query.

    Pass exactly one of end_date or horizon_days (horizon_days is
    shorthand for end_date = start_date + horizon_days).

    team_index/division_index should be the SAME objects passed to (or
    returned by) load_training_matches() for this run, so a team's
    array position here matches the position its fitted parameter
    lives at. Left as None only for standalone/inspection use.

    promotion_status/include_promotion_covariate: same as
    load_training_matches — pass the same promotion_status dict used
    for the training call in this run so the fixture window's flags are
    consistent with whatever the model was fit on.
    """
    start = _to_date(start_date)
    if end_date is not None and horizon_days is not None:
        raise ValueError("pass end_date or horizon_days, not both")
    if end_date is not None:
        end = _to_date(end_date)
    elif horizon_days is not None:
        end = start + dt.timedelta(days=horizon_days)
    else:
        raise ValueError("pass one of end_date or horizon_days")

    where = ["match_date >= ?", "match_date < ?"]
    params: list = [start.isoformat(), end.isoformat()]

    rows = _run_match_query(conn, where, params, competitions)

    team_index = team_index or build_team_index(conn)
    division_index = division_index or build_division_index(conn)
    if include_promotion_covariate and promotion_status is None:
        promotion_status = build_promotion_status(conn)
    elif not include_promotion_covariate:
        promotion_status = None

    return _rows_to_dataset(
        rows, team_index, division_index, weight_as_of=None, half_life_days=None, promotion_status=promotion_status
    )