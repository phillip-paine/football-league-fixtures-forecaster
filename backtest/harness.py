"""
Walk-forward backtest harness: monthly refit checkpoints, score every
match in between. Per forecast_engine_design_notes.md's validation
framework — "Walk-forward / rolling-origin backtesting only — never
random splits."

Deliberately dependency-injected (loader functions passed in as
arguments) rather than hardcoding `from features import ...` — this lets
the harness be unit-tested against a synthetic in-memory dataset (see
tests/test_backtest.py) without a real DB. `run_backtest.py` is the thin
CLI that wires in the real `features`/`model` packages.
"""

import os
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from .baselines import naive_frequency_baseline
from .naming import backtest_filename
from .rps import brier_score, log_loss_score, outcome_idx_from_goals, rank_probability_score


def monthly_checkpoints(start: date, end: date) -> List[date]:
    """First-of-month dates in [start, end). `start` itself is included
    only if it's already the 1st."""
    checkpoints = []
    y, m = start.year, start.month
    if start.day != 1:
        m += 1
        if m > 12:
            m, y = 1, y + 1
    current = date(y, m, 1)
    while current < end:
        checkpoints.append(current)
        y, m = current.year, current.month
        m += 1
        if m > 12:
            m, y = 1, y + 1
        current = date(y, m, 1)
    return checkpoints


@dataclass
class BacktestConfig:
    experiment: str
    method: str
    half_life_days: float
    min_history_days: int = 180  # skip scoring checkpoints with less training history than this
    draws: int = 500
    tune: int = 500
    chains: int = 2
    target_accept: float = 0.9
    max_goals: int = 10
    model_config: Optional[object] = None  # model.ModelConfig override, for ablation variants
    random_seed: Optional[int] = None
    loader_kwargs: dict = field(default_factory=dict)
    # loader_kwargs: (e.g. the
    # promotion/relegation flag's `include_promotion_covariate` +
    # `promotion_status`). The harness deliberately doesn't know what's
    # in here; that's what keeps it covariate-agnostic. A covariate that
    # instead changes the model's structure (not just data presence)
    # goes through `fit_fn` instead, which is already dependency-injected.


def _filter_played(dataset):
    """Restrict a MatchDataset-shaped dataclass to is_played rows.
    load_fixtures can return a mix of played/scheduled rows depending on
    the window; for a backtest (a historical window) we expect all of
    them played, but filter defensively rather than assume."""
    mask = np.asarray(dataset.is_played, dtype=bool)
    try:
        return replace(
            dataset,
            match_id=np.asarray(dataset.match_id)[mask],
            match_date=np.asarray(dataset.match_date)[mask],
            season_id=np.asarray(dataset.season_id)[mask],
            competition_id=np.asarray(dataset.competition_id)[mask],
            division_idx=np.asarray(dataset.division_idx)[mask],
            home_team_id=np.asarray(dataset.home_team_id)[mask],
            away_team_id=np.asarray(dataset.away_team_id)[mask],
            home_idx=np.asarray(dataset.home_idx)[mask],
            away_idx=np.asarray(dataset.away_idx)[mask],
            home_goals=np.asarray(dataset.home_goals)[mask],
            away_goals=np.asarray(dataset.away_goals)[mask],
            is_played=np.asarray(dataset.is_played)[mask],
            decay_weight=None if dataset.decay_weight is None else np.asarray(dataset.decay_weight)[mask],
        )
    except TypeError as e:
        raise TypeError(
            "run_walk_forward_backtest's played-row filter expects a "
            "dataclass-shaped MatchDataset with the fields documented in "
            "feature_extraction_handoff.md; got a different shape "
            f"({type(dataset)})."
        ) from e


def run_walk_forward_backtest(
        conn,
        *,
        start_date: date,
        end_date: date,
        config: BacktestConfig,
        build_team_index_fn: Callable,
        build_division_index_fn: Callable,
        load_training_matches_fn: Callable,
        load_fixtures_fn: Callable,
        fit_fn: Callable,
        posterior_rates_fn: Callable,
        match_outcome_probs_fn: Callable,
        out_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Runs the monthly walk-forward sweep and returns one row per scored
    match. If `out_dir` is given, also writes one CSV per checkpoint to
    `out_dir/<experiment>/<filename>` (see naming.py for the convention).

    `fit_fn`/`posterior_rates_fn`/`match_outcome_probs_fn` are
    `model.fit`/`model.posterior_rates`/`model.match_outcome_probs` (or
    equivalents) — passed in rather than imported so this module has no
    hard dependency on `model` either, keeping it a pure orchestration
    layer over whatever fitting/prediction functions are given it.
    """
    team_index = build_team_index_fn(conn)
    division_index = build_division_index_fn(conn)

    checkpoints = monthly_checkpoints(start_date, end_date)
    if len(checkpoints) < 2:
        raise ValueError(
            f"need at least 2 monthly checkpoints between {start_date} and {end_date} "
            "(one to refit on, one to mark the end of the last scoring window)"
        )

    all_rows = []
    for as_of, window_end in zip(checkpoints[:-1], checkpoints[1:]):
        train = load_training_matches_fn(
            conn,
            as_of_date=as_of,
            half_life_days=config.half_life_days,
            team_index=team_index,
            division_index=division_index,
            **config.loader_kwargs,
        )
        if train.n_matches == 0:
            print(f"[{as_of}] no training matches yet, skipping")
            continue

        earliest = pd.to_datetime(np.asarray(train.match_date)).min()
        history_days = (pd.Timestamp(as_of) - earliest).days
        if history_days < config.min_history_days:
            print(f"[{as_of}] skipping — only {history_days}d of training history (need {config.min_history_days}d)")
            continue

        fixtures = load_fixtures_fn(
            conn,
            start_date=as_of,
            end_date=window_end,
            team_index=team_index,
            division_index=division_index,
            **config.loader_kwargs,
        )
        fixtures_played = _filter_played(fixtures)
        if fixtures_played.n_matches == 0:
            print(f"[{as_of}] no played matches in [{as_of}, {window_end}) to score, skipping")
            continue

        print(
            f"[{as_of}] fitting on {train.n_matches} matches, "
            f"scoring {fixtures_played.n_matches} in [{as_of}, {window_end})"
        )
        _, idata = fit_fn(
            train,
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
            config=config.model_config,
        )

        rates = posterior_rates_fn(
            idata, fixtures_played.home_idx, fixtures_played.away_idx, fixtures_played.division_idx
        )
        probs = match_outcome_probs_fn(rates, max_goals=config.max_goals)
        outcome_idx = outcome_idx_from_goals(fixtures_played.home_goals, fixtures_played.away_goals)

        rps = rank_probability_score(probs, outcome_idx)
        brier = brier_score(probs, outcome_idx)
        logloss = log_loss_score(probs, outcome_idx)

        naive_probs = naive_frequency_baseline(train.home_goals, train.away_goals, fixtures_played.n_matches)
        naive_rps = rank_probability_score(naive_probs, outcome_idx)

        checkpoint_df = pd.DataFrame(
            {
                "experiment": config.experiment,
                "method": config.method,
                "as_of_date": as_of.isoformat(),
                "match_id": fixtures_played.match_id,
                "match_date": fixtures_played.match_date,
                "division_idx": fixtures_played.division_idx,
                "home_idx": fixtures_played.home_idx,
                "away_idx": fixtures_played.away_idx,
                "home_goals": fixtures_played.home_goals,
                "away_goals": fixtures_played.away_goals,
                "outcome_idx": outcome_idx,
                "p_home_win": probs[:, 0],
                "p_draw": probs[:, 1],
                "p_away_win": probs[:, 2],
                "rps": rps,
                "brier": brier,
                "log_loss": logloss,
                "naive_p_home_win": naive_probs[:, 0],
                "naive_p_draw": naive_probs[:, 1],
                "naive_p_away_win": naive_probs[:, 2],
                "naive_rps": naive_rps,
            }
        )
        all_rows.append(checkpoint_df)

        if out_dir is not None:
            exp_dir = os.path.join(out_dir, config.experiment)
            os.makedirs(exp_dir, exist_ok=True)
            fname = backtest_filename(config.experiment, config.method, as_of, as_of, window_end)
            path = os.path.join(exp_dir, fname)
            checkpoint_df.to_csv(path, index=False)
            print(f"  saved {path}")

    if not all_rows:
        raise ValueError(
            "no checkpoints produced scoreable results — widen the date range, "
            "check the DB actually has matches in it, or lower --min-history-days"
        )

    return pd.concat(all_rows, ignore_index=True)
