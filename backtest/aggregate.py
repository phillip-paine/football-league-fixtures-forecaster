"""
Aggregate per-checkpoint backtest CSVs (written by backtest.harness) into
monthly/season summary tables, and compare across experiments.

Reads from the CSVs' own embedded `experiment`/`method`/`match_date`
columns, never from filenames — filenames encode the same information
for human browsing (see naming.py), but aggregation always trusts the
data, not the path.
"""

import glob
import os
from typing import List

import pandas as pd

from .naming import season_code


def load_experiment(backtest_dir: str, experiment: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(backtest_dir, experiment, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"no backtest CSVs found under {os.path.join(backtest_dir, experiment)}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _add_season_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df["match_date"])
    df["season"] = [season_code(d.date()) for d in dates]
    df["month"] = dates.dt.month
    return df


def _summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    df = _add_season_month(df)
    grouped = (
        df.groupby(group_cols)
        .agg(
            n_matches=("rps", "size"),
            mean_rps=("rps", "mean"),
            mean_brier=("brier", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_naive_rps=("naive_rps", "mean"),
        )
        .reset_index()
    )
    grouped["skill_vs_naive"] = 1 - grouped["mean_rps"] / grouped["mean_naive_rps"]
    return grouped.sort_values(group_cols).reset_index(drop=True)


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _summarize(df, ["experiment", "method", "season", "month"])


def season_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _summarize(df, ["experiment", "method", "season"])


def compare_experiments(backtest_dir: str, experiments: List[str]) -> pd.DataFrame:
    """Load and combine several experiments' backtest output for
    side-by-side monthly comparison — this is the ablation-testing
    entry point: run the harness once per model variant into separate
    experiment directories, then compare here."""
    dfs = [load_experiment(backtest_dir, e) for e in experiments]
    combined = pd.concat(dfs, ignore_index=True)
    return monthly_summary(combined)
