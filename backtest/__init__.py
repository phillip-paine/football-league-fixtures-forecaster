"""
backtest/ — walk-forward validation harness, per build_log.md's
next-step item "Backtest / RPS scoring framework (walk-forward only)"
and forecast_engine_design_notes.md's "Validation framework" section.
"""

from .rps import (
    rank_probability_score,
    brier_score,
    log_loss_score,
    skill_score,
    paired_significance_test,
    outcome_idx_from_goals,
)
from .baselines import naive_frequency_baseline, market_implied_probs, elo_baseline
from .naming import season_code, checkpoint_label, window_label, backtest_filename
from .harness import BacktestConfig, monthly_checkpoints, run_walk_forward_backtest
from .aggregate import load_experiment, monthly_summary, season_summary, compare_experiments

__all__ = [
    "rank_probability_score",
    "brier_score",
    "log_loss_score",
    "skill_score",
    "paired_significance_test",
    "outcome_idx_from_goals",
    "naive_frequency_baseline",
    "market_implied_probs",
    "elo_baseline",
    "season_code",
    "checkpoint_label",
    "window_label",
    "backtest_filename",
    "BacktestConfig",
    "monthly_checkpoints",
    "run_walk_forward_backtest",
    "load_experiment",
    "monthly_summary",
    "season_summary",
    "compare_experiments",
]
