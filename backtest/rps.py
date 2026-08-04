"""
Scoring metrics for match outcome forecasts.

Rank Probability Score (RPS) is the primary metric per
forecast_engine_design_notes.md ("Validation framework" section) — it
respects the ordering of outcomes (home win -> draw -> away win), so a
home-win prediction that's wrong by landing on a draw is penalized less
than one that's wrong by landing on an away win. Brier score and log loss
are secondary diagnostics, also named in the design notes.
"""

import numpy as np
from scipy.stats import wilcoxon


def rank_probability_score(probs: np.ndarray, outcome_idx: np.ndarray) -> np.ndarray:
    """
    probs: (n, 3) array of [P(home), P(draw), P(away)] — MUST be in this
    order; RPS depends on the outcomes being ordered along one axis
    (home win -> draw -> away win is the natural ordering for football).
    outcome_idx: (n,) int array in {0, 1, 2}, same ordering.

    Returns (n,) array of per-match RPS in [0, 1] — lower is better.
    """
    probs = np.asarray(probs, dtype=float)
    outcome_idx = np.asarray(outcome_idx, dtype=int)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"rank_probability_score expects an (n,3) array, got shape {probs.shape}")
    if probs.shape[0] != outcome_idx.shape[0]:
        raise ValueError("probs and outcome_idx length mismatch")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("each row of probs must sum to 1")
    if ((outcome_idx < 0) | (outcome_idx > 2)).any():
        raise ValueError("outcome_idx must be in {0, 1, 2}")

    n, r = probs.shape
    actual = np.zeros((n, r))
    actual[np.arange(n), outcome_idx] = 1.0

    cum_pred = np.cumsum(probs, axis=1)
    cum_actual = np.cumsum(actual, axis=1)

    sq_diff = (cum_pred[:, :-1] - cum_actual[:, :-1]) ** 2
    return sq_diff.sum(axis=1) / (r - 1)


def brier_score(probs: np.ndarray, outcome_idx: np.ndarray) -> np.ndarray:
    """Multi-class Brier score — sum of squared differences across all
    outcome classes (no ordering assumption, unlike RPS)."""
    probs = np.asarray(probs, dtype=float)
    outcome_idx = np.asarray(outcome_idx, dtype=int)
    n, r = probs.shape
    actual = np.zeros((n, r))
    actual[np.arange(n), outcome_idx] = 1.0
    return ((probs - actual) ** 2).sum(axis=1)


def log_loss_score(probs: np.ndarray, outcome_idx: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    outcome_idx = np.asarray(outcome_idx, dtype=int)
    p_actual = probs[np.arange(len(outcome_idx)), outcome_idx]
    return -np.log(np.clip(p_actual, eps, 1.0))


def skill_score(scores_model: np.ndarray, scores_baseline: np.ndarray) -> float:
    """1 - mean(model)/mean(baseline), for any lower-is-better score
    (RPS, Brier, log loss). Positive = model beats the baseline;
    e.g. 0.05 means the model's average score is 5% better."""
    m = float(np.mean(scores_model))
    b = float(np.mean(scores_baseline))
    if b == 0:
        raise ValueError("baseline mean score is 0 — skill score is undefined")
    return 1 - (m / b)


def paired_significance_test(scores_a: np.ndarray, scores_b: np.ndarray):
    """Wilcoxon signed-rank test on per-match score differences between
    two model variants scored on the SAME matches (e.g. baseline vs a
    candidate covariate in an ablation test) — per
    forecast_engine_design_notes.md's "new covariates ship only if they
    improve out-of-sample RPS in ablation tests" rule, this is meant to
    check that an improvement is a real, consistent effect and not a
    handful of lucky matches, not just eyeball a lower mean.

    Returns (statistic, p_value). A small p-value with scores_a's mean
    lower than scores_b's mean supports "a is really better than b".
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if scores_a.shape != scores_b.shape:
        raise ValueError("paired_significance_test requires scores_a and scores_b to be scored on the same matches (same shape)")
    return wilcoxon(scores_a, scores_b)


def outcome_idx_from_goals(home_goals: np.ndarray, away_goals: np.ndarray) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    idx = np.full(home_goals.shape, 1, dtype=int)
    idx[home_goals > away_goals] = 0
    idx[home_goals < away_goals] = 2
    return idx
