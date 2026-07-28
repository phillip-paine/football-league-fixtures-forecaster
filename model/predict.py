"""
Turn a fitted model's posterior into match forecasts.

Approach: for each match and each posterior draw, compute the *exact*
Dixon-Coles-adjusted scoreline PMF over a bounded goal grid (0..max_goals),
then average that PMF across draws. This is equivalent to Monte Carlo
simulating scorelines from the posterior and tabulating frequencies, but
exact (no simulation noise) and cheap at this data scale.

match_outcome_probs() and scoreline_distribution() both accept the
`lam, mu, rho` arrays returned by `posterior_rates()` — kept as a separate
step so a future backtest/RPS harness can call `posterior_rates()` once per
as-of date and reuse it for both scoreline output and market-vs-RPS scoring.
"""

from typing import NamedTuple, Optional

import numpy as np
from scipy.stats import poisson


class PosteriorRates(NamedTuple):
    lam: np.ndarray  # (n_matches, n_samples) posterior draws of home expected goals
    mu: np.ndarray   # (n_matches, n_samples) posterior draws of away expected goals
    rho: np.ndarray  # (n_samples,) posterior draws of the DC correlation term


def posterior_rates(idata, home_idx: np.ndarray, away_idx: np.ndarray, division_idx: np.ndarray) -> PosteriorRates:
    """Compute posterior draws of (lam, mu, rho) for an arbitrary set of
    matches (e.g. an upcoming fixture window from `features.load_fixtures`)
    by combining posterior samples of the fitted team/division parameters —
    this is what lets one fit serve both "this week's predictions" and
    "backtest window N" without refitting.
    """
    post = idata.posterior
    # flatten (chain, draw) -> sample
    division_intercept = post["division_intercept"].values.reshape(-1, post.sizes["division"])
    home_advantage = post["home_advantage"].values.reshape(-1, post.sizes["division"])
    attack = post["attack"].values.reshape(-1, post.sizes["team"])
    defense = post["defense"].values.reshape(-1, post.sizes["team"])
    rho = post["rho"].values.reshape(-1)

    home_idx = np.asarray(home_idx, dtype="int64")
    away_idx = np.asarray(away_idx, dtype="int64")
    division_idx = np.asarray(division_idx, dtype="int64")

    # (n_samples, n_matches)
    log_lambda_home = (
            division_intercept[:, division_idx]
            + home_advantage[:, division_idx]
            + attack[:, home_idx]
            - defense[:, away_idx]
    )
    log_lambda_away = (
            division_intercept[:, division_idx]
            + attack[:, away_idx]
            - defense[:, home_idx]
    )
    lam = np.exp(log_lambda_home).T  # -> (n_matches, n_samples)
    mu = np.exp(log_lambda_away).T
    return PosteriorRates(lam=lam, mu=mu, rho=rho)


def scoreline_distribution(rates: PosteriorRates, max_goals: int = 10) -> np.ndarray:
    """Posterior-mean Dixon-Coles-adjusted scoreline PMF.

    Returns an (n_matches, max_goals+1, max_goals+1) array where
    `[m, x, y]` is P(home scores x, away scores y) for match m, averaged
    over the posterior (so it already reflects parameter uncertainty, not
    just the posterior-mean lambda/mu).
    """
    lam, mu, rho = rates.lam, rates.mu, rates.rho
    n_matches, n_samples = lam.shape
    goals = np.arange(max_goals + 1)

    out = np.empty((n_matches, max_goals + 1, max_goals + 1))
    for m in range(n_matches):
        # (max_goals+1, n_samples)
        pois_x = poisson.pmf(goals[:, None], lam[m][None, :])
        pois_y = poisson.pmf(goals[:, None], mu[m][None, :])
        # (max_goals+1, max_goals+1, n_samples)
        joint = pois_x[:, None, :] * pois_y[None, :, :]

        tau = np.ones_like(joint)
        tau[0, 0, :] = 1 - lam[m] * mu[m] * rho
        tau[0, 1, :] = 1 + lam[m] * rho
        tau[1, 0, :] = 1 + mu[m] * rho
        tau[1, 1, :] = 1 - rho
        joint = np.clip(joint * tau, 0.0, None)

        # renormalize per draw (tau shifts a small amount of mass; the
        # truncation at max_goals also loses a negligible tail) then
        # average over the posterior
        joint /= joint.sum(axis=(0, 1), keepdims=True)
        out[m] = joint.mean(axis=2)

    return out


def match_outcome_probs(rates: PosteriorRates, max_goals: int = 10) -> np.ndarray:
    """(n_matches, 3) array of [P(home win), P(draw), P(away win)] —
    the immediate deliverable for a public match-prediction page, and the
    shape the eventual RPS scorer will consume (RPS needs the ordered
    3-outcome cumulative distribution).
    """
    pmf = scoreline_distribution(rates, max_goals=max_goals)
    n = pmf.shape[1]
    x_goals, y_goals = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")  # [x, y]
    mask_home = x_goals > y_goals  # home scores more
    mask_draw = x_goals == y_goals
    mask_away = x_goals < y_goals

    p_home = (pmf * mask_home).sum(axis=(1, 2))
    p_draw = (pmf * mask_draw).sum(axis=(1, 2))
    p_away = (pmf * mask_away).sum(axis=(1, 2))
    return np.stack([p_home, p_draw, p_away], axis=1)
