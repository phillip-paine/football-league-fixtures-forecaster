"""
Standalone integration test for model/ (no pytest dependency, matching
tests/test_extraction.py's style). Run with:

    python tests/test_dixon_coles.py

Uses the synthetic MatchDataset in tests/fixtures.py (NOT the real
features package / forecast.db — see fixtures.py docstring) to exercise
the full build -> fit -> predict path end to end, and specifically checks
the promoted/thin-team partial-pooling mechanism the model is meant to
provide.
"""

import sys
import time
import warnings

import numpy as np

sys.path.insert(0, ".")

from model.tau import dixon_coles_tau
from model.dixon_coles import build_model, build_model_from_arrays, fit, ModelConfig
from model.predict import posterior_rates, match_outcome_probs, scoreline_distribution
from tests.fixtures import make_synthetic_dataset

warnings.filterwarnings("ignore")

N_CHECKS = 0


def check(name, cond):
    global N_CHECKS
    N_CHECKS += 1
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        raise AssertionError(name)


def test_tau_reference():
    lam, mu, rho = 1.3, 0.9, -0.1
    check("tau(0,0) matches formula", np.isclose(dixon_coles_tau(0, 0, lam, mu, rho), 1 - lam * mu * rho))
    check("tau(0,1) matches formula", np.isclose(dixon_coles_tau(0, 1, lam, mu, rho), 1 + lam * rho))
    check("tau(1,0) matches formula", np.isclose(dixon_coles_tau(1, 0, lam, mu, rho), 1 + mu * rho))
    check("tau(1,1) matches formula", np.isclose(dixon_coles_tau(1, 1, lam, mu, rho), 1 - rho))
    check("tau(2,3) is unadjusted (1.0)", dixon_coles_tau(2, 3, lam, mu, rho) == 1.0)
    check("tau(0,0) with rho=0 collapses to independent Poisson", dixon_coles_tau(0, 0, lam, mu, 0.0) == 1.0)


def test_input_validation():
    n_teams, n_div, n = 4, 2, 5
    base = dict(
        home_idx=np.array([0, 1, 2, 3, 0]),
        away_idx=np.array([1, 2, 3, 0, 2]),
        division_idx=np.array([0, 0, 1, 1, 0]),
        home_goals=np.array([1, 2, 0, 1, 3]),
        away_goals=np.array([0, 1, 0, 1, 2]),
        weight=np.ones(n),
        n_teams=n_teams,
        n_divisions=n_div,
    )

    # baseline should build fine
    m = build_model_from_arrays(**base)
    check("valid input builds a pm.Model", m is not None)

    bad = dict(base)
    bad["home_idx"] = np.array([0, 1, 2, 3])  # wrong length
    try:
        build_model_from_arrays(**bad)
        check("length mismatch raises", False)
    except ValueError:
        check("length mismatch raises", True)

    bad = dict(base)
    bad["home_idx"] = np.array([0, 1, 2, 99, 0])  # out of range
    try:
        build_model_from_arrays(**bad)
        check("out-of-range team index raises", False)
    except ValueError:
        check("out-of-range team index raises", True)

    bad = dict(base)
    bad["home_goals"] = np.array([1.0, np.nan, 0, 1, 3])
    try:
        build_model_from_arrays(**bad)
        check("NaN goals raises", False)
    except ValueError:
        check("NaN goals raises", True)


def test_dataset_level_guards():
    train, fixtures, _ = make_synthetic_dataset(seed=1)

    try:
        build_model(fixtures)
        check("build_model rejects a fixtures dataset (decay_weight=None)", False)
    except ValueError:
        check("build_model rejects a fixtures dataset (decay_weight=None)", True)

    bad_train = train
    bad_train.is_played[0] = False
    try:
        build_model(bad_train)
        check("build_model rejects unplayed rows in training data", False)
    except ValueError:
        check("build_model rejects unplayed rows in training data", True)
    finally:
        bad_train.is_played[0] = True  # restore for downstream tests


def main():
    test_tau_reference()
    test_input_validation()
    test_dataset_level_guards()

    print("\nBuilding synthetic dataset (established teams + 2 thin/newcomer teams)...")
    train, fixtures, true_params = make_synthetic_dataset(seed=0)
    check(f"train has matches (n={train.n_matches})", train.n_matches > 0)
    check(f"n_teams={train.n_teams}, n_divisions={train.n_divisions}", train.n_teams == 12 and train.n_divisions == 2)

    print("Fitting (short chains, for end-to-end pipeline validation, not production-quality inference)...")
    t0 = time.time()
    model, idata = fit(train, draws=800, tune=800, chains=2, target_accept=0.9, random_seed=42)
    print(f"  sampled in {time.time() - t0:.1f}s")

    import arviz as az

    summary = az.summary(
        idata,
        var_names=["mu_intercept", "mu_home", "sigma_attack", "sigma_defense", "rho"],
    )
    print(summary)

    # this arviz version's summary() returns display-formatted strings for
    # some columns (e.g. "1.00") rather than floats — coerce explicitly
    import pandas as pd

    max_rhat = pd.to_numeric(summary["r_hat"]).max()
    min_ess = pd.to_numeric(summary["ess_bulk"]).min()
    check(f"r_hat reasonable for top-level params (max={max_rhat:.3f} < 1.1)", max_rhat < 1.1)
    check(f"ess_bulk reasonable for top-level params (min={min_ess:.0f} > 100)", min_ess > 100)

    n_divergent = int(idata.sample_stats["diverging"].sum())
    total_draws = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    check(
        f"divergence rate acceptable ({n_divergent}/{total_draws})",
        n_divergent < 0.05 * total_draws,
        )

    # --- shrinkage / partial-pooling check ---
    # teams 10 & 11 (ThinStrong/ThinWeak, 4 matches each) vs several
    # established teams (~16-20 matches each) — averaged over multiple
    # teams on each side to get a stable statistic (a single-pair
    # comparison is noisy at this chain length: the true posterior-sd gap
    # is real and consistent across seeds, but small enough that MCMC
    # sampling noise alone can flip an individual pair).
    attack_post = idata.posterior["attack"].values  # (chain, draw, team)
    sd_thin_teams = attack_post[..., [10, 11]].std(axis=(0, 1))
    sd_established_teams = attack_post[..., [0, 1, 2, 4]].std(axis=(0, 1))
    sd_thin_avg = sd_thin_teams.mean()
    sd_established_avg = sd_established_teams.mean()
    print(
        f"  avg posterior sd(attack): thin/newcomer teams (4 matches each)={sd_thin_avg:.3f} | "
        f"established teams (~16-20 matches each)={sd_established_avg:.3f}"
    )
    check(
        "thin/newcomer teams have wider average posterior uncertainty than established teams",
        sd_thin_avg > sd_established_avg,
        )

    # --- posterior-predictive sanity checks ---
    rates = posterior_rates(idata, fixtures.home_idx, fixtures.away_idx, fixtures.division_idx)
    check("posterior_rates lam/mu are positive", (rates.lam > 0).all() and (rates.mu > 0).all())

    probs = match_outcome_probs(rates, max_goals=10)
    check("match_outcome_probs shape is (n_fixtures, 3)", probs.shape == (fixtures.n_matches, 3))
    row_sums = probs.sum(axis=1)
    check(
        f"outcome probabilities sum to 1 per match (max abs err={np.max(np.abs(row_sums - 1)):.2e})",
        np.allclose(row_sums, 1.0, atol=1e-6),
    )
    check("all probabilities are non-negative", (probs >= -1e-9).all())

    pmf = scoreline_distribution(rates, max_goals=10)
    pmf_sums = pmf.sum(axis=(1, 2))
    check(
        "full scoreline PMF sums to 1 per match",
        np.allclose(pmf_sums, 1.0, atol=1e-6),
    )

    # cross-check scoreline_distribution's tau handling against the pure
    # reference function, using a single posterior draw in isolation
    single_lam = np.array([[1.4]])
    single_mu = np.array([[0.8]])
    single_rho = np.array([-0.15])
    from model.predict import PosteriorRates

    single_rates = PosteriorRates(lam=single_lam, mu=single_mu, rho=single_rho)
    single_pmf = scoreline_distribution(single_rates, max_goals=5)[0]
    from scipy.stats import poisson as sp_poisson

    manual_unnorm = np.array(
        [
            [
                dixon_coles_tau(x, y, 1.4, 0.8, -0.15) * sp_poisson.pmf(x, 1.4) * sp_poisson.pmf(y, 0.8)
                for y in range(6)
            ]
            for x in range(6)
        ]
    )
    manual = manual_unnorm / manual_unnorm.sum()
    check(
        "scoreline_distribution agrees with the manual tau-reference computation",
        np.allclose(single_pmf, manual, atol=1e-8),
    )

    print(f"\n{N_CHECKS}/{N_CHECKS} checks passed.")


if __name__ == "__main__":
    main()
