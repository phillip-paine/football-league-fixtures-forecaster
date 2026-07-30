"""
Hierarchical Dixon-Coles Poisson goal model — minimal, no covariates.

Design decisions this file encodes (see model_handoff.md for the full
rationale and open questions):

- log-link Poisson goal model with a Dixon-Coles low-score correlation
  correction (rho), per forecast_engine_design_notes.md.
- Hierarchy is team -> division -> competition, but split across two
  *different* quantities rather than one nested parameter:
    * division_intercept / home_advantage are pooled toward a single
      top-level ("competition") mean — this captures each division's
      distinct scoring environment (Championship != Premier League) and
      its own home-advantage level.
    * attack / defense are pooled toward a single *global* mean across
      all teams (sum-to-zero constrained for identifiability), not
      toward their division. This is the actual promoted-team fix: a
      team with two games of top-flight history has almost no signal
      of its own, so its attack/defense shrinks hard toward the global
      average team, while the division_intercept for whichever tier
      it's playing in corrects for the different goal environment.
      (A team's `home_idx`/`away_idx` slot is stable across divisions
      per feature_extraction_handoff.md, so a division-specific attack
      prior isn't meaningful — a team doesn't have a "division", a
      *match* does.)
- decay_weight is folded in as a tempered/weighted log-likelihood via
  pm.Potential, since PyMC's Poisson has no native per-observation
  weight argument. weight=1 recovers the unweighted likelihood exactly.
- rho, division_intercept and home_advantage all use non-centered
  parameterizations for sampler geometry; attack/defense too.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pymc as pm
import pytensor.tensor as pt


@dataclass
class ModelConfig:
    """Prior-scale knobs. Defaults are weakly-informative log-goal-rate
    scales (a change of 1.0 on this log scale is roughly a 2.7x change
    in expected goals — deliberately loose for a first end-to-end fit).
    """

    mu_intercept_sd: float = 1.0
    sigma_intercept_sd: float = 0.5
    mu_home_mean: float = 0.3   # ~ log(1.35), a typical home-goals bump
    mu_home_sd: float = 0.25
    sigma_home_sd: float = 0.25
    sigma_attack_sd: float = 0.5
    sigma_defense_sd: float = 0.5
    rho_sd: float = 0.2
    max_goals_for_tau: int = 1  # Dixon-Coles correction only touches 0/1 scores


def _low_score_masks(home_goals: np.ndarray, away_goals: np.ndarray):
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_11 = (home_goals == 1) & (away_goals == 1)
    return mask_00, mask_01, mask_10, mask_11


def build_model_from_arrays(
        *,
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        division_idx: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weight: np.ndarray,
        n_teams: int,
        n_divisions: int,
        team_names: Optional[Sequence[str]] = None,
        division_names: Optional[Sequence[str]] = None,
        config: Optional[ModelConfig] = None,
) -> pm.Model:
    """Build (but do not fit) the hierarchical Dixon-Coles model from raw
    arrays. This is the primitive entry point — `build_model` below is a
    thin adapter over a `MatchDataset`; use this one directly if you're
    not going through the `features` package (e.g. in tests).

    All arrays must be 1D, same length (n_matches). `home_idx`/`away_idx`
    index into [0, n_teams); `division_idx` indexes into [0, n_divisions).
    `weight` is the per-match decay weight (features.decay_weights output);
    pass an array of ones for an unweighted fit.
    """
    cfg = config or ModelConfig()

    home_idx = np.asarray(home_idx, dtype="int64")
    away_idx = np.asarray(away_idx, dtype="int64")
    division_idx = np.asarray(division_idx, dtype="int64")
    home_goals_raw = np.asarray(home_goals, dtype="float64")
    away_goals_raw = np.asarray(away_goals, dtype="float64")
    if np.isnan(home_goals_raw).any() or np.isnan(away_goals_raw).any():
        raise ValueError(
            "build_model_from_arrays received NaN goals — filter to is_played "
            "rows first (this should be a training MatchDataset, not fixtures)."
        )
    home_goals = home_goals_raw.astype("int64")
    away_goals = away_goals_raw.astype("int64")
    weight = np.asarray(weight, dtype="float64")

    n_matches = home_goals.shape[0]
    for name, arr in [
        ("home_idx", home_idx),
        ("away_idx", away_idx),
        ("division_idx", division_idx),
        ("away_goals", away_goals),
        ("weight", weight),
    ]:
        if arr.shape[0] != n_matches:
            raise ValueError(
                f"length mismatch: home_goals has {n_matches} rows, {name} has {arr.shape[0]}"
            )
    if home_idx.min(initial=0) < 0 or (n_matches and home_idx.max() >= n_teams):
        raise ValueError("home_idx out of range [0, n_teams)")
    if away_idx.min(initial=0) < 0 or (n_matches and away_idx.max() >= n_teams):
        raise ValueError("away_idx out of range [0, n_teams)")
    if n_matches and (division_idx.min() < 0 or division_idx.max() >= n_divisions):
        raise ValueError("division_idx out of range [0, n_divisions)")
    team_names = list(team_names) if team_names is not None else [str(i) for i in range(n_teams)]
    division_names = (
        list(division_names) if division_names is not None else [str(i) for i in range(n_divisions)]
    )
    if len(team_names) != n_teams:
        raise ValueError(f"team_names has {len(team_names)} entries, n_teams={n_teams}")
    if len(division_names) != n_divisions:
        raise ValueError(f"division_names has {len(division_names)} entries, n_divisions={n_divisions}")

    mask_00, mask_01, mask_10, mask_11 = _low_score_masks(home_goals, away_goals)

    coords = {
        "team": team_names,
        "division": division_names,
        "match": np.arange(n_matches),
    }

    with pm.Model(coords=coords) as model:
        home_idx_d = pm.Data("home_idx", home_idx, dims="match")
        away_idx_d = pm.Data("away_idx", away_idx, dims="match")
        division_idx_d = pm.Data("division_idx", division_idx, dims="match")
        weight_d = pm.Data("weight", weight, dims="match")
        home_goals_d = pm.Data("home_goals_obs", home_goals, dims="match")
        away_goals_d = pm.Data("away_goals_obs", away_goals, dims="match")

        # --- top level ("competition"): one mean scoring rate, one mean
        # home advantage, shared by every division ---
        mu_intercept = pm.Normal("mu_intercept", 0.0, cfg.mu_intercept_sd)
        sigma_intercept = pm.HalfNormal("sigma_intercept", cfg.sigma_intercept_sd)
        mu_home = pm.Normal("mu_home", cfg.mu_home_mean, cfg.mu_home_sd)
        sigma_home = pm.HalfNormal("sigma_home", cfg.sigma_home_sd)

        # --- division level: each tier's own scoring environment and
        # home-advantage level, non-centered around the competition mean ---
        division_intercept_raw = pm.Normal("division_intercept_raw", 0.0, 1.0, dims="division")
        division_intercept = pm.Deterministic(
            "division_intercept",
            mu_intercept + sigma_intercept * division_intercept_raw,
            dims="division",
            )
        home_advantage_raw = pm.Normal("home_advantage_raw", 0.0, 1.0, dims="division")
        home_advantage = pm.Deterministic(
            "home_advantage",
            mu_home + sigma_home * home_advantage_raw,
            dims="division",
            )

        # --- team level: global partial pooling, sum-to-zero constrained.
        # This is the mechanism that shrinks a newly-promoted team's
        # attack/defense toward the average team until it earns its own
        # signal — see module docstring. ---
        sigma_attack = pm.HalfNormal("sigma_attack", cfg.sigma_attack_sd)
        sigma_defense = pm.HalfNormal("sigma_defense", cfg.sigma_defense_sd)
        attack_raw = pm.Normal("attack_raw", 0.0, 1.0, dims="team")
        defense_raw = pm.Normal("defense_raw", 0.0, 1.0, dims="team")
        attack = pm.Deterministic(
            "attack", sigma_attack * (attack_raw - attack_raw.mean()), dims="team"
        )
        defense = pm.Deterministic(
            "defense", sigma_defense * (defense_raw - defense_raw.mean()), dims="team"
        )

        # --- Dixon-Coles low-score correlation parameter ---
        rho = pm.Normal("rho", 0.0, cfg.rho_sd)

        log_lambda_home = (
                division_intercept[division_idx_d]
                + home_advantage[division_idx_d]
                + attack[home_idx_d]
                - defense[away_idx_d]
        )
        log_lambda_away = (
                division_intercept[division_idx_d]
                + attack[away_idx_d]
                - defense[home_idx_d]
        )
        lam = pm.Deterministic("lam", pt.exp(log_lambda_home), dims="match")
        mu = pm.Deterministic("mu", pt.exp(log_lambda_away), dims="match")

        # tau(x,y) Dixon-Coles correction, static masks (data-derived, not
        # parameters) picking out which rows get which branch
        tau = pt.ones_like(lam)
        tau = pt.switch(mask_00, 1 - lam * mu * rho, tau)
        tau = pt.switch(mask_01, 1 + lam * rho, tau)
        tau = pt.switch(mask_10, 1 + mu * rho, tau)
        tau = pt.switch(mask_11, 1 - rho, tau)
        # tau can go <=0 for extreme rho during warmup; floor it rather than
        # let log() produce -inf/NaN and stall the sampler
        log_tau = pt.log(pt.maximum(tau, 1e-8))

        loglike_home = pm.logp(pm.Poisson.dist(mu=lam), home_goals_d)
        loglike_away = pm.logp(pm.Poisson.dist(mu=mu), away_goals_d)

        # weighted/tempered log-likelihood — this is how decay_weight
        # (features.decay_weights) enters the model. weight==1 everywhere
        # recovers the ordinary (unweighted) Poisson likelihood exactly.
        pm.Potential(
            "weighted_loglike", weight_d * (loglike_home + loglike_away + log_tau)
        )

    return model


def build_model(dataset, config: Optional[ModelConfig] = None) -> pm.Model:
    """Build the model from a `features.MatchDataset` (see
    feature_extraction_handoff.md for the contract). Expects a *training*
    dataset from `load_training_matches` — i.e. every row already played,
    with a real (non-None) decay_weight — not a `load_fixtures` window.

    Coordinate labels for the 'team'/'division' dims come from
    `dataset.team_index.idx_to_id` / `dataset.division_index.idx_to_id`
    (raw db ids, stringified) — NOT human names. `IndexMap` only ever
    stores id<->position mappings; it has no name lookup, so pretending
    otherwise here would just be guessing. If you want human-readable
    names in downstream output (e.g. a predictions CSV), resolve
    id -> name against the `teams`/`competitions` tables at the point
    where you have a DB connection (see run_dixon_coles.py's
    `resolve_id_to_name` for the pattern) — don't thread DB access into
    this module.
    """
    is_played = np.asarray(dataset.is_played, dtype=bool)
    if not is_played.all():
        raise ValueError(
            "build_model expects an all-played training MatchDataset "
            f"(is_played all True); got {(~is_played).sum()} unplayed rows out of "
            f"{is_played.shape[0]}. Use features.load_training_matches, not "
            "load_fixtures, to build the training set."
        )
    if dataset.decay_weight is None:
        raise ValueError(
            "dataset.decay_weight is None — this looks like a fixtures/backtest "
            "window (features.load_fixtures), which doesn't carry weights. "
            "Pass the output of features.load_training_matches instead."
        )

    team_ids = _extract_coord_labels(getattr(dataset, "team_index", None), dataset.n_teams)
    division_ids = _extract_coord_labels(getattr(dataset, "division_index", None), dataset.n_divisions)

    return build_model_from_arrays(
        home_idx=dataset.home_idx,
        away_idx=dataset.away_idx,
        division_idx=dataset.division_idx,
        home_goals=dataset.home_goals,
        away_goals=dataset.away_goals,
        weight=dataset.decay_weight,
        n_teams=dataset.n_teams,
        n_divisions=dataset.n_divisions,
        team_names=team_ids,
        division_names=division_ids,
        config=config,
    )


def _extract_coord_labels(index_map, n: int):
    """Coordinate labels for the 'team'/'division' pm.Model dims.

    Uses `IndexMap.idx_to_id` — position -> raw db id — which is
    guaranteed to exist on the real `features.IndexMap` (confirmed against
    the actual class: it only ever stores `id_to_idx`/`idx_to_id`, never
    names; name resolution lives in the `teams`/`competitions` tables, not
    on IndexMap). This intentionally does NOT attempt to look up human
    names here — model/ has no DB connection and shouldn't guess at one;
    name resolution for display/output belongs at the CLI layer, which
    does have `conn` (see run_dixon_coles.py's `resolve_id_to_name`).

    Falls back to positional strings only if `idx_to_id` isn't present at
    all (e.g. a bare dataclass with no ids, as in ad-hoc testing) — never
    silently guesses at unrelated attribute names.
    """
    if index_map is None:
        return None
    if hasattr(index_map, "idx_to_id"):
        ids = list(index_map.idx_to_id)
        if len(ids) == n:
            return [str(i) for i in ids]
    return None


def fit(
        dataset_or_arrays,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        cores: Optional[int] = None,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        config: Optional[ModelConfig] = None,
        **build_kwargs,
):
    """build_model(_from_arrays) + pm.sample in one call.

    `dataset_or_arrays` is either a `features.MatchDataset` (goes through
    `build_model`) or, if `build_kwargs` are supplied (home_idx=..., etc.),
    is ignored and `build_model_from_arrays(**build_kwargs)` is used
    instead — pass `dataset_or_arrays=None` in that case.

    Returns an arviz.InferenceData with posterior samples of
    `division_intercept`, `home_advantage`, `attack`, `defense`, `rho`,
    plus `lam`/`mu` (per-match expected goals) as Deterministics.
    """
    if build_kwargs:
        model = build_model_from_arrays(config=config, **build_kwargs)
    else:
        model = build_model(dataset_or_arrays, config=config)

    if cores is None:
        # PyMC's own auto-detection can divide by zero on hosts where
        # os.cpu_count() reports 1 (seen in this sandbox) — compute it
        # ourselves rather than pass cores=None through.
        import os

        cores = max(1, min(chains, os.cpu_count() or 1))

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
        )
    return model, idata
