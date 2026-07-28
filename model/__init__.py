"""
model/ — the minimal hierarchical Dixon-Coles Poisson model.

Consumes a `features.MatchDataset` (see feature_extraction_handoff.md) and
produces a fitted PyMC model + posterior, per forecast_engine_design_notes.md
("Prediction model" section) and build_log.md next-step #2:

    "Minimal Dixon-Coles Poisson model (PyMC) — core hierarchical model
    only, no covariates yet, fit against the small test chunk. Goal is
    proving the full pipeline (DB -> features -> model -> predictions)
    end to end, not a production-quality fit."

Public API:
    build_model              — construct the PyMC model from a MatchDataset
    build_model_from_arrays  — same, from raw arrays (no MatchDataset needed)
    fit                      — build_model + pm.sample, returns InferenceData
    match_outcome_probs      — posterior -> P(home win)/P(draw)/P(away win)
    scoreline_distribution   — posterior -> full scoreline PMF (for RPS later)
"""

from .dixon_coles import build_model, build_model_from_arrays, fit, ModelConfig
from .predict import match_outcome_probs, scoreline_distribution, posterior_rates
from .tau import dixon_coles_tau

__all__ = [
    "build_model",
    "build_model_from_arrays",
    "fit",
    "ModelConfig",
    "match_outcome_probs",
    "scoreline_distribution",
    "posterior_rates",
    "dixon_coles_tau",
]