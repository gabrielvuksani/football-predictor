"""Advanced mathematical functions for football prediction.

DEPRECATED: This module is maintained for backward compatibility.
New code should import from footy.models.math submodules directly:

    from footy.models.math.distributions import skellam_probs
    from footy.models.math.scoring import ranked_probability_score
    from footy.models.math.simulation import monte_carlo_simulate

All functionality has been reorganized into focused sub-modules:
- distributions: Poisson, Skellam, Bivariate Poisson, COM-Poisson, ZIP, NB
- empirical_bayes: Beta-Binomial shrinkage and league priors
- copulas: Frank, Clayton, Gumbel, Gaussian copulas
- scoring: RPS, Brier, log loss, calibration metrics
- simulation: Monte Carlo, Bradley-Terry, Platt scaling

This module re-exports all functions for legacy compatibility.
"""
from __future__ import annotations

# Re-export all functions from submodules for backward compatibility
from footy.models.math import (
    adaptive_ewma,
    apply_temperature_scaling,
    bivariate_poisson_pmf,
    beta_binomial_shrink,
    bradley_terry_probs,
    brier_score,
    build_bivariate_poisson_matrix,
    build_com_poisson_matrix,
    build_copula_score_matrix,
    build_dc_score_matrix,
    build_negative_binomial_matrix,
    build_zip_score_matrix,
    com_poisson_pmf,
    dixon_coles_tau,
    estimate_rho_mle,
    expected_calibration_error,
    extract_match_probs,
    find_optimal_temperature,
    inv_logit,
    jensen_shannon_divergence,
    kl_divergence,
    league_specific_prior,
    log_loss,
    log_transform,
    logit,
    logit_space_delta,
    monte_carlo_simulate,
    multi_expert_jsd,
    negative_binomial_pmf,
    odds_dispersion,
    odds_entropy,
    platt_scale,
    ranked_probability_score,
    remove_overround,
    rps_from_result,
    schedule_difficulty,
    skellam_probs,
    tanh_transform,
    zero_inflated_poisson_pmf,
)

__all__ = [
    "adaptive_ewma",
    "apply_temperature_scaling",
    "bivariate_poisson_pmf",
    "beta_binomial_shrink",
    "bradley_terry_probs",
    "brier_score",
    "build_bivariate_poisson_matrix",
    "build_com_poisson_matrix",
    "build_copula_score_matrix",
    "build_dc_score_matrix",
    "build_negative_binomial_matrix",
    "build_zip_score_matrix",
    "com_poisson_pmf",
    "dixon_coles_tau",
    "estimate_rho_mle",
    "expected_calibration_error",
    "extract_match_probs",
    "find_optimal_temperature",
    "inv_logit",
    "jensen_shannon_divergence",
    "kl_divergence",
    "league_specific_prior",
    "log_loss",
    "log_transform",
    "logit",
    "logit_space_delta",
    "monte_carlo_simulate",
    "multi_expert_jsd",
    "negative_binomial_pmf",
    "odds_dispersion",
    "odds_entropy",
    "platt_scale",
    "ranked_probability_score",
    "remove_overround",
    "rps_from_result",
    "schedule_difficulty",
    "skellam_probs",
    "tanh_transform",
    "zero_inflated_poisson_pmf",
]
