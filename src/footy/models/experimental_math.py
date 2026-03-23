"""Experimental mathematical models for football forecasting.

DEPRECATED: This module is maintained for backward compatibility.
New code should import from footy.models.math submodules directly:

    from footy.models.math.kalman import kalman_attack_defence_update
    from footy.models.math.weibull import build_weibull_count_matrix
    from footy.models.math.attention import recency_attention_weights
    from footy.models.math.bma import bayesian_model_average

All experimental functionality has been reorganized into focused sub-modules:
- kalman: Kalman filtering for dynamic team strength
- negative_binomial: Negative Binomial distributions (also in distributions.py)
- weibull: Weibull count models
- attention: Recency-attention weighting and ensemble blending
- double_poisson: Efron's Double Poisson (also in distributions.py)
- bma: Bayesian Model Averaging
- ensemble: Ensemble methods, calibration, unified predictions
- scoring: Proper scoring rules and calibration metrics
- simulation: Monte Carlo and Bradley-Terry models
- distributions: Skellam, bivariate Poisson, COM-Poisson, etc.

This module re-exports all functions for legacy compatibility.
"""
from __future__ import annotations

# Re-export from math submodules for backward compatibility
from footy.models.math.attention import (
    information_gain_from_odds,
    recency_attention_weights,
    score_matrix_entropy,
    variance_adjusted_blend,
)
from footy.models.math.bma import (
    bayesian_model_average,
    compute_model_likelihoods,
    opponent_adjusted_xg,
)
from footy.models.math.distributions import (
    build_negative_binomial_matrix,
    build_zip_score_matrix,
    negative_binomial_pmf,
    zero_inflated_poisson_pmf,
)
from footy.models.math.double_poisson import (
    build_double_poisson_matrix,
    double_poisson_pmf,
)
from footy.models.math.kalman import (
    KalmanStrengthState,
    kalman_attack_defence_update,
    kalman_batch_update,
)
from footy.models.math.weibull import (
    build_weibull_count_matrix,
    weibull_count_pmf,
)
from footy.models.math.ensemble import (
    expected_points_from_score_matrix,
    ordered_probit_1x2,
    skellam_match_probs,
    bradley_terry_davidson,
    pythagorean_win_pct,
    regression_to_mean_signal,
    match_state_transition_probs,
    markov_chain_simulate,
    bivariate_normal_goal_diff,
    goal_timing_intensity,
    build_all_score_matrices,
    aggregate_model_predictions,
    SelfCalibratingEnsemble,
    ensemble_disagreement,
    unified_prediction,
)
from footy.models.math.scoring import (
    expected_calibration_error,
)

__all__ = [
    # Kalman
    "KalmanStrengthState",
    "kalman_attack_defence_update",
    "kalman_batch_update",
    # ZIP (from distributions)
    "zero_inflated_poisson_pmf",
    "build_zip_score_matrix",
    # Negative Binomial (from distributions)
    "negative_binomial_pmf",
    "build_negative_binomial_matrix",
    # Weibull
    "weibull_count_pmf",
    "build_weibull_count_matrix",
    # Double Poisson
    "double_poisson_pmf",
    "build_double_poisson_matrix",
    # Attention and Ensemble
    "recency_attention_weights",
    "variance_adjusted_blend",
    "score_matrix_entropy",
    "information_gain_from_odds",
    # BMA
    "bayesian_model_average",
    "compute_model_likelihoods",
    "opponent_adjusted_xg",
    # Ensemble & Calibration
    "expected_points_from_score_matrix",
    "score_matrix_entropy",
    "ordered_probit_1x2",
    "skellam_match_probs",
    "bradley_terry_davidson",
    "pythagorean_win_pct",
    "regression_to_mean_signal",
    "match_state_transition_probs",
    "markov_chain_simulate",
    "bivariate_normal_goal_diff",
    "goal_timing_intensity",
    "build_all_score_matrices",
    "aggregate_model_predictions",
    "SelfCalibratingEnsemble",
    "ensemble_disagreement",
    "expected_calibration_error",
    "unified_prediction",
]
