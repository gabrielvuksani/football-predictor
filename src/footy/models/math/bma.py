"""Bayesian Model Averaging for combining multiple football prediction models.

Implements principled Bayesian combination of score distribution models
(Dixon-Coles, Bivariate Poisson, Copulas, COM-Poisson, etc.) using
posterior model weights based on likelihood and prior beliefs.

References:
    Hoeting et al. (1999) "Bayesian Model Averaging: A Tutorial"
    Raftery et al. (1997) "Bayesian Model Averaging for Linear Regression Models"
    Madigan & Raftery (1994) "Model Selection and Accounting for Model Uncertainty"
"""
from __future__ import annotations

import numpy as np


def bayesian_model_average(
    score_matrices: list[np.ndarray],
    model_log_likelihoods: list[float] | None = None,
    prior_weights: list[float] | None = None,
) -> np.ndarray:
    """Bayesian Model Averaging across multiple score distribution models.

    Combines probabilistic forecasts from different models (Dixon-Coles,
    Bivariate Poisson, Copulas, COM-Poisson, etc.) using:

    P(outcome | data) = Σ P(outcome | model_i, data) · P(model_i | data)

    where posterior model weights are:

    P(model_i | data) ∝ L(data | model_i) · P(model_i)

    This is theoretically superior to simple averaging because:
    1. Better models (higher likelihood) get more weight
    2. Incorporates prior beliefs about model quality
    3. Provides uncertainty calibration

    Args:
        score_matrices: List of N score matrices from different models
                       Each is shape (max_goals+1, max_goals+1)
        model_log_likelihoods: Log-likelihood of each model given data
                              If None, uses uniform likelihoods (equal weight)
        prior_weights: Prior probability for each model
                      If None, uses uniform priors

    Returns:
        Weighted average of score matrices

    Raises:
        ValueError: If score_matrices is empty

    Example:
        >>> dc_matrix = build_dc_score_matrix(1.5, 1.2, rho=-0.13)
        >>> bp_matrix = build_bivariate_poisson_matrix(1.5, 1.2, lambda_3=0.1)
        >>> copula_matrix = build_copula_score_matrix(1.5, 1.2, theta=-2.0)
        >>>
        >>> # If we have model likelihoods from cross-validation
        >>> likelihoods = [-10.5, -10.2, -10.8]
        >>> bma_matrix = bayesian_model_average(
        ...     [dc_matrix, bp_matrix, copula_matrix],
        ...     model_log_likelihoods=likelihoods
        ... )

    References:
        Hoeting et al. (1999) "Bayesian Model Averaging: A Tutorial"
    """
    n_models = len(score_matrices)

    if n_models == 0:
        raise ValueError("Need at least one score matrix")

    if n_models == 1:
        return score_matrices[0].copy()

    # Convert likelihoods to numpy array, defaulting to zeros (uniform)
    if model_log_likelihoods is not None:
        log_liks = np.array(model_log_likelihoods, dtype=float)
    else:
        log_liks = np.zeros(n_models)

    # Convert priors to log space, defaulting to zeros (uniform)
    if prior_weights is not None:
        log_priors = np.log(np.array(prior_weights, dtype=float) + 1e-15)
    else:
        log_priors = np.zeros(n_models)

    # Posterior log probabilities (unnormalized)
    log_posteriors = log_liks + log_priors

    # Normalize to avoid numerical overflow
    # Use log-sum-exp trick
    max_log = log_posteriors.max()
    weights = np.exp(log_posteriors - max_log)
    weights = weights / weights.sum()

    # Weighted combination of matrices
    result = np.zeros_like(score_matrices[0])
    for mat, w in zip(score_matrices, weights):
        result += w * mat

    # Renormalize full matrix (probabilities should sum to 1)
    total = result.sum()
    if total > 0:
        result /= total

    return result


def compute_model_likelihoods(
    score_matrices_history: dict[str, list[np.ndarray]],
    actual_home_goals: np.ndarray | list[int],
    actual_away_goals: np.ndarray | list[int],
) -> dict[str, float]:
    """Compute log-likelihoods for each score model from historical predictions.

    For each model, evaluates how well its predicted score distributions matched
    actual match outcomes. Models that assigned higher probability to the
    observed scorelines get higher log-likelihood, and therefore more weight
    in Bayesian Model Averaging.

    Log-likelihood for model m:
        LL_m = sum_i log P_m(home_i, away_i)

    where P_m(h, a) is the probability model m assigned to scoreline (h, a)
    for match i.

    Args:
        score_matrices_history: Dict mapping model_name to a list of score matrices,
                                one per historical match. Each matrix is shape
                                (max_goals+1, max_goals+1).
        actual_home_goals: Array of actual home goals for each match.
        actual_away_goals: Array of actual away goals for each match.

    Returns:
        Dict mapping model_name to total log-likelihood (higher = better fit).

    Example:
        >>> from footy.models.math.bma import compute_model_likelihoods
        >>> # Suppose we have 3 past matches with predictions from 2 models
        >>> history = {
        ...     "dixon_coles": [dc_mat_1, dc_mat_2, dc_mat_3],
        ...     "bivariate_poisson": [bp_mat_1, bp_mat_2, bp_mat_3],
        ... }
        >>> home_goals = [2, 0, 1]
        >>> away_goals = [1, 0, 3]
        >>> lls = compute_model_likelihoods(history, home_goals, away_goals)
        >>> # lls = {"dixon_coles": -4.32, "bivariate_poisson": -4.67}
    """
    home_g = np.asarray(actual_home_goals, dtype=int)
    away_g = np.asarray(actual_away_goals, dtype=int)
    n_matches = len(home_g)

    if n_matches == 0:
        return {name: 0.0 for name in score_matrices_history}

    # Floor for log(0) protection
    LOG_FLOOR = 1e-12

    log_likelihoods: dict[str, float] = {}

    for model_name, matrices in score_matrices_history.items():
        if len(matrices) != n_matches:
            # Mismatch: skip this model, give it neutral likelihood
            log_likelihoods[model_name] = 0.0
            continue

        total_ll = 0.0
        for i in range(n_matches):
            mat = matrices[i]
            h, a = int(home_g[i]), int(away_g[i])

            # Clamp to matrix bounds (e.g., if actual goals > max_goals)
            max_idx = mat.shape[0] - 1
            h_clamped = min(h, max_idx)
            a_clamped = min(a, max_idx)

            prob = float(mat[h_clamped, a_clamped])
            total_ll += np.log(max(prob, LOG_FLOOR))

        log_likelihoods[model_name] = total_ll

    return log_likelihoods


def opponent_adjusted_xg(
    raw_xg: float,
    opponent_defensive_strength: float,
    league_average_xg: float = 1.35,
) -> float:
    """Adjust raw expected goals by opponent defensive strength.

    Implements SRS (Simple Rating System) style adjustment:
    Adjusted_xG = raw_xG × (league_avg / opponent_defence)

    This corrects for opponent quality:
    - Weak opponents (strength << league_avg): adjustment > 1 (xG increases)
    - Strong opponents (strength > league_avg): adjustment < 1 (xG decreases)

    Common use case: teams padding xG stats against weak opponents should
    have their xG adjusted downward; teams creating chances against
    elite defences should be adjusted upward.

    Args:
        raw_xg: Observed expected goals from model
        opponent_defensive_strength: Opponent's defensive strength rating
                                    (same scale as xG, e.g., 1.0-2.0)
        league_average_xg: League-wide average xG conceded per match
                          Default: 1.35 (typical value)

    Returns:
        Adjusted expected goals value
    """
    if opponent_defensive_strength <= 0:
        return raw_xg

    # Compute adjustment ratio
    adjustment = league_average_xg / max(opponent_defensive_strength, 0.3)

    # Clamp adjustment to reasonable range (0.5x to 2.0x)
    adjustment = max(0.5, min(2.0, adjustment))

    return raw_xg * adjustment
