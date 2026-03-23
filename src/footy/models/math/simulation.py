"""Monte Carlo simulation and Bradley-Terry models for football prediction.

Implements:
- Monte Carlo scoreline simulation with Dixon-Coles adjustment
- Bradley-Terry pairwise comparison model
- Platt scaling for probability calibration
- Temperature-based calibration optimization

References:
    Bradley & Terry (1952) "Rank Analysis of Incomplete Block Designs"
    Agresti (2002) "Categorical Data Analysis"
    Platt (1999) "Probabilistic Outputs for Support Vector Machines"
    Guo et al. (2017) "On Calibration of Modern Neural Networks"
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# MONTE CARLO SCORELINE SIMULATION
# ═══════════════════════════════════════════════════════════════════
def monte_carlo_simulate(
    lambda_h: float,
    lambda_a: float,
    rho: float = -0.13,
    n_sims: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Monte Carlo scoreline simulation with Dixon-Coles correlation.

    Draws n_sims scorelines from a bivariate Poisson model with
    Dixon-Coles low-score correction, then derives empirical probabilities.

    Args:
        lambda_h: Home expected goals
        lambda_a: Away expected goals
        rho: Dixon-Coles correlation parameter
        n_sims: Number of simulations
        rng: Random number generator (for reproducibility)

    Returns:
        Dict with p_home, p_draw, p_away, p_btts, p_o15, p_o25, p_o35,
        p_cs, mean_total, var_total, most_likely_score
    """

    if rng is None:
        rng = np.random.default_rng()  # v12: use random seed in production

    # Generate correlated Poisson draws using rejection sampling
    # with Dixon-Coles adjustment
    h_goals = rng.poisson(lambda_h, n_sims)
    a_goals = rng.poisson(lambda_a, n_sims)

    # Apply DC correction via acceptance-rejection
    # For each simulated scoreline, compute acceptance probability
    # proportional to τ(h, a) / max_τ
    if abs(rho) > 1e-6:
        def dixon_coles_tau(hg: int, ag: int) -> float:
            """DC adjustment factor τ(x, y, λ, μ, ρ)."""
            if hg == 0 and ag == 0:
                return max(0.0, 1.0 - lambda_h * lambda_a * rho)
            elif hg == 1 and ag == 0:
                return max(0.0, 1.0 + lambda_h * rho)
            elif hg == 0 and ag == 1:
                return max(0.0, 1.0 + lambda_a * rho)
            elif hg == 1 and ag == 1:
                return max(0.0, 1.0 - rho)
            return 1.0

        accept_probs = np.array([
            dixon_coles_tau(int(h), int(a))
            for h, a in zip(h_goals, a_goals)
        ])
        accept_probs = np.clip(accept_probs, 0, 2)  # safety clamp
        # Normalize to max 1
        max_tau = accept_probs.max()
        if max_tau > 0:
            accept_probs /= max_tau
        # Accept/reject
        u = rng.uniform(0, 1, n_sims)
        mask = u < accept_probs
        h_goals = h_goals[mask]
        a_goals = a_goals[mask]

    n_accepted = len(h_goals)
    if n_accepted < 100:
        # Fallback to analytical if too few accepted
        return {
            "mc_p_home": 0.0, "mc_p_draw": 0.0, "mc_p_away": 0.0,
            "mc_btts": 0.0, "mc_o15": 0.0, "mc_o25": 0.0, "mc_o35": 0.0,
            "mc_cs": 0.0, "mc_mean_total": lambda_h + lambda_a,
            "mc_var_total": lambda_h + lambda_a,
            "mc_n_sims": 0,
        }

    total = h_goals + a_goals
    diff = h_goals.astype(int) - a_goals.astype(int)

    p_home = float(np.mean(diff > 0))
    p_draw = float(np.mean(diff == 0))
    p_away = float(np.mean(diff < 0))
    p_btts = float(np.mean((h_goals > 0) & (a_goals > 0)))
    p_o15 = float(np.mean(total > 1.5))
    p_o25 = float(np.mean(total > 2.5))
    p_o35 = float(np.mean(total > 3.5))
    p_cs = float(np.mean((h_goals == 0) | (a_goals == 0)))
    mean_total = float(np.mean(total))
    var_total = float(np.var(total))

    # Most likely score
    score_counts = Counter(zip(h_goals.tolist(), a_goals.tolist()))
    ml_score = score_counts.most_common(1)[0][0] if score_counts else (1, 1)

    return {
        "mc_p_home": p_home, "mc_p_draw": p_draw, "mc_p_away": p_away,
        "mc_btts": p_btts, "mc_o15": p_o15, "mc_o25": p_o25, "mc_o35": p_o35,
        "mc_cs": p_cs, "mc_mean_total": mean_total, "mc_var_total": var_total,
        "mc_n_sims": n_accepted,
        "mc_ml_hg": ml_score[0], "mc_ml_ag": ml_score[1],
    }


# ═══════════════════════════════════════════════════════════════════
# BRADLEY-TERRY MODEL
# ═══════════════════════════════════════════════════════════════════
def bradley_terry_probs(
    strength_h: float,
    strength_a: float,
    home_adv: float = 0.3,
    draw_factor: float = 0.25,
) -> tuple[float, float, float]:
    """Bradley-Terry model with home advantage and draw extension.

    The Bradley-Terry (1952) model gives:
        P(i beats j) = π_i / (π_i + π_j)

    Extended with home advantage (Agresti 2002):
        P(Home wins) = θ·π_h / (θ·π_h + π_a)

    Extended with draws (Davidson 1970):
        P(Draw) = ν·√(π_h·π_a) / (θ·π_h + π_a + ν·√(π_h·π_a))

    where:
        π_i = exp(strength_i) is team i's strength
        θ = exp(home_adv) is the home advantage factor
        ν = draw_factor controls draw probability

    This is algebraically different from Elo and can capture
    different aspects of team comparison.

    Args:
        strength_h: Home team log-strength
        strength_a: Away team log-strength
        home_adv: Log home advantage (θ = e^{home_adv})
        draw_factor: Draw intensity (ν ≥ 0)

    Returns:
        (p_home, p_draw, p_away) summing to 1.0
    """
    pi_h = math.exp(strength_h)
    pi_a = math.exp(strength_a)
    theta = math.exp(home_adv)

    numerator_h = theta * pi_h
    numerator_a = pi_a
    numerator_d = draw_factor * math.sqrt(pi_h * pi_a)

    total = numerator_h + numerator_a + numerator_d
    if total <= 0:
        return (1.0 / 3, 1.0 / 3, 1.0 / 3)

    p_h = numerator_h / total
    p_d = numerator_d / total
    p_a = numerator_a / total

    return (p_h, p_d, p_a)


# ═══════════════════════════════════════════════════════════════════
# PLATT SCALING (TEMPERATURE CALIBRATION)
# ═══════════════════════════════════════════════════════════════════
def platt_scale(
    probs: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Platt/temperature scaling for probability calibration.

    Applies temperature scaling in logit-space:
        p_calibrated = softmax(logit(p) / T)

    Temperature effects:
        T = 1.0: no change (identity)
        T > 1.0: softer probabilities (less confident)
        T < 1.0: sharper probabilities (more confident)

    This is a post-hoc calibration method that preserves the ranking
    of predicted classes while adjusting confidence levels.

    Guo et al. (2017) "On Calibration of Modern Neural Networks" showed
    temperature scaling is highly effective for multi-class calibration.

    Args:
        probs: Array of shape (N, K) or (K,) — probabilities
        temperature: Temperature parameter T > 0

    Returns:
        Calibrated probabilities, same shape as input
    """
    eps = 1e-8
    single = probs.ndim == 1
    if single:
        probs = probs[np.newaxis, :]

    # Log-probability space (correct for multi-class temperature scaling)
    p_clipped = np.clip(probs, eps, 1.0)
    log_probs = np.log(p_clipped)

    # Scale by temperature
    scaled_log_probs = log_probs / max(temperature, eps)

    # Softmax
    shifted = scaled_log_probs - scaled_log_probs.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    result = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return result[0] if single else result


def find_optimal_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    t_range: tuple[float, float] = (0.1, 10.0),
) -> float:
    """Find optimal temperature T that minimizes NLL on validation set.

    Delegates to the upgraded implementation in
    ``footy.models.math.scoring.find_optimal_temperature`` which uses
    L-BFGS-B with proper multiclass NLL minimization.

    This wrapper is kept for backward compatibility.  New code should
    import directly from ``footy.models.math.scoring``.

    Args:
        probs: Predicted probabilities or logits (N, K)
        labels: True class indices (N,) — 0, 1, or 2
        t_range: Search bounds for temperature

    Returns:
        Optimal temperature value
    """
    from footy.models.math.scoring import (
        find_optimal_temperature as _find_optimal_temperature,
    )

    return _find_optimal_temperature(probs, labels, bounds=t_range)


# ═══════════════════════════════════════════════════════════════════
# MATCH PROBABILITY EXTRACTION
# ═══════════════════════════════════════════════════════════════════
def extract_match_probs(mat: np.ndarray) -> dict[str, float]:
    """Extract comprehensive match probabilities from ANY score matrix.

    Works with: Dixon-Coles, bivariate Poisson, Frank copula, COM-Poisson.

    Returns dict with:
        1X2 probabilities, BTTS, Over/Under lines, clean sheet,
        expected goals, most likely scoreline, and dispersion metrics.
    """
    n = mat.shape[0]
    goals_range = np.arange(n)

    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())

    # BTTS (both teams to score)
    p_btts = float(mat[1:, 1:].sum())

    # Over/Under lines
    total_goals = np.add.outer(goals_range, goals_range)
    p_o15 = float(mat[total_goals >= 2].sum())
    p_o25 = float(mat[total_goals >= 3].sum())
    p_o35 = float(mat[total_goals >= 4].sum())
    p_o45 = float(mat[total_goals >= 5].sum())

    # Clean sheet
    p_cs_home = float(mat[:, 0].sum())  # away scores 0
    p_cs_away = float(mat[0, :].sum())  # home scores 0

    # Expected goals from the matrix
    eg_h = float(np.sum(goals_range[:, None] * mat))
    eg_a = float(np.sum(goals_range[None, :] * mat))

    # Variance of goals (for dispersion detection)
    var_h = float(np.sum(goals_range[:, None] ** 2 * mat)) - eg_h ** 2
    var_a = float(np.sum(goals_range[None, :] ** 2 * mat)) - eg_a ** 2

    # Dispersion index (variance / mean) — > 1 = over-dispersed
    disp_h = var_h / max(eg_h, 1e-6)
    disp_a = var_a / max(eg_a, 1e-6)

    # Most likely scoreline
    flat_idx = int(np.argmax(mat))
    ml_h, ml_a = divmod(flat_idx, n)

    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "p_btts": p_btts,
        "p_o15": p_o15, "p_o25": p_o25, "p_o35": p_o35, "p_o45": p_o45,
        "p_cs_home": p_cs_home, "p_cs_away": p_cs_away,
        "eg_home": eg_h, "eg_away": eg_a,
        "var_home": var_h, "var_away": var_a,
        "disp_home": disp_h, "disp_away": disp_a,
        "ml_score_h": ml_h, "ml_score_a": ml_a,
    }
