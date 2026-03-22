"""Proper scoring rules and calibration metrics for football prediction.

Implements:
- Ranked Probability Score (RPS) for ordered categorical outcomes
- Brier score for binary classification
- Log loss for probabilistic calibration
- Calibration metrics and error measurement

References:
    Epstein (1969) "A Scoring System for Probability Forecasts of
        Ranked Categories"
    Constantinou & Fenton (2012) "Solving the Problem of Inadequate Scoring
        Rules for Assessing Probabilistic Football Forecast Models"
    Guo et al. (2017) "On Calibration of Modern Neural Networks"
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# RANKED PROBABILITY SCORE (RPS)
# ═══════════════════════════════════════════════════════════════════
def ranked_probability_score(
    probs: Sequence[float],
    outcome_idx: int,
) -> float:
    """Ranked Probability Score for ordered categorical outcomes.

    The RPS is the proper scoring rule for ordered categories.
    For football: categories are [Home, Draw, Away] — ordered by goal diff.

    RPS = 1/(K-1) · Σ_{k=1}^{K-1} (CDF_pred(k) - CDF_actual(k))²

    where CDF_pred(k) = Σ_{j=1}^{k} p_j  and CDF_actual is step function.

    Properties:
        - RPS ∈ [0, 1], lower is better
        - Unlike logloss, RPS penalizes predictions that are "close"
          less than predictions that are "far" (e.g., predicting Away
          when Home wins is worse than predicting Draw when Home wins)
        - This is critical for football where the ordering matters

    Reference: Epstein (1969), Constantinou & Fenton (2012)

    Args:
        probs: Predicted probabilities [p_home, p_draw, p_away]
        outcome_idx: Actual outcome index (0=Home, 1=Draw, 2=Away)

    Returns:
        RPS score (lower = better prediction)
    """
    k = len(probs)
    if k < 2:
        return 0.0

    cdf_pred = 0.0
    rps = 0.0
    for j in range(k - 1):
        cdf_pred += probs[j]
        cdf_actual = 1.0 if j >= outcome_idx else 0.0
        rps += (cdf_pred - cdf_actual) ** 2

    return rps / (k - 1)


def rps_from_result(
    probs: Sequence[float],
    home_goals: int,
    away_goals: int,
) -> float:
    """Convenience wrapper: compute RPS from match result.

    Maps result to outcome_idx: home win → 0, draw → 1, away win → 2.
    """
    if home_goals > away_goals:
        idx = 0
    elif home_goals == away_goals:
        idx = 1
    else:
        idx = 2
    return ranked_probability_score(probs, idx)


def brier_score(
    probs: Sequence[float],
    outcome_idx: int,
) -> float:
    """Brier score for multi-class classification.

    BS = 1/K · Σ (p_i - y_i)²

    where y_i is 1 for true class, 0 otherwise.

    Args:
        probs: Predicted probabilities
        outcome_idx: True outcome index

    Returns:
        Brier score in [0, 1], lower is better
    """
    k = len(probs)
    bs = 0.0
    for i in range(k):
        y_i = 1.0 if i == outcome_idx else 0.0
        bs += (probs[i] - y_i) ** 2
    return bs / k


def log_loss(
    probs: Sequence[float],
    outcome_idx: int,
    eps: float = 1e-15,
) -> float:
    """Log loss (cross-entropy) for probabilistic prediction.

    LogLoss = -log(p_true)

    Args:
        probs: Predicted probabilities
        outcome_idx: True outcome index
        eps: Smoothing to avoid log(0)

    Returns:
        Log loss value (lower is better, unbounded)
    """
    p_true = max(probs[outcome_idx], eps)
    return -math.log(p_true)


# ═══════════════════════════════════════════════════════════════════
# CALIBRATION & DIVERGENCE METRICS
# ═══════════════════════════════════════════════════════════════════
def kl_divergence(
    p: Sequence[float],
    q: Sequence[float],
    eps: float = 1e-12,
) -> float:
    """KL divergence D_KL(P || Q) = Σ p_i · ln(p_i / q_i).

    Measures how much expert P diverges from ensemble Q.
    Higher values → expert disagrees more strongly with consensus.

    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Smoothing constant

    Returns:
        KL divergence value (non-negative, unbounded)
    """
    return sum(
        max(pi, eps) * math.log(max(pi, eps) / max(qi, eps))
        for pi, qi in zip(p, q)
    )


def jensen_shannon_divergence(
    p: Sequence[float],
    q: Sequence[float],
    eps: float = 1e-12,
) -> float:
    """Jensen-Shannon divergence — symmetric, bounded version of KL.

    JSD(P || Q) = ½ KL(P || M) + ½ KL(Q || M),  where M = ½(P + Q)

    Properties:
        JSD ∈ [0, ln(2)] ≈ [0, 0.693]
        JSD is symmetric: JSD(P||Q) = JSD(Q||P)
        √JSD is a proper metric (satisfies triangle inequality)

    Better than KL divergence for measuring expert disagreement because:
    1. Symmetric (expert A vs B = expert B vs A)
    2. Always finite (no division-by-zero risk)
    3. Bounded (easier to interpret as a feature)

    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Smoothing constant

    Returns:
        JSD value in [0, ln(2)]
    """
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    kl_pm = sum(
        max(pi, eps) * math.log(max(pi, eps) / max(mi, eps))
        for pi, mi in zip(p, m)
    )
    kl_qm = sum(
        max(qi, eps) * math.log(max(qi, eps) / max(mi, eps))
        for qi, mi in zip(q, m)
    )
    return 0.5 * kl_pm + 0.5 * kl_qm


def multi_expert_jsd(
    expert_probs: list[Sequence[float]],
    eps: float = 1e-12,
) -> float:
    """Generalized JSD across N expert distributions.

    JSD(P₁, ..., Pₙ) = H(M) - 1/n Σ H(Pᵢ)

    where M = 1/n Σ Pᵢ  and H is Shannon entropy.

    This measures total disagreement across all experts simultaneously,
    rather than pairwise. Higher values → more confusion / uncertainty.

    Args:
        expert_probs: List of N probability distributions
        eps: Smoothing constant

    Returns:
        Multi-expert JSD value
    """
    if not expert_probs:
        return 0.0
    n = len(expert_probs)
    k = len(expert_probs[0])

    # Mixture distribution M
    m = [sum(expert_probs[i][j] for i in range(n)) / n for j in range(k)]

    # H(M) — entropy of mixture
    h_m = -sum(max(mi, eps) * math.log(max(mi, eps)) for mi in m)

    # Average entropy of individual experts
    avg_h = 0.0
    for p in expert_probs:
        avg_h -= sum(max(pi, eps) * math.log(max(pi, eps)) for pi in p) / n

    return max(0.0, h_m - avg_h)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) for probability predictions.

    ECE = 1/N Σ |accuracy_bin - confidence_bin| · |bin|

    Measures whether predicted probabilities match actual frequencies.
    ECE ∈ [0, 1], lower is better (perfectly calibrated = 0).

    For 1D input (binary case), treats as probabilities of positive class
    and computes Brier-like calibration metric.

    Args:
        probs: Predicted probabilities (N, K) or (N,) for binary
        labels: True class indices (N,) or binary labels (0/1)
        n_bins: Number of bins for calibration curve

    Returns:
        Expected Calibration Error value
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)

    if probs.ndim == 1:
        # Binary case: treat as probabilities of positive class
        # Compute Brier score as calibration measure
        return float(np.mean((probs - labels) ** 2))

    n_samples = len(labels)
    bin_width = 1.0 / n_bins
    ece = 0.0

    for bin_idx in range(n_bins):
        lower = bin_idx * bin_width
        upper = (bin_idx + 1) * bin_width

        # Find samples in this confidence bin
        max_probs = np.max(probs, axis=1)
        in_bin = (max_probs >= lower) & (max_probs < upper)
        bin_size = np.sum(in_bin)

        if bin_size == 0:
            continue

        # Average confidence in bin
        avg_confidence = np.mean(max_probs[in_bin])

        # Accuracy in bin
        predictions = np.argmax(probs[in_bin], axis=1)
        accuracy = np.mean(predictions == labels[in_bin])

        # Weighted contribution
        ece += (bin_size / n_samples) * abs(accuracy - avg_confidence)

    return ece


# ═══════════════════════════════════════════════════════════════════
# LOGIT AND ODDS TRANSFORMS
# ═══════════════════════════════════════════════════════════════════
def logit(p: float, eps: float = 1e-7) -> float:
    """Logit transform: log-odds = ln(p / (1 - p)).

    Maps probabilities p ∈ (0, 1) to log-odds ∈ ℝ.

    The logit is useful for:
    - Modeling probability updates in log-odds space (additive)
    - Normalizing skewed probability distributions
    - Feature engineering for linear models

    Args:
        p: Probability in (0, 1)
        eps: Clipping to avoid log(0)

    Returns:
        Log-odds value (unbounded)

    Example:
        >>> logit(0.5)
        0.0
        >>> logit(0.75)
        1.0986...
    """
    p = np.clip(float(p), eps, 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


def inv_logit(x: float) -> float:
    """Inverse logit (sigmoid): σ(x) = 1 / (1 + exp(-x)).

    Maps log-odds x ∈ ℝ back to probabilities p ∈ (0, 1).

    Args:
        x: Log-odds value (unbounded)

    Returns:
        Probability in (0, 1)

    Example:
        >>> inv_logit(0.0)
        0.5
        >>> inv_logit(np.log(3))
        0.75
    """
    x = float(x)
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))


def logit_space_delta(p1: float, p2: float) -> float:
    """Difference in logit space: logit(p1) - logit(p2).

    Measures probability change in the more-interpretable log-odds scale.
    Useful for:
    - Quantifying "shift in value" when probability changes from p2 to p1
    - Linear models naturally work in log-odds space
    - Comparing very small/large probability updates

    With logit space:
        - Change 0.49 → 0.51 = -0.04 change in odds
        - Change 0.01 → 0.03 = 1.10 change in odds (much more sensitive)

    Args:
        p1: First probability
        p2: Second probability

    Returns:
        Logit-space delta (higher p1 → positive, higher p2 → negative)

    Example:
        >>> logit_space_delta(0.6, 0.5)
        0.405...
        >>> logit_space_delta(0.5, 0.6)
        -0.405...
    """
    return logit(p1) - logit(p2)


def remove_overround(
    probs: Sequence[float],
    eps: float = 1e-12,
) -> list[float]:
    """Remove bookmaker margin (overround) from implied probabilities.

    Bookmakers build in a margin (e.g., 1.95-2.05 odds instead of 2.00)
    so that implied probabilities sum to > 1.0. This "overround" is the
    bookmaker's profit margin.

    To extract fair probabilities: p_fair = p_implied / Σ p_implied

    Args:
        probs: Implied probabilities (typically sum to 1.03-1.10)
        eps: Smoothing to avoid division by zero

    Returns:
        Fair probabilities summing to exactly 1.0

    Example:
        >>> odds = [2.0, 3.5, 2.4]  # Bet365 closing odds
        >>> imp = [1/o for o in odds]  # [0.5, 0.286, 0.417] → sum=1.203
        >>> fair = remove_overround(imp)
        >>> sum(fair)
        1.0
    """
    total = sum(probs) + eps
    return [float(p / total) for p in probs]


def odds_entropy(probs: Sequence[float], eps: float = 1e-12) -> float:
    """Shannon entropy of probability distribution.

    H(P) = -Σ p_i · log(p_i)

    Measures uncertainty:
        - Uniform [1/3, 1/3, 1/3] → H ≈ 1.099 (maximum for 3-way)
        - Skewed [0.8, 0.1, 0.1] → H ≈ 0.639 (lower)

    Used to quantify "confidence" in market odds:
        - High entropy = uncertain market (conflicting views)
        - Low entropy = confident market (consensus)

    Args:
        probs: Probability distribution
        eps: Smoothing constant

    Returns:
        Shannon entropy value (non-negative)

    Example:
        >>> odds_entropy([1/3, 1/3, 1/3])
        1.0986...
        >>> odds_entropy([0.8, 0.1, 0.1])
        0.6394...
    """
    return float(-sum(
        max(p, eps) * np.log(max(p, eps)) for p in probs
    ))


def odds_dispersion(
    probs: Sequence[float],
) -> float:
    """Measure dispersion (spread) of a probability distribution.

    Uses standard deviation of probabilities as a proxy for spread:
        σ = sqrt(Σ (p_i - mean(p))²)

    Interpretation:
        - σ ≈ 0 → concentrated (one outcome dominates)
        - σ ≈ max → dispersed (uniform distribution)

    For 3-way (home/draw/away):
        - Max dispersion: σ([1, 0, 0]) ≈ 0.471
        - Min dispersion: σ([1/3, 1/3, 1/3]) = 0

    Args:
        probs: Probability distribution

    Returns:
        Standard deviation of probabilities

    Example:
        >>> odds_dispersion([0.8, 0.1, 0.1])
        0.3266...
        >>> odds_dispersion([1/3, 1/3, 1/3])
        0.0
    """
    probs_arr = np.array(probs, dtype=float)
    return float(np.std(probs_arr))
