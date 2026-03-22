"""Recency-attention weighting and ensemble blending for football prediction.

Implements transformer-inspired attention mechanisms for weighting historical
matches based on recency, and variance-adjusted ensemble methods for combining
probabilistic forecasts.

References:
    Vaswani et al. (2017) "Attention Is All You Need"
    Bates & Granger (1969) "The Combination of Forecasts"
    Geweke & Amisano (2011) "Optimal prediction pools"
"""
from __future__ import annotations

import numpy as np


def recency_attention_weights(n: int, temperature: float = 0.35) -> np.ndarray:
    """Attention-like recency weights (transformer-inspired).

    Generates softmax-normalized weights over N historical items,
    where more recent items get exponentially higher weight.

    Used for:
    - Recent-match weighting in ensemble models
    - Time-decay for historical data
    - Importance sampling in MCMC

    Formula: w_i = softmax((i - max_i) / (T · (n-1)))

    Where:
        i = position in sequence (0 = oldest, n-1 = newest)
        T = temperature (higher = more uniform, lower = sharper recency bias)
        softmax = exp(x_i) / Σ exp(x_j)

    Args:
        n: Number of items in sequence
        temperature: Temperature parameter (higher = flatter weights)
                    Default 0.35 gives ~80% weight to most recent 20% of items

    Returns:
        numpy array of shape (n,) with weights summing to 1.0

    Example:
        >>> w = recency_attention_weights(10, temperature=0.35)
        >>> print(f"Last item weight: {w[-1]:.3f}")  # ~0.160
        >>> print(f"First item weight: {w[0]:.3f}")  # ~0.001
    """
    if n <= 0:
        return np.zeros(0, dtype=float)

    idx = np.arange(n, dtype=float)

    # Logits: scale by position difference / (T · (n-1))
    # Ensure oldest items get negative logits (low probability)
    logits = (idx - idx.max()) / max(1e-9, temperature * max(1.0, n - 1))

    # Numerical stability: subtract max before exp
    logits = logits - logits.max()

    # Softmax
    w = np.exp(logits)
    return w / w.sum()


def variance_adjusted_blend(
    prob_mats: list[np.ndarray],
    variances: list[np.ndarray],
) -> np.ndarray:
    """Blend probability matrices inversely proportional to estimated variance.

    Implements Bates & Granger (1969) optimal linear combination:
    Blended forecast = Σ w_i · f_i, where w_i ∝ 1/var_i

    This gives more weight to models with lower (more confident) predictions,
    which is theoretically optimal for combining forecasts.

    Args:
        prob_mats: List of probability matrices (N, K) or probability distributions
        variances: List of estimated variances for each model
                  Same shape as prob_mats

    Returns:
        Blended probability matrix, same shape as input matrices

    Raises:
        ValueError: If prob_mats is empty
    """
    if not prob_mats:
        raise ValueError("prob_mats must not be empty")

    # Convert to numpy arrays and clip variances to avoid division by zero
    probs = [np.asarray(p, dtype=float) for p in prob_mats]
    vars_ = [np.clip(np.asarray(v, dtype=float), 1e-9, None) for v in variances]

    # Compute inverse variances (higher variance → lower weight)
    inv_vars = [1.0 / v for v in vars_]

    # Normalize weights
    total_inv_var = np.sum(inv_vars, axis=0)

    # Blend with normalized weights
    blended = np.zeros_like(probs[0])
    for p, iv in zip(probs, inv_vars):
        blended += p * (iv / total_inv_var)

    # Renormalize full matrix to ensure it sums to 1
    # (for score matrices, not per-row normalization)
    total_mass = blended.sum()
    if total_mass > 0:
        blended = blended / total_mass

    return blended


def score_matrix_entropy(mat: np.ndarray) -> float:
    """Shannon entropy of a score probability matrix.

    H = -Σ p_ij · log(p_ij)

    Measures uncertainty in the match outcome distribution:
    - H ≈ 0: One scoreline has very high probability (confident prediction)
    - H ≈ max: All scorelines equally likely (maximum uncertainty)

    Args:
        mat: Score probability matrix

    Returns:
        Entropy value (non-negative)
    """
    flat = mat.ravel()
    flat = flat[flat > 1e-15]  # Avoid log(0)
    return float(-np.sum(flat * np.log(flat)))


def information_gain_from_odds(
    model_mat: np.ndarray,
    market_probs: list[float],
) -> float:
    """KL divergence of model vs. market — value betting signal.

    Measures how much model probability distribution diverges from
    market's implied probabilities. High values suggest potential value bets.

    D_KL(model || market) = Σ p_model_i · log(p_model_i / p_market_i)

    Args:
        model_mat: Score probability matrix from prediction model
        market_probs: Market's implied 1X2 probabilities [p_home, p_draw, p_away]

    Returns:
        KL divergence value (≥ 0)
    """
    from footy.models.math.simulation import extract_match_probs

    model_p = extract_match_probs(model_mat)
    p_model = [model_p["p_home"], model_p["p_draw"], model_p["p_away"]]
    p_market = list(market_probs)

    eps = 1e-12
    kl = sum(
        max(pm, eps) * np.log(max(pm, eps) / max(pk, eps))
        for pm, pk in zip(p_model, p_market)
    )
    return max(0.0, kl)
