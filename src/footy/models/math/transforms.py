"""Nonlinear transforms and feature engineering for football prediction.

Implements:
- Hyperbolic tangent transform for bounded Elo strength
- Sign-preserving log transform for rate features
- Adaptive exponential weighted moving average (EWMA)
- Schedule difficulty rating based on upcoming opponents
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# NONLINEAR STRENGTH TRANSFORMS
# ═══════════════════════════════════════════════════════════════════
def tanh_transform(diff: float, scale: float = 400.0) -> float:
    """Bounded hyperbolic tangent transform for strength differences.

    Maps unbounded strength differences to [-1, 1] using tanh.
    This is the standard transform for Elo-style rating differences.

    f(x) = tanh(x / scale)

    With scale ≈ 400 (from Elo), this gives:
        - f(400) ≈ 0.76 (moderate advantage)
        - f(0) = 0 (equal strength)
        - f(-400) ≈ -0.76 (moderate disadvantage)

    The tanh transformation is common in sports because:
    1. Non-linear: small differences matter less at the extremes
    2. Bounded: prevents extreme weights
    3. Continuous: smooth gradients

    Args:
        diff: Strength difference (unbounded, e.g., Elo difference)
        scale: Scaling factor (default 400, typical for Elo)

    Returns:
        Bounded value in [-1, 1]

    Example:
        >>> tanh_transform(400.0)
        0.761...
        >>> tanh_transform(0.0)
        0.0
        >>> tanh_transform(-400.0)
        -0.761...
    """
    diff = float(diff)
    scale = float(scale)
    # Clip to prevent overflow in tanh
    clipped = np.clip(diff / scale, -10, 10)
    return float(np.tanh(clipped))


def log_transform(x: float, offset: float = 1.0) -> float:
    """Sign-preserving log transform for rate features.

    Applies natural log while preserving sign:
        - If x > 0: sign(x) · ln(|x| + offset)
        - If x = 0: return 0.0
        - If x < 0: sign(x) · ln(|x| + offset)

    Useful for skewed distributions like goals, shots, etc.
    The offset prevents -inf when x → 0.

    Example: Goals scored
        - 0 goals → 0.0
        - 1 goal → ln(2) ≈ 0.693
        - 2 goals → ln(3) ≈ 1.099
        - 10 goals → ln(11) ≈ 2.398

    Args:
        x: Input value (typically rate/count, unbounded)
        offset: Offset to apply before log (default 1.0)

    Returns:
        Log-transformed value, sign-preserving

    Example:
        >>> log_transform(0.0)
        0.0
        >>> log_transform(100.0)
        4.605...
        >>> log_transform(-100.0)
        -4.605...
    """
    x = float(x)
    offset = float(offset)

    if x == 0.0:
        return 0.0

    sign = 1.0 if x > 0 else -1.0
    return float(sign * np.log(abs(x) + offset))


# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE EXPONENTIAL WEIGHTED MOVING AVERAGE (EWMA)
# ═══════════════════════════════════════════════════════════════════
def adaptive_ewma(
    values: Sequence[float],
    span: int = 5,
) -> float:
    """Adaptive exponentially weighted moving average (EWMA).

    Computes: EWMA = Σ w_i · x_i, where w_i ∝ exp(-|i - n| / span)

    Weights are computed with exponential decay from the most recent value:
        w_n (most recent) > w_{n-1} > ... > w_1 (oldest)

    With span=5:
        - Effective window ≈ 5 values with geometric decay
        - Recent values are ~2.7× more important than older values

    Used for:
    - Form tracking (more recent = more relevant)
    - Volatility estimation
    - Momentum indicators

    Args:
        values: Sequence of values (typically [oldest, ..., most recent])
        span: Decay half-life (effective number of periods)

    Returns:
        Weighted average emphasizing recent values

    Example:
        >>> adaptive_ewma([1, 2, 3, 4, 5])
        3.8...  # Weighted towards 5
        >>> adaptive_ewma([5])
        5.0  # Single value → return as-is
        >>> adaptive_ewma([])
        0.0  # Empty → 0
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    values = [float(v) for v in values]
    n = len(values)
    span = float(max(1, span))

    # Weights: w_i = exp(-|i - (n-1)| / span)
    # Most recent (i = n-1) has w = exp(0) = 1.0
    # Oldest (i = 0) has w = exp(-(n-1) / span)
    weights = np.array([
        np.exp(-abs(i - (n - 1)) / span) for i in range(n)
    ])

    # Normalize
    weights /= weights.sum()

    return float(np.dot(values, weights))


# ═══════════════════════════════════════════════════════════════════
# SCHEDULE DIFFICULTY RATING
# ═══════════════════════════════════════════════════════════════════
def schedule_difficulty(
    upcoming_opponents_elo: Sequence[float],
    league_avg_elo: float = 1500.0,
    window: int = 5,
) -> float:
    """Compute schedule difficulty rating from upcoming opponent Elos.

    Measures the "difficulty" of the next N opponents using:
    1. Recency weighting: more recent opponents matter more
    2. Opponent strength: tougher opponents → higher rating
    3. Competition boost: consecutive tough opponents amplify difficulty

    The difficulty score is typically in Elo-like units (e.g., 1400-1700 for
    top leagues), representing the "equivalent opponent" strength faced.

    Args:
        upcoming_opponents_elo: Elos of next N opponents (ascending by time)
        league_avg_elo: League-wide average Elo (default 1500)
        window: Number of upcoming matches to consider (uses LAST N elements)

    Returns:
        Difficulty rating in Elo-like units

    Example:
        >>> sd = schedule_difficulty([1400, 1500, 1600])
        >>> 1400 < sd < 1700  # Should be in range
        True
    """
    if not upcoming_opponents_elo:
        return float(league_avg_elo)

    # Take the LAST window elements (most recent opponents)
    opponents = [float(e) for e in upcoming_opponents_elo][-window:] if window > 0 else []
    if not opponents:
        return float(league_avg_elo)

    n = len(opponents)

    # Recency weighting: most recent (last) opponent gets highest weight
    weights = np.array([
        np.exp(i / float(max(1, n - 1))) if n > 1 else 1.0
        for i in range(n)
    ])
    weights /= weights.sum()

    # Base weighted opponent strength
    weighted_strength = float(np.dot(opponents, weights))

    # Maximum opponent bonus: add strength of toughest opponent
    # This captures "series of tough opponents" effect
    max_opponent = max(opponents)
    max_bonus = (max_opponent - league_avg_elo) * 0.25  # 25% of gap (increased for effect)

    difficulty = weighted_strength + max_bonus

    return float(difficulty)
