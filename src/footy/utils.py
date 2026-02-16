"""Shared utility functions used across footy modules."""
from __future__ import annotations

import math
import numpy as np


def safe_num(v) -> float | None:
    """Convert a value to float, returning None for NaN/Inf/empty/invalid."""
    try:
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        result = float(v)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    except Exception:
        return None


def outcome_label(home_goals: int, away_goals: int) -> int:
    """Return match outcome: 0 = Home win, 1 = Draw, 2 = Away win."""
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def compute_metrics(
    P: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-12,
) -> dict:
    """
    Compute standard prediction metrics.

    Args:
        P: (n, 3) probability array [home, draw, away]
        y: (n,) integer labels 0/1/2
        eps: small constant for log stability

    Returns:
        dict with logloss, brier, accuracy
    """
    logloss = float(np.mean(-np.log(P[np.arange(len(y)), y] + eps)))
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
    accuracy = float(np.mean(np.argmax(P, axis=1) == y))
    return {"logloss": logloss, "brier": brier, "accuracy": accuracy}
