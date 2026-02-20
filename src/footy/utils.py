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
        dict with logloss, brier, accuracy, rps
    """
    from footy.models.advanced_math import ranked_probability_score

    logloss = float(np.mean(-np.log(P[np.arange(len(y)), y] + eps)))
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1) / 3))
    accuracy = float(np.mean(np.argmax(P, axis=1) == y))
    # Ranked Probability Score — proper scoring rule for ordered outcomes
    rps = float(np.mean([
        ranked_probability_score(P[i].tolist(), int(y[i]))
        for i in range(len(y))
    ]))
    return {"logloss": logloss, "brier": brier, "accuracy": accuracy, "rps": rps}


def score_prediction(probs: list[float] | tuple[float, ...], outcome: int) -> dict:
    """Score a single 1×2 prediction against the actual outcome.

    Args:
        probs: [p_home, p_draw, p_away] probabilities
        outcome: 0 = Home win, 1 = Draw, 2 = Away win

    Returns:
        dict with logloss, brier, rps, predicted (int), correct (bool)
    """
    from footy.models.advanced_math import ranked_probability_score

    outcome_prob = probs[outcome]
    logloss = -math.log(max(outcome_prob, 1e-15))
    brier = sum((p - (1.0 if i == outcome else 0.0)) ** 2
                for i, p in enumerate(probs)) / 3
    rps = ranked_probability_score(list(probs), outcome)
    predicted = max(range(3), key=lambda i: probs[i])
    return {
        "logloss": logloss,
        "brier": brier,
        "rps": rps,
        "predicted": predicted,
        "correct": predicted == outcome,
    }
