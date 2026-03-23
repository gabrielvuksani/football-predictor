"""Base classes and shared helpers for expert modules."""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def _f(x) -> float:
    """Safe float cast."""
    try:
        if x is None:
            return 0.0
        v = float(x)
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _raw(raw_json) -> dict:
    if raw_json is None:
        return {}
    if isinstance(raw_json, dict):
        return raw_json
    try:
        return json.loads(raw_json)
    except Exception:
        return {}


def _pts(gf: int, ga: int) -> int:
    return 3 if gf > ga else (1 if gf == ga else 0)


def _is_finished(r) -> bool:
    """Check if a match row has real (non-NaN) goal data.

    Used to prevent batch contamination: upcoming dummy rows have NaN goals
    and should NOT update rolling state in experts.
    """
    hg = getattr(r, "home_goals", None)
    ag = getattr(r, "away_goals", None)
    if hg is None or ag is None:
        return False
    try:
        return math.isfinite(float(hg)) and math.isfinite(float(ag))
    except (ValueError, TypeError):
        return False


def _label(hg: int, ag: int) -> int:
    if hg > ag:
        return 0
    if hg == ag:
        return 1
    return 2


def _entropy3(p) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _norm3(a, b, c) -> tuple[float, float, float]:
    s = a + b + c
    if s <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (a / s, b / s, c / s)


def _shin_implied(odds: list[float], max_iter: int = 1000, tol: float = 1e-12) -> list[float]:
    """Shin (1991, 1993) method to extract true probabilities from bookmaker odds.

    Accounts for the favourite-longshot bias by modelling the bookmaker's
    response to a fraction *z* of insider bettors.  Produces more accurate
    implied probabilities than naive normalisation.

    Algorithm from the reference implementation (mberk/shin on GitHub),
    using fixed-point iteration to find the insider fraction z.

    Reference:
        Shin (1993) "Measuring the Incidence of Insider Trading"
        Jullien & Salanié (1994) — fixed-point iteration for z

    Args:
        odds: List of decimal odds (e.g. [2.1, 3.3, 3.5]).
        max_iter: Maximum iterations.
        tol: Convergence tolerance for z.

    Returns:
        List of true probabilities (same length as odds), summing to ~1.0.
    """
    n = len(odds)
    inv_odds = [1.0 / o for o in odds]
    S = sum(inv_odds)  # booksum (> 1 due to overround)

    # If overround is tiny (< 0.5%), basic normalisation is sufficient
    if S < 1.005:
        return [p / S for p in inv_odds]

    # Fixed-point iteration to find insider fraction z
    z = 0.0
    for _ in range(max_iter):
        z0 = z
        z = (
            sum(
                math.sqrt(z ** 2 + 4.0 * (1.0 - z) * io ** 2 / S)
                for io in inv_odds
            )
            - 2.0
        ) / (n - 2)
        if abs(z - z0) < tol:
            break

    # Compute true probabilities with converged z
    probs = []
    for io in inv_odds:
        disc = z ** 2 + 4.0 * (1.0 - z) * io ** 2 / S
        p = (math.sqrt(disc) - z) / (2.0 * (1.0 - z)) if z < 1.0 else io / S
        probs.append(max(0.0, p))

    # Safety normalise (should already sum to ~1.0)
    total = sum(probs)
    if total > 0 and abs(total - 1.0) > 1e-6:
        probs = [p / total for p in probs]

    return probs


def _power_implied(odds: list[float]) -> list[float]:
    """Remove overround via the power method — outperforms Shin.

    Finds k > 1 such that sum(raw_i^k) = 1, where raw_i = 1/odds_i.
    Since all raw_i < 1, raising to k > 1 shrinks longshots more than
    favourites, naturally correcting the favourite-longshot bias.
    Never produces out-of-range probabilities (raw_i in (0,1) => raw_i^k in (0,1)).

    Reference:
        Keith Cheung (2015) "Overround Removal Methods"
        — power method universally outperforms multiplicative,
          outperforms or matches Shin.
    """
    from scipy.optimize import brentq

    raw = 1.0 / np.array(odds, dtype=float)
    total = raw.sum()
    if total <= 1.0:
        return raw.tolist()  # no overround to remove

    # At k=1: sum(raw^1) = booksum > 1 (positive objective)
    # As k grows: raw^k -> 0 for all raw < 1 (negative objective)
    # So the root lies in (1, upper) for any reasonable overround.
    def objective(k):
        return np.sum(raw ** k) - 1.0

    try:
        k = brentq(objective, 1.0, 100.0, xtol=1e-8)
        probs = raw ** k
        return probs.tolist()
    except (ValueError, RuntimeError):
        # Fallback to simple normalisation if brentq fails
        return (raw / total).tolist()


def _implied(h, d, a):
    """Convert decimal odds to implied probabilities.

    Uses the power method (superior overround removal) as default,
    falls back to Shin, then to basic normalisation.
    Returns (p_home, p_draw, p_away, overround).
    """
    h, d, a = _f(h), _f(d), _f(a)
    if h <= 1 or d <= 1 or a <= 1:
        return (0.0, 0.0, 0.0, 0.0)
    overround = (1.0 / h + 1.0 / d + 1.0 / a) - 1.0
    try:
        probs = _power_implied([h, d, a])
        return (probs[0], probs[1], probs[2], overround)
    except Exception:
        pass
    try:
        probs = _shin_implied([h, d, a])
        return (probs[0], probs[1], probs[2], overround)
    except Exception:
        # Fallback to basic normalisation
        ih, id_, ia = 1 / h, 1 / d, 1 / a
        s = ih + id_ + ia
        return (ih / s, id_ / s, ia / s, overround)


# ---------------------------------------------------------------------------
# Expert result container
# ---------------------------------------------------------------------------
@dataclass
class ExpertResult:
    """Output from a single expert for n matches."""
    probs: np.ndarray          # (n, 3) analytical P(H, D, A)
    confidence: np.ndarray     # (n,) ∈ [0, 1]
    features: dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract expert
# ---------------------------------------------------------------------------
class Expert(ABC):
    name: str

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> ExpertResult:
        """Compute expert features over matches sorted by utc_date ASC.

        ``df`` has columns: utc_date, home_team, away_team, home_goals,
        away_goals (for finished) or NaN (for upcoming dummies).
        Plus optional extras columns (hs, hst, ...).

        Important: Use ``_is_finished(row)`` to check if a row has real goals
        before updating rolling state. Upcoming rows (NaN goals) should get
        features computed from current state but NOT update that state.
        """
        ...
