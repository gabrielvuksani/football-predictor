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


def _implied(h, d, a):
    h, d, a = _f(h), _f(d), _f(a)
    if h <= 1 or d <= 1 or a <= 1:
        return (0.0, 0.0, 0.0, 0.0)
    ih, id_, ia = 1 / h, 1 / d, 1 / a
    s = ih + id_ + ia
    return (ih / s, id_ / s, ia / s, s - 1.0)


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
        away_goals (for finished) or 0 placeholders (for upcoming dummies).
        Plus optional extras columns (hs, hst, ...).
        """
        ...
