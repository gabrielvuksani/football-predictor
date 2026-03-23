"""xPtsExpert — expected points analysis and regression signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult


def _sf(value: int | float | str | None) -> float:
    try:
        return float(value) if value is not None else 0.0
    except Exception:
        return 0.0


class XPtsExpert(Expert):
    """Expected points (xPts) analysis.

    Computes the gap between actual accumulated points and expected points
    (derived from xG-based win probabilities). Large positive gaps indicate
    over-performance likely to regress; large negative gaps indicate
    under-performance likely to recover.
    """

    name = "xpts"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        home_luck = np.zeros(n)   # actual pts - xPts
        away_luck = np.zeros(n)
        home_xpts_ratio = np.zeros(n)
        away_xpts_ratio = np.zeros(n)
        luck_diff = np.zeros(n)
        regression_signal = np.zeros(n)
        has_xpts = np.zeros(n)
        overperformance = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h_pts = _sf(getattr(r, "home_actual_pts", None))
            h_xpts = _sf(getattr(r, "home_xpts", None))
            a_pts = _sf(getattr(r, "away_actual_pts", None))
            a_xpts = _sf(getattr(r, "away_xpts", None))
            h_matches = max(1, _sf(getattr(r, "home_matches_played", None)))
            a_matches = max(1, _sf(getattr(r, "away_matches_played", None)))

            if h_xpts > 0 and a_xpts > 0:
                has_xpts[i] = 1.0

                # Luck index: actual - expected (positive = over-performing)
                home_luck[i] = (h_pts - h_xpts) / h_matches
                away_luck[i] = (a_pts - a_xpts) / a_matches

                # xPts ratio: actual/expected
                home_xpts_ratio[i] = h_pts / max(0.1, h_xpts)
                away_xpts_ratio[i] = a_pts / max(0.1, a_xpts)

                luck_diff[i] = home_luck[i] - away_luck[i]

                # Regression signal: over-performers regress, under-performers recover
                # If home is over-performing, their "true strength" is lower → less likely to win
                regression_strength = np.clip(luck_diff[i], -1.5, 1.5)
                regression_signal[i] = -regression_strength * 0.3  # invert: luck hurts future probs

                overperformance[i] = 1.0 if abs(luck_diff[i]) > 0.3 else 0.0

                # Adjust probabilities
                adj = regression_signal[i] * 0.08
                p_h = 0.36 + adj
                p_d = 0.28
                p_a = 0.36 - adj

                s = p_h + p_d + p_a
                if s > 0:
                    probs[i] = [p_h / s, p_d / s, p_a / s]

                conf[i] = min(0.85, 0.1 + abs(luck_diff[i]) * 0.15)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "xpts_home_luck": home_luck,
                "xpts_away_luck": away_luck,
                "xpts_home_ratio": home_xpts_ratio,
                "xpts_away_ratio": away_xpts_ratio,
                "xpts_luck_diff": luck_diff,
                "xpts_regression_signal": regression_signal,
                "xpts_has_data": has_xpts,
                "xpts_overperformance": overperformance,
            },
        )
