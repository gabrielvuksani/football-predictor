"""MarketValueExpert — squad market value differentials from Transfermarkt."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult


def _sf(value: int | float | str | None) -> float:
    try:
        return float(value) if value is not None else 0.0
    except Exception:
        return 0.0


class MarketValueExpert(Expert):
    """Use squad market values as a team-quality signal.

    Market value correlates strongly with squad quality, depth, and
    expected performance. This expert captures value differentials
    and squad composition features.
    """

    name = "market_value"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        value_ratio = np.zeros(n)
        value_diff_log = np.zeros(n)
        avg_value_ratio = np.zeros(n)
        depth_diff = np.zeros(n)
        age_diff = np.zeros(n)
        has_data = np.zeros(n)
        home_stronger = np.zeros(n)
        value_dominance = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            hv = _sf(getattr(r, "mv_home_squad_value", None))
            av = _sf(getattr(r, "mv_away_squad_value", None))
            hav = _sf(getattr(r, "mv_home_avg_value", None))
            aav = _sf(getattr(r, "mv_away_avg_value", None))
            hd = _sf(getattr(r, "mv_home_depth", None))
            ad = _sf(getattr(r, "mv_away_depth", None))
            ha = _sf(getattr(r, "mv_home_avg_age", None))
            aa = _sf(getattr(r, "mv_away_avg_age", None))

            if hv > 0 and av > 0:
                has_data[i] = 1.0
                value_ratio[i] = hv / av
                value_diff_log[i] = np.log(hv / av)
                home_stronger[i] = 1.0 if hv > av else 0.0
                value_dominance[i] = min(3.0, max(-3.0, np.log(hv / av)))

                if hav > 0 and aav > 0:
                    avg_value_ratio[i] = hav / aav

                depth_diff[i] = hd - ad
                age_diff[i] = ha - aa

                # Probability adjustment based on value ratio
                log_ratio = np.log(hv / av)
                # Sigmoid-based probability shift
                shift = np.tanh(log_ratio * 0.3) * 0.15

                p_h = 0.36 + shift + 0.04  # home advantage
                p_d = 0.28 - abs(shift) * 0.3  # less draw when mismatch
                p_a = 0.36 - shift

                s = p_h + p_d + p_a
                if s > 0:
                    probs[i] = [p_h / s, p_d / s, p_a / s]

                conf[i] = min(0.85, 0.2 + abs(log_ratio) * 0.1)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "mv_value_ratio": value_ratio,
                "mv_value_diff_log": value_diff_log,
                "mv_avg_value_ratio": avg_value_ratio,
                "mv_depth_diff": depth_diff,
                "mv_age_diff": age_diff,
                "mv_has_data": has_data,
                "mv_home_stronger": home_stronger,
                "mv_value_dominance": value_dominance,
            },
        )
