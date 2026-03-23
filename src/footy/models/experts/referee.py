"""RefereeExpert — referee assignment bias and tendencies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _f


class RefereeExpert(Expert):
    """Model referee tendencies: cards, penalties, home bias.

    Referees vary significantly in their rates of yellow/red cards,
    penalties awarded, and implicit home/away bias. This expert
    captures those tendencies as features and adjusts probabilities
    based on historical referee patterns.
    """

    name = "referee"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        # Feature arrays
        yellow_rate = np.zeros(n)
        red_rate = np.zeros(n)
        penalty_rate = np.zeros(n)
        home_bias = np.zeros(n)
        ref_matches = np.zeros(n)
        has_referee = np.zeros(n)
        strict_ref = np.zeros(n)
        lenient_ref = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            ref_name = getattr(r, "referee_name", None)
            if not ref_name or ref_name == "":
                continue

            has_referee[i] = 1.0
            ym = _f(getattr(r, "ref_yellow_per_match", None))
            rm = _f(getattr(r, "ref_red_per_match", None))
            pm = _f(getattr(r, "ref_penalty_per_match", None))
            hb = _f(getattr(r, "ref_home_bias_ratio", None))
            nm = _f(getattr(r, "ref_historical_matches", None))

            yellow_rate[i] = ym
            red_rate[i] = rm
            penalty_rate[i] = pm
            home_bias[i] = hb
            ref_matches[i] = nm

            # Classify referee strictness
            strict_ref[i] = 1.0 if ym > 4.5 or pm > 0.35 else 0.0
            lenient_ref[i] = 1.0 if ym < 3.0 and pm < 0.15 else 0.0

            # Home bias adjustments
            home_boost = 0.0
            draw_adj = 0.0

            if hb > 1.1:  # referee slightly favors home
                home_boost = min(0.04, (hb - 1.0) * 0.03)
            elif hb < 0.9 and hb > 0:  # referee slightly favors away
                home_boost = max(-0.04, (hb - 1.0) * 0.03)

            # Strict referees tend to produce more goals (penalties, red cards leading to
            # tactical changes), slightly favoring decisive outcomes over draws
            if strict_ref[i]:
                draw_adj = -0.02
            elif lenient_ref[i]:
                draw_adj = 0.01  # fewer incidents → slightly more draws

            p_h = 0.36 + home_boost
            p_d = 0.28 + draw_adj
            p_a = 0.36 - home_boost * 0.6

            s = p_h + p_d + p_a
            if s > 0:
                probs[i] = [p_h / s, p_d / s, p_a / s]

            # Confidence based on data availability
            conf[i] = min(0.85, nm * 0.005 + has_referee[i] * 0.05)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "ref_yellow_rate": yellow_rate,
                "ref_red_rate": red_rate,
                "ref_penalty_rate": penalty_rate,
                "ref_home_bias": home_bias,
                "ref_matches": ref_matches,
                "ref_has_data": has_referee,
                "ref_strict": strict_ref,
                "ref_lenient": lenient_ref,
            },
        )
