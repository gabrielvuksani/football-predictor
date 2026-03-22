"""WeibullExpert — Weibull count model with per-team shape parameter tracking.

Uses the Weibull count distribution (McShane et al. 2008) which generalizes
Poisson by allowing non-constant goal-scoring rates.  The shape parameter c
is estimated per team from the observed pattern of goal deviations.

Reference:
    McShane et al. (2008) "Count models based on Weibull interarrival times"
    Baker & McHale (2015) "Forecasting exact scores in football matches"
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _f, _is_finished, _norm3
from footy.models.math.weibull import build_weibull_count_matrix


class WeibullExpert(Expert):
    """Attack/defence EMA + per-team shape parameter -> Weibull count score matrix."""

    name = "weibull"

    ALPHA = 0.05          # EMA smoothing for attack/defence rates
    SHAPE_ALPHA = 0.05    # EMA smoothing for shape parameter tracking
    MAX_GOALS = 8
    AVG = 1.35            # league-average goals per team
    HOME_ATT = 1.15       # home advantage multiplier on attack
    HOME_DEF = 0.85       # home advantage multiplier on defence (lower = better)
    DEFAULT_C = 1.08      # initial shape param (slight late-match acceleration)

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)

        # Per-team rolling state
        attack: dict[str, float] = {}
        defense: dict[str, float] = {}
        shape_c: dict[str, float] = {}        # Weibull shape parameter per team
        game_count: dict[str, int] = {}
        # Running deviation tracker for shape estimation
        # Stores EMA of signed (actual - expected) to detect systematic over/under
        dev_tracker: dict[str, float] = {}

        # Output arrays
        probs = np.full((n, 3), 1.0 / 3)
        conf = np.zeros(n)
        out_lam_h = np.zeros(n)
        out_lam_a = np.zeros(n)
        out_c_h = np.zeros(n)
        out_c_a = np.zeros(n)
        out_p00 = np.zeros(n)
        out_btts = np.zeros(n)
        out_o25 = np.zeros(n)
        out_entropy = np.zeros(n)
        out_ml_h = np.zeros(n)
        out_ml_a = np.zeros(n)

        def _att(t: str) -> float:
            return attack.get(t, self.AVG)

        def _def(t: str) -> float:
            return defense.get(t, self.AVG * 0.9)

        def _c(t: str) -> float:
            return shape_c.get(t, self.DEFAULT_C)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            att_h, def_h = _att(h), _def(h)
            att_a, def_a = _att(a), _def(a)
            c_h, c_a = _c(h), _c(a)

            # Expected goals with home advantage
            lam_h = max(0.25, min(5.0, att_h * self.HOME_ATT * def_a / self.AVG))
            lam_a = max(0.25, min(5.0, att_a * def_h * self.HOME_DEF / self.AVG))

            out_lam_h[i] = lam_h
            out_lam_a[i] = lam_a
            out_c_h[i] = c_h
            out_c_a[i] = c_a

            # Build score matrix via Weibull count model
            mat = build_weibull_count_matrix(
                lam_h, lam_a, c_h, c_a, max_goals=self.MAX_GOALS,
            )

            mg = self.MAX_GOALS + 1

            # 1X2 probabilities
            p_home = float(np.tril(mat, -1).sum())
            p_draw = float(np.trace(mat))
            p_away = float(np.triu(mat, 1).sum())
            ph, pd_, pa = _norm3(p_home, p_draw, p_away)
            probs[i] = [ph, pd_, pa]

            # P(0-0)
            out_p00[i] = float(mat[0, 0])

            # P(BTTS) = 1 - P(home=0) - P(away=0) + P(0-0)
            p_h0 = float(mat[0, :].sum())
            p_a0 = float(mat[:, 0].sum())
            out_btts[i] = max(0.0, 1.0 - p_h0 - p_a0 + mat[0, 0])

            # P(over 2.5)
            under25 = sum(
                float(mat[hg, ag])
                for hg in range(mg)
                for ag in range(mg)
                if hg + ag <= 2
            )
            out_o25[i] = 1.0 - under25

            # Entropy of 1X2 distribution
            p3 = np.array([ph, pd_, pa])
            p3 = np.clip(p3, 1e-12, 1.0)
            out_entropy[i] = float(-(p3 * np.log(p3)).sum())

            # Most likely scoreline
            best = np.unravel_index(np.argmax(mat), mat.shape)
            out_ml_h[i] = best[0]
            out_ml_a[i] = best[1]

            # Confidence based on games played
            gc_h = game_count.get(h, 0)
            gc_a = game_count.get(a, 0)
            conf[i] = min(1.0, (gc_h + gc_a) / 20.0)

            # --- Update rolling state for finished matches only ---
            if not _is_finished(r):
                continue

            hg, ag = int(r.home_goals), int(r.away_goals)

            # Update attack/defence EMA
            if h in attack:
                attack[h] = (1 - self.ALPHA) * att_h + self.ALPHA * hg
                defense[h] = (1 - self.ALPHA) * def_h + self.ALPHA * ag
            else:
                attack[h] = float(hg) if hg > 0 else self.AVG
                defense[h] = float(ag) if ag > 0 else self.AVG * 0.9

            if a in attack:
                attack[a] = (1 - self.ALPHA) * att_a + self.ALPHA * ag
                defense[a] = (1 - self.ALPHA) * def_a + self.ALPHA * hg
            else:
                attack[a] = float(ag) if ag > 0 else self.AVG
                defense[a] = float(hg) if hg > 0 else self.AVG * 0.9

            # Update shape parameter tracking
            # c captures whether a team tends to score more than expected (c > 1,
            # increasing rate / late-match intensity) or less (c < 1, defensive
            # tightening).  We track the signed deviation and map it to c.
            for team, actual, expected in [(h, hg, lam_h), (a, ag, lam_a)]:
                signed_dev = actual - expected
                if team in dev_tracker:
                    dev_tracker[team] = (
                        (1 - self.SHAPE_ALPHA) * dev_tracker[team]
                        + self.SHAPE_ALPHA * signed_dev
                    )
                else:
                    dev_tracker[team] = signed_dev

                # Map accumulated deviation to shape parameter:
                # Positive deviation -> higher c (goals come in bursts / late)
                # Negative deviation -> lower c (defensive, fewer than expected)
                # Scale factor 0.15 keeps adjustments moderate
                raw_c = self.DEFAULT_C + dev_tracker[team] * 0.15
                shape_c[team] = max(0.5, min(2.0, raw_c))

            game_count[h] = game_count.get(h, 0) + 1
            game_count[a] = game_count.get(a, 0) + 1

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "weib_lambda_h": out_lam_h,
                "weib_lambda_a": out_lam_a,
                "weib_c_h": out_c_h,
                "weib_c_a": out_c_a,
                "weib_p00": out_p00,
                "weib_btts": out_btts,
                "weib_o25": out_o25,
                "weib_entropy": out_entropy,
                "weib_most_likely_h": out_ml_h,
                "weib_most_likely_a": out_ml_a,
            },
        )
