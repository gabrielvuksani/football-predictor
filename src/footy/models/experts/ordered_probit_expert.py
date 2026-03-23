"""OrderedProbitExpert — Ordinal outcome model for match results.

Football outcomes are naturally ORDERED: Away Win < Draw < Home Win.
Standard multinomial approaches (logistic regression, gradient boosting)
ignore this ordering. The ordered probit model captures it through a
latent variable framework.

Math:
    y* = β₀ + β₁·elo_diff + β₂·form_diff + β₃·market_diff + ε,  ε ~ N(0,1)
    P(Away win) = Φ(c₁ - y*)
    P(Draw) = Φ(c₂ - y*) - Φ(c₁ - y*)
    P(Home win) = 1 - Φ(c₂ - y*)
    where Φ is the standard normal CDF, c₁ < c₂ are cutpoints

Uses existing math: ordered_probit_1x2() from math/ensemble.py

References:
    McKelvey & Zavoina (1975) "A statistical model for the analysis of ordinal data"
    Goddard (2005) "Regression models for forecasting goals and match results"
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from footy.models.experts._base import Expert, ExpertResult, _f, _is_finished, _norm3, _pts


class OrderedProbitExpert(Expert):
    """Ordered probit model treating match outcomes as ordinal."""

    name = "ordered_probit"

    # Cutpoints estimated from typical European football:
    # ~45% home wins, ~27% draws, ~28% away wins
    # Φ⁻¹(0.28) ≈ -0.58, Φ⁻¹(0.55) ≈ 0.13
    C1_INIT = -0.58  # boundary between away win and draw
    C2_INIT = 0.13   # boundary between draw and home win

    # Coefficient learning rate
    LEARNING_RATE = 0.005
    WINDOW = 200  # calibration window

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)

        # Track Elo for strength input
        elo: dict[str, float] = {}
        form: dict[str, list[float]] = {}  # recent points
        ELO_DEFAULT = 1500.0
        ELO_K = 20.0

        # Cutpoint calibration via rolling window
        calibration_data: list[tuple[float, int]] = []  # (mu, outcome)
        c1 = self.C1_INIT
        c2 = self.C2_INIT

        # Output arrays
        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)
        op_latent = np.zeros(n)
        op_c1 = np.zeros(n)
        op_c2 = np.zeros(n)
        op_draw_width = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            # Compute latent variable inputs
            elo_h = elo.get(h, ELO_DEFAULT)
            elo_a = elo.get(a, ELO_DEFAULT)
            elo_diff = (elo_h - elo_a) / 400.0  # normalized

            # Form differential
            form_h = form.get(h, [])
            form_a = form.get(a, [])
            form_diff = 0.0
            if len(form_h) >= 3 and len(form_a) >= 3:
                form_diff = (np.mean(form_h[-5:]) - np.mean(form_a[-5:])) / 3.0

            # Market signal (if available)
            mkt_h = _f(getattr(r, "b365h", None))
            mkt_a = _f(getattr(r, "b365a", None))
            mkt_diff = 0.0
            if mkt_h > 1 and mkt_a > 1:
                mkt_diff = (1.0 / mkt_h - 1.0 / mkt_a)  # implied prob difference

            # Latent variable: linear combination of signals
            mu = 0.15 + 0.50 * elo_diff + 0.25 * form_diff + 0.30 * mkt_diff
            op_latent[i] = mu
            op_c1[i] = c1
            op_c2[i] = c2
            op_draw_width[i] = c2 - c1

            # Ordered probit probabilities
            p_away = float(norm.cdf(c1 - mu))
            p_draw = float(norm.cdf(c2 - mu) - norm.cdf(c1 - mu))
            p_home = float(1.0 - norm.cdf(c2 - mu))

            ph, pd_, pa = _norm3(
                max(0.03, p_home),
                max(0.05, p_draw),
                max(0.03, p_away)
            )
            probs[i] = [ph, pd_, pa]

            # Confidence: higher when latent is far from cutpoints
            dist_to_c1 = abs(mu - c1)
            dist_to_c2 = abs(mu - c2)
            min_dist = min(dist_to_c1, dist_to_c2)
            conf[i] = min(0.7, 0.2 + min_dist * 0.3)

            # Update state
            if not _is_finished(r):
                continue
            hg, ag = int(r.home_goals), int(r.away_goals)
            outcome = 0 if hg > ag else (1 if hg == ag else 2)

            # Update Elo
            e_h = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400.0))
            s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            gd = abs(hg - ag)
            k = ELO_K * (1 + 0.5 * math.log1p(gd))
            delta = k * (s_h - e_h)
            elo[h] = elo_h + delta
            elo[a] = elo_a - delta

            # Update form
            form.setdefault(h, []).append(float(_pts(hg, ag)))
            form.setdefault(a, []).append(float(_pts(ag, hg)))
            if len(form[h]) > 20:
                form[h] = form[h][-20:]
            if len(form[a]) > 20:
                form[a] = form[a][-20:]

            # Calibrate cutpoints from recent data
            calibration_data.append((mu, outcome))
            if len(calibration_data) > self.WINDOW:
                calibration_data = calibration_data[-self.WINDOW:]

            # Re-estimate cutpoints every 100 matches via grid search MLE
            if len(calibration_data) >= 100 and len(calibration_data) % 100 == 0:
                mus = np.array([d[0] for d in calibration_data])
                outs = np.array([d[1] for d in calibration_data])

                # Grid search over c1, c2 in [-1.0, 1.0] with step 0.05
                # Minimize classification error (equivalent to MLE approximation)
                best_c1, best_c2 = c1, c2
                best_error = float('inf')
                grid = np.arange(-1.0, 1.05, 0.05)

                for c1_cand in grid:
                    for c2_cand in grid:
                        if c2_cand <= c1_cand + 0.05:
                            continue  # enforce c1 < c2 with minimum gap
                        # Compute predicted outcomes for all samples
                        pred = np.where(
                            mus > c2_cand, 0,  # home win: mu > c2
                            np.where(mus < c1_cand, 2, 1)  # away win: mu < c1, else draw
                        )
                        err = float(np.sum(pred != outs))
                        if err < best_error:
                            best_error = err
                            best_c1 = c1_cand
                            best_c2 = c2_cand

                # Smooth update to avoid sudden jumps
                c1 = c1 * 0.7 + best_c1 * 0.3
                c2 = c2 * 0.7 + best_c2 * 0.3

                # Ensure c1 < c2
                if c1 >= c2:
                    c1 = c2 - 0.3

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "op_latent": op_latent,
                "op_c1": op_c1,
                "op_c2": op_c2,
                "op_draw_width": op_draw_width,
            },
        )
