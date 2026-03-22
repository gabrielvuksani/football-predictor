"""NegBinExpert — Negative Binomial goal model for overdispersed counts.

The Negative Binomial distribution generalizes Poisson by allowing the
variance to exceed the mean (overdispersion). This is common in football
where goal variance is typically 10-20% higher than Poisson predicts,
especially in high-scoring leagues (Bundesliga, Eredivisie).

Math:
    P(X=k) = C(k+r-1, k) × p^r × (1-p)^k
    Mean = r(1-p)/p,  Variance = r(1-p)/p² = Mean + Mean²/r

    When r → ∞, NegBin → Poisson (no overdispersion).
    Smaller r = more overdispersion.

References:
    Karlis & Ntzoufras (2009) "Bayesian modelling of football outcomes"
    Cameron & Trivedi (2013) "Regression Analysis of Count Data"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3
from footy.models.math.distributions import build_negative_binomial_matrix
from footy.models.math.simulation import extract_match_probs


class NegBinExpert(Expert):
    """Negative Binomial goal model with per-league overdispersion tracking."""

    name = "negbin"

    ALPHA = 0.08       # EMA smoothing for attack/defense
    AVG = 1.35         # league average goals per team
    HOME_MULT = 1.05   # home advantage multiplier
    MAX_GOALS = 8

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        attack: dict[str, float] = {}
        defense: dict[str, float] = {}
        game_count: dict[str, int] = {}

        # Per-league overdispersion tracking
        # Track variance/mean ratio of goals to estimate r parameter
        league_goals: dict[str, list[int]] = {}  # comp -> list of goals scored

        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)
        nb_lam_h = np.zeros(n)
        nb_lam_a = np.zeros(n)
        nb_r_h = np.zeros(n)
        nb_r_a = np.zeros(n)
        nb_overdispersion = np.zeros(n)
        nb_btts = np.zeros(n)
        nb_o25 = np.zeros(n)
        nb_cs00 = np.zeros(n)

        def _att(t: str) -> float:
            return attack.get(t, self.AVG)

        def _def(t: str) -> float:
            return defense.get(t, self.AVG * 0.9)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            comp = getattr(r, "competition", "default")

            ah, dh = _att(h), _def(h)
            aa, da = _att(a), _def(a)

            # Expected goals
            lam_h = max(0.3, min(4.5, ah * da / self.AVG * self.HOME_MULT))
            lam_a = max(0.3, min(4.5, aa * dh / self.AVG))
            nb_lam_h[i] = lam_h
            nb_lam_a[i] = lam_a

            # Estimate r from league-level overdispersion
            # r = mean² / (variance - mean), where variance > mean for overdispersion
            lg = league_goals.get(comp, [])
            if len(lg) >= 50:
                arr = np.array(lg[-500:])  # use last 500 goals
                mean_g = arr.mean()
                var_g = arr.var()
                if var_g > mean_g + 0.01:
                    r_est = max(2.0, mean_g ** 2 / (var_g - mean_g))
                else:
                    r_est = 50.0  # effectively Poisson (no overdispersion)
            else:
                r_est = 8.0  # default: moderate overdispersion

            nb_r_h[i] = r_est
            nb_r_a[i] = r_est
            nb_overdispersion[i] = 1.0 / r_est  # higher = more overdispersed

            # Build NegBin score matrix
            mat = build_negative_binomial_matrix(
                lam_h, lam_a, alpha_h=r_est, alpha_a=r_est, max_goals=self.MAX_GOALS
            )
            feats = extract_match_probs(mat)
            ph, pd_, pa = feats["p_home"], feats["p_draw"], feats["p_away"]
            ph, pd_, pa = _norm3(ph, pd_, pa)
            probs[i] = [ph, pd_, pa]
            nb_btts[i] = feats.get("p_btts", 0.5)
            nb_o25[i] = feats.get("p_o25", 0.5)
            nb_cs00[i] = float(mat[0, 0]) if mat.shape[0] > 0 else 0.0

            # Confidence
            gc_h = game_count.get(h, 0)
            gc_a = game_count.get(a, 0)
            conf[i] = min(1.0, (gc_h + gc_a) / 20.0)

            # Update state
            if not _is_finished(r):
                continue
            hg, ag = int(r.home_goals), int(r.away_goals)

            # EMA update
            attack[h] = (1 - self.ALPHA) * ah + self.ALPHA * hg if h in attack else (float(hg) if hg > 0 else self.AVG)
            defense[h] = (1 - self.ALPHA) * dh + self.ALPHA * ag if h in defense else (float(ag) if ag > 0 else self.AVG * 0.9)
            attack[a] = (1 - self.ALPHA) * aa + self.ALPHA * ag if a in attack else (float(ag) if ag > 0 else self.AVG)
            defense[a] = (1 - self.ALPHA) * da + self.ALPHA * hg if a in defense else (float(hg) if hg > 0 else self.AVG * 0.9)
            game_count[h] = game_count.get(h, 0) + 1
            game_count[a] = game_count.get(a, 0) + 1

            # Track league goals for overdispersion estimation
            league_goals.setdefault(comp, []).extend([hg, ag])

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "nb_lambda_h": nb_lam_h,
                "nb_lambda_a": nb_lam_a,
                "nb_lambda_diff": nb_lam_h - nb_lam_a,
                "nb_r_param": nb_r_h,
                "nb_overdispersion": nb_overdispersion,
                "nb_btts": nb_btts,
                "nb_o25": nb_o25,
                "nb_cs00": nb_cs00,
            },
        )
