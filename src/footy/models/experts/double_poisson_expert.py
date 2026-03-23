"""DoublePoissonExpert — Efron's Double Poisson with per-team dispersion tracking.

Uses the Double Poisson distribution (Efron 1986) which decouples variance
from the mean via an independent dispersion parameter phi.  Each team's
phi is estimated online from the observed deviation of actual goals from
the EMA-predicted rate.

Reference:
    Efron (1986) "Double Exponential Families and Their Use in
        Generalized Linear Models"
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3
from footy.models.math.double_poisson import build_double_poisson_matrix


class DoublePoissonExpert(Expert):
    """Attack/defence EMA + per-team dispersion -> Double Poisson score matrix."""

    name = "double_poisson"

    ALPHA = 0.05          # EMA smoothing for attack/defence rates
    DISP_ALPHA = 0.02     # EMA smoothing for dispersion tracking (smooth)
    MAX_GOALS = 8
    AVG = 1.35            # league-average goals per team
    HOME_ATT = 1.15       # home advantage multiplier on attack
    HOME_DEF = 0.85       # home advantage multiplier on defence (lower = better)
    DEFAULT_PHI = 0.92    # initial dispersion (slightly over-dispersed)

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)

        # Per-team rolling state
        attack: dict[str, float] = {}
        defense: dict[str, float] = {}
        phi: dict[str, float] = {}            # dispersion parameter per team
        game_count: dict[str, int] = {}
        # Running variance tracker for dispersion estimation
        # Stores EMA of (actual - expected)^2
        var_tracker: dict[str, float] = {}

        # Output arrays
        probs = np.full((n, 3), 1.0 / 3)
        conf = np.zeros(n)
        out_lam_h = np.zeros(n)
        out_lam_a = np.zeros(n)
        out_phi_h = np.zeros(n)
        out_phi_a = np.zeros(n)
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

        def _phi(t: str) -> float:
            return phi.get(t, self.DEFAULT_PHI)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            att_h, def_h = _att(h), _def(h)
            att_a, def_a = _att(a), _def(a)
            phi_h, phi_a = _phi(h), _phi(a)

            # Expected goals with home advantage
            lam_h = max(0.25, min(5.0, att_h * self.HOME_ATT * def_a / self.AVG))
            lam_a = max(0.25, min(5.0, att_a * def_h * self.HOME_DEF / self.AVG))

            out_lam_h[i] = lam_h
            out_lam_a[i] = lam_a
            out_phi_h[i] = phi_h
            out_phi_a[i] = phi_a

            # Build score matrix via Double Poisson
            mat = build_double_poisson_matrix(
                lam_h, lam_a, phi_h, phi_a, max_goals=self.MAX_GOALS,
            )

            mg = self.MAX_GOALS + 1

            # 1X2 probabilities
            p_home = float(np.tril(mat, -1).sum())
            p_draw = float(np.trace(mat))
            p_away = float(np.triu(mat, 1).sum())
            ph, pd_, pa = _norm3(p_home, p_draw, p_away)
            # Clamp output probabilities to prevent extreme predictions
            ph = max(0.03, min(0.95, ph))
            pd_ = max(0.03, min(0.95, pd_))
            pa = max(0.03, min(0.95, pa))
            # Re-normalize after clamping
            s_clamped = ph + pd_ + pa
            probs[i] = [ph / s_clamped, pd_ / s_clamped, pa / s_clamped]

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

            # Update dispersion tracking
            # phi tracks how dispersed actual goals are relative to expected.
            # If actual goals deviate more than Poisson would predict, phi < 1
            # (over-dispersed); if less, phi > 1 (under-dispersed).
            for team, actual, expected in [(h, hg, lam_h), (a, ag, lam_a)]:
                sq_dev = (actual - expected) ** 2
                if team in var_tracker:
                    var_tracker[team] = (
                        (1 - self.DISP_ALPHA) * var_tracker[team]
                        + self.DISP_ALPHA * sq_dev
                    )
                else:
                    var_tracker[team] = sq_dev

                # phi = expected / variance  (Poisson: variance == mean, so phi=1)
                # Clamp to reasonable range [0.5, 2.0]
                tracked_var = var_tracker[team]
                if tracked_var > 1e-6 and expected > 1e-6:
                    phi[team] = max(0.5, min(2.0, expected / tracked_var))
                else:
                    phi[team] = self.DEFAULT_PHI

            game_count[h] = game_count.get(h, 0) + 1
            game_count[a] = game_count.get(a, 0) + 1

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "dpois_lambda_h": out_lam_h,
                "dpois_lambda_a": out_lam_a,
                "dpois_phi_h": out_phi_h,
                "dpois_phi_a": out_phi_a,
                "dpois_p00": out_p00,
                "dpois_btts": out_btts,
                "dpois_o25": out_o25,
                "dpois_entropy": out_entropy,
                "dpois_most_likely_h": out_ml_h,
                "dpois_most_likely_a": out_ml_a,
            },
        )
