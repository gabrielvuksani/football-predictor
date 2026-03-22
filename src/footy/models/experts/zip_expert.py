"""ZIPExpert — Zero-Inflated Poisson + Kalman-filtered strengths.

Combines two cutting-edge football prediction techniques:

1. **Zero-Inflated Poisson (ZIP):** Accounts for the excess of 0-0 and
   low-scoring draws that plain Poisson under-predicts.  The ZIP model
   adds a "structural zero" component (probability `π` of an extra-
   zero state — parking the bus, extreme weather, dead rubber, etc.).

2. **Kalman-filtered attack/defence strengths:** Instead of raw EWMA,
   this expert maintains a Kalman state estimate of each team's
   attacking and defensive capability, with process noise modelling the
   natural evolution of team strength over time (transfers, injuries,
   coaching changes).

Reference:
    Karlis & Ntzoufras (2003) — Analysis of sports data using bivariate
    Poisson models.  Bayesian state-space models (Owen 2011).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist

from footy.models.experts._base import Expert, ExpertResult, _f, _is_finished, _norm3
from footy.models.experimental_math import (
    zero_inflated_poisson_pmf,
    build_zip_score_matrix,
    kalman_attack_defence_update,
)


class ZIPExpert(Expert):
    """Zero-Inflated Poisson with Kalman-filtered strengths."""
    name = "zip_model"

    # Kalman filter hyperparameters
    PROCESS_NOISE = 0.04       # σ² for team-strength random walk
    OBSERVATION_NOISE = 1.2    # measurement noise for goals
    INITIAL_VARIANCE = 0.5     # initial uncertainty

    # ZIP hyperparameters
    DEFAULT_ZERO_INFLATION = 0.06  # base rate for structural zeros
    MAX_GOALS = 8

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)

        # Output arrays
        probs = np.full((n, 3), 1.0 / 3)
        conf = np.zeros(n)
        zip_ph = np.zeros(n); zip_pd = np.zeros(n); zip_pa = np.zeros(n)
        zip_btts = np.zeros(n); zip_o25 = np.zeros(n)
        zip_pi = np.zeros(n)  # estimated zero-inflation parameter
        lam_h_out = np.zeros(n); lam_a_out = np.zeros(n)
        kalman_atk_h = np.zeros(n); kalman_atk_a = np.zeros(n)
        kalman_def_h = np.zeros(n); kalman_def_a = np.zeros(n)
        kalman_var_h = np.zeros(n); kalman_var_a = np.zeros(n)

        # Kalman state: {team: {"atk_mean", "atk_var", "def_mean", "def_var"}}
        state: dict[str, dict[str, float]] = {}
        # Competition-level home advantage accumulators
        comp_home_goals: dict[str, list[float]] = {}
        comp_away_goals: dict[str, list[float]] = {}
        # 0-0 draw rate tracker for adaptive zero-inflation
        comp_zero_count: dict[str, int] = {}
        comp_match_count: dict[str, int] = {}

        def _get_state(team: str) -> dict[str, float]:
            if team not in state:
                state[team] = {
                    "atk_mean": 1.15,
                    "atk_var": self.INITIAL_VARIANCE,
                    "def_mean": 1.15,
                    "def_var": self.INITIAL_VARIANCE,
                }
            return state[team]

        for i, r in enumerate(df.itertuples(index=False)):
            h = getattr(r, "home_team", "")
            a = getattr(r, "away_team", "")
            comp = str(getattr(r, "competition", "UNK"))

            s_h = _get_state(h)
            s_a = _get_state(a)

            # Compute expected goals from Kalman state
            home_adv = 0.25  # default
            h_goals = comp_home_goals.get(comp, [])
            a_goals = comp_away_goals.get(comp, [])
            if len(h_goals) >= 30:
                avg_h = float(np.mean(h_goals[-200:]))
                avg_a = float(np.mean(a_goals[-200:]))
                home_adv = max(0.0, avg_h - avg_a) * 0.5

            lam_h = max(0.3, s_h["atk_mean"] * (1.0 + s_a["def_mean"] - 1.0) * 0.5 + home_adv)
            lam_a = max(0.3, s_a["atk_mean"] * (1.0 + s_h["def_mean"] - 1.0) * 0.5)

            # Adaptive zero-inflation: higher when competition has more 0-0s
            n_zero = comp_zero_count.get(comp, 0)
            n_total = comp_match_count.get(comp, 0)
            if n_total >= 20:
                pi = max(0.02, min(0.15, n_zero / n_total))
            else:
                pi = self.DEFAULT_ZERO_INFLATION

            # Build ZIP score matrix
            try:
                mat = build_zip_score_matrix(
                    lam_h, lam_a,
                    zero_mass_h=pi * 0.7,  # slightly less zero-inflation for home
                    zero_mass_a=pi,
                    max_goals=self.MAX_GOALS,
                )
                total = mat.sum()
                if total > 0:
                    mat /= total

                # Extract probabilities
                mg = mat.shape[0]
                p_h = sum(mat[hh, aa] for hh in range(mg) for aa in range(mg) if hh > aa)
                p_d = sum(mat[hh, hh] for hh in range(mg))
                p_a = sum(mat[hh, aa] for hh in range(mg) for aa in range(mg) if aa > hh)

                s = p_h + p_d + p_a
                if s > 0:
                    zip_ph[i] = p_h / s
                    zip_pd[i] = p_d / s
                    zip_pa[i] = p_a / s
                    probs[i] = [zip_ph[i], zip_pd[i], zip_pa[i]]

                # BTTS: P(home > 0 AND away > 0)
                zip_btts[i] = sum(mat[hh, aa] for hh in range(1, mg) for aa in range(1, mg))
                # Over 2.5
                zip_o25[i] = sum(mat[hh, aa] for hh in range(mg) for aa in range(mg) if hh + aa > 2)
            except Exception:
                probs[i] = [1.0 / 3, 1.0 / 3, 1.0 / 3]

            zip_pi[i] = pi
            lam_h_out[i] = lam_h
            lam_a_out[i] = lam_a
            kalman_atk_h[i] = s_h["atk_mean"]
            kalman_atk_a[i] = s_a["atk_mean"]
            kalman_def_h[i] = s_h["def_mean"]
            kalman_def_a[i] = s_a["def_mean"]
            kalman_var_h[i] = s_h["atk_var"] + s_h["def_var"]
            kalman_var_a[i] = s_a["atk_var"] + s_a["def_var"]

            # Confidence based on state certainty
            total_var = s_h["atk_var"] + s_h["def_var"] + s_a["atk_var"] + s_a["def_var"]
            conf[i] = max(0.1, min(0.9, 1.0 - total_var / (4 * self.INITIAL_VARIANCE)))

            # --- Update Kalman state for finished matches only ---
            if _is_finished(r):
                hg, ag = int(r.home_goals), int(r.away_goals)

                # Update home team attack (observed: goals scored)
                s_h["atk_var"] += self.PROCESS_NOISE
                k_gain = s_h["atk_var"] / (s_h["atk_var"] + self.OBSERVATION_NOISE)
                s_h["atk_mean"] += k_gain * (hg - s_h["atk_mean"])
                s_h["atk_var"] *= (1 - k_gain)

                # Update home team defence (observed: goals conceded)
                s_h["def_var"] += self.PROCESS_NOISE
                k_gain = s_h["def_var"] / (s_h["def_var"] + self.OBSERVATION_NOISE)
                s_h["def_mean"] += k_gain * (ag - s_h["def_mean"])
                s_h["def_var"] *= (1 - k_gain)

                # Update away team attack
                s_a["atk_var"] += self.PROCESS_NOISE
                k_gain = s_a["atk_var"] / (s_a["atk_var"] + self.OBSERVATION_NOISE)
                s_a["atk_mean"] += k_gain * (ag - s_a["atk_mean"])
                s_a["atk_var"] *= (1 - k_gain)

                # Update away team defence
                s_a["def_var"] += self.PROCESS_NOISE
                k_gain = s_a["def_var"] / (s_a["def_var"] + self.OBSERVATION_NOISE)
                s_a["def_mean"] += k_gain * (hg - s_a["def_mean"])
                s_a["def_var"] *= (1 - k_gain)

                # Track competition stats
                comp_home_goals.setdefault(comp, []).append(float(hg))
                comp_away_goals.setdefault(comp, []).append(float(ag))
                comp_match_count[comp] = comp_match_count.get(comp, 0) + 1
                if hg == 0 and ag == 0:
                    comp_zero_count[comp] = comp_zero_count.get(comp, 0) + 1

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "zip_ph": zip_ph, "zip_pd": zip_pd, "zip_pa": zip_pa,
                "zip_btts": zip_btts, "zip_o25": zip_o25,
                "zip_pi": zip_pi,
                "zip_lam_h": lam_h_out, "zip_lam_a": lam_a_out,
                "zip_kalman_atk_h": kalman_atk_h, "zip_kalman_atk_a": kalman_atk_a,
                "zip_kalman_def_h": kalman_def_h, "zip_kalman_def_a": kalman_def_a,
                "zip_kalman_var_h": kalman_var_h, "zip_kalman_var_a": kalman_var_a,
            },
        )
