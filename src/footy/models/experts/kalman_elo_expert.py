"""KalmanEloExpert — Kalman-filtered team strength with optimal Bayesian updates.

Unlike standard Elo (fixed K-factor), the Kalman filter provides OPTIMAL
updates by automatically balancing process noise (how much team strength
changes between games) against observation noise (match randomness).

The Kalman gain adapts: uncertain teams (new, volatile) get large updates;
established teams get small updates. This naturally handles promoted teams,
early-season uncertainty, and mid-season stability.

Math:
    State: θ_t (team strength), unobserved
    Process: θ_t = θ_{t-1} + η_t,  η_t ~ N(0, Q)
    Observation: result_t = f(θ_home - θ_away) + ε_t
    Kalman gain: K = P_{t|t-1} / (P_{t|t-1} + R)
    Update: θ_t = θ_{t-1} + K × (observed - expected)
    Uncertainty: P_{t|t} = (1 - K) × P_{t|t-1}

References:
    Harvey (1989) "Forecasting, Structural Time Series and the Kalman Filter"
    Koopman & Lit (2015) "Dynamic bivariate Poisson model"
    Rue & Salvesen (2000) "Prediction and retrospective analysis of soccer"
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3
from footy.models.math.kalman import KalmanStrengthState, kalman_attack_defence_update


class KalmanEloExpert(Expert):
    """Kalman-filtered team strength with natural uncertainty quantification."""

    name = "kalman_elo"

    PROCESS_VAR = 0.04     # how much team strength can change per match
    MEASUREMENT_VAR = 0.25  # observation noise (match randomness)
    HOME_ADV = 0.25        # log-space home advantage

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        states: dict[str, KalmanStrengthState] = {}

        # Output arrays
        kalman_h = np.zeros(n)
        kalman_a = np.zeros(n)
        kalman_diff = np.zeros(n)
        kalman_unc_h = np.zeros(n)
        kalman_unc_a = np.zeros(n)
        kalman_gain_h = np.zeros(n)
        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            h_state = states.get(h, KalmanStrengthState())
            a_state = states.get(a, KalmanStrengthState())

            # Compute strength difference (attack - opponent defense + home advantage)
            strength_h = h_state.attack - a_state.defence + self.HOME_ADV
            strength_a = a_state.attack - h_state.defence

            # Convert to expected goals (exponential link)
            exp_h = max(0.3, min(4.0, math.exp(strength_h) * 1.35))
            exp_a = max(0.3, min(4.0, math.exp(strength_a) * 1.35))

            # 1X2 probabilities from normal approximation of goal difference
            gd_mean = exp_h - exp_a
            gd_var = max(exp_h + exp_a, 0.5)
            gd_std = math.sqrt(gd_var)

            from scipy.stats import norm
            p_home = 1.0 - norm.cdf(0.5, gd_mean, gd_std)
            p_draw = norm.cdf(0.5, gd_mean, gd_std) - norm.cdf(-0.5, gd_mean, gd_std)
            p_away = norm.cdf(-0.5, gd_mean, gd_std)
            ph, pd_, pa = _norm3(p_home, p_draw, p_away)

            # Store pre-match features
            kalman_h[i] = h_state.attack - h_state.defence
            kalman_a[i] = a_state.attack - a_state.defence
            kalman_diff[i] = kalman_h[i] - kalman_a[i]
            kalman_unc_h[i] = h_state.variance
            kalman_unc_a[i] = a_state.variance
            probs[i] = [ph, pd_, pa]

            # Confidence: based on uncertainty (lower variance = higher confidence)
            avg_var = (h_state.variance + a_state.variance) / 2.0
            conf[i] = max(0.0, min(1.0, 1.0 - avg_var))

            # Kalman gain for interpretability
            prior_var = h_state.variance + self.PROCESS_VAR
            k_gain = prior_var / (prior_var + self.MEASUREMENT_VAR)
            kalman_gain_h[i] = k_gain

            # Update state AFTER recording pre-match values
            if not _is_finished(r):
                continue
            hg, ag = int(r.home_goals), int(r.away_goals)

            # Update home team state
            new_h = kalman_attack_defence_update(
                h_state,
                observed_for=float(hg),
                observed_against=float(ag),
                expected_for=exp_h,
                expected_against=exp_a,
                process_var=self.PROCESS_VAR,
                measurement_var=self.MEASUREMENT_VAR,
            )
            states[h] = new_h

            # Update away team state
            new_a = kalman_attack_defence_update(
                a_state,
                observed_for=float(ag),
                observed_against=float(hg),
                expected_for=exp_a,
                expected_against=exp_h,
                process_var=self.PROCESS_VAR,
                measurement_var=self.MEASUREMENT_VAR,
            )
            states[a] = new_a

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "kalman_strength_h": kalman_h,
                "kalman_strength_a": kalman_a,
                "kalman_diff": kalman_diff,
                "kalman_uncertainty_h": kalman_unc_h,
                "kalman_uncertainty_a": kalman_unc_a,
                "kalman_gain": kalman_gain_h,
            },
        )
