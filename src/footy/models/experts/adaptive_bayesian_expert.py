"""AdaptiveBayesianExpert — Spike-and-slab inspired adaptive team strength.

Theory (arXiv:2508.05891, 2025):
    Uses adaptive shrinkage for team strength evolution:
    - "Spike" (tight precision) = team is consistent, borrow from history
    - "Slab" (wide precision) = team has changed, allow rapid adjustment
    - Data decides which component is active per team per period

    Achieved 5.5% RPS improvement over fixed-precision models.

Implementation: simplified version using adaptive process noise.
When a team's recent results diverge from their historical strength,
the process noise increases (slab active), allowing faster adaptation.
When results are consistent, process noise decreases (spike active).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3


class AdaptiveBayesianExpert(Expert):
    """Spike-and-slab inspired adaptive Bayesian team strength."""

    name = "adaptive_bayesian"

    # Hyperparameters
    BASE_PROCESS_NOISE = 0.02    # minimum process noise (spike)
    MAX_PROCESS_NOISE = 0.20     # maximum process noise (slab)
    MEASUREMENT_NOISE = 0.25     # observation noise
    DIVERGENCE_THRESHOLD = 1.5   # deviation threshold to activate slab
    DECAY = 0.95                 # how fast divergence decays

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        # Feature arrays
        abs_strength_h = np.zeros(n)
        abs_strength_a = np.zeros(n)
        abs_diff = np.zeros(n)
        abs_regime_h = np.zeros(n)  # 0 = spike (stable), 1 = slab (changing)
        abs_regime_a = np.zeros(n)
        abs_change_speed_h = np.zeros(n)
        abs_change_speed_a = np.zeros(n)
        abs_uncertainty_h = np.zeros(n)
        abs_uncertainty_a = np.zeros(n)

        # Per-team state
        strength: dict[str, float] = {}     # current estimated strength
        variance: dict[str, float] = {}     # estimation uncertainty
        divergence: dict[str, float] = {}   # running divergence from prediction
        prev_strength: dict[str, float] = {}  # previous strength for change speed
        games: dict[str, int] = {}

        home_adv = 0.25  # initial home advantage estimate

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            for t in (h, a):
                if t not in strength:
                    strength[t] = 0.0
                    variance[t] = 1.0
                    divergence[t] = 0.0
                    prev_strength[t] = 0.0
                    games[t] = 0

            # Current estimates
            s_h = strength[h]
            s_a = strength[a]
            v_h = variance[h]
            v_a = variance[a]
            d_h = divergence[h]
            d_a = divergence[a]

            # Determine regime: spike or slab
            # High divergence = team is changing (slab), low = stable (spike)
            regime_h = min(1.0, d_h / self.DIVERGENCE_THRESHOLD)
            regime_a = min(1.0, d_a / self.DIVERGENCE_THRESHOLD)

            abs_strength_h[i] = s_h
            abs_strength_a[i] = s_a
            abs_diff[i] = s_h - s_a + home_adv
            abs_regime_h[i] = regime_h
            abs_regime_a[i] = regime_a
            abs_change_speed_h[i] = abs(s_h - prev_strength[h])
            abs_change_speed_a[i] = abs(s_a - prev_strength[a])
            abs_uncertainty_h[i] = v_h
            abs_uncertainty_a[i] = v_a

            # Expected goal difference
            expected_gd = s_h - s_a + home_adv

            # Convert to probabilities via normal approximation
            total_var = v_h + v_a + self.MEASUREMENT_NOISE
            std = math.sqrt(max(total_var, 0.01))

            # P(H) = P(GD > 0.5), P(D) = P(-0.5 < GD < 0.5), P(A) = P(GD < -0.5)
            from math import erf
            def phi(x):
                return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))

            p_h = 1.0 - phi((0.5 - expected_gd) / std)
            p_a = phi((-0.5 - expected_gd) / std)
            p_d = 1.0 - p_h - p_a

            probs[i] = _norm3(p_h, max(p_d, 0.05), p_a)

            # Confidence from data and certainty
            min_g = min(games[h], games[a])
            conf[i] = min(0.85, min_g * 0.015 + (1.0 - (v_h + v_a) * 0.3))
            conf[i] = max(conf[i], 0.0)

            # Bayesian update
            if _is_finished(r):
                hg, ag = int(r.home_goals), int(r.away_goals)
                actual_gd = hg - ag

                # Prediction error
                error = actual_gd - expected_gd

                for t, sign in [(h, 1.0), (a, -1.0)]:
                    v_t = variance[t]
                    d_t = divergence[t]

                    # Adaptive process noise (spike-and-slab)
                    regime_t = min(1.0, d_t / self.DIVERGENCE_THRESHOLD)
                    process_noise = (
                        self.BASE_PROCESS_NOISE * (1.0 - regime_t) +
                        self.MAX_PROCESS_NOISE * regime_t
                    )

                    # Kalman-style predict step
                    v_predict = v_t + process_noise

                    # Kalman gain
                    K = v_predict / (v_predict + self.MEASUREMENT_NOISE)

                    # Update strength
                    prev_strength[t] = strength[t]
                    strength[t] = strength[t] + K * sign * error

                    # Update variance
                    variance[t] = (1.0 - K) * v_predict

                    # Update divergence: exponential decay + new surprise
                    surprise = abs(sign * error)
                    divergence[t] = self.DECAY * d_t + (1.0 - self.DECAY) * surprise

                    games[t] += 1

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "abs_strength_h": abs_strength_h,
                "abs_strength_a": abs_strength_a,
                "abs_diff": abs_diff,
                "abs_regime_h": abs_regime_h,
                "abs_regime_a": abs_regime_a,
                "abs_change_speed_h": abs_change_speed_h,
                "abs_change_speed_a": abs_change_speed_a,
                "abs_uncertainty_h": abs_uncertainty_h,
                "abs_uncertainty_a": abs_uncertainty_a,
            },
        )
