"""GASExpert — Generalized Autoregressive Score (score-driven) dynamic team strength.

Theory (Koopman & Lit, 2019):
    f_{t+1} = omega + A * s_t + B * f_t

    where s_t is the scaled score (gradient) of the conditional Poisson log-likelihood.

Advantages over Kalman filtering:
- Closed-form likelihood (fast estimation)
- Naturally handles Poisson observations (not Gaussian assumption)
- Robust to outliers (score function downweights surprising results)
- Successfully applied to EPL and Serie A for 1X2, O/U, and red card prediction

References:
- Koopman & Lit (2019): "A Dynamic Bivariate Poisson Model for Analysing
  and Forecasting Match Results in the English Premier League" (JRSS-C)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3


class GASExpert(Expert):
    """Score-driven dynamic team strength estimation."""

    name = "gas"

    # GAS parameters (could be learned from data; these are reasonable defaults)
    OMEGA = 0.0        # intercept (mean-reversion target)
    A = 0.05           # score coefficient (learning rate from surprises)
    B = 0.98           # autoregressive coefficient (persistence)
    HOME_ADV = 0.25    # initial home advantage in log-space

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        # Feature arrays
        gas_strength_h = np.zeros(n)
        gas_strength_a = np.zeros(n)
        gas_diff = np.zeros(n)
        gas_home_adv = np.zeros(n)
        gas_volatility_h = np.zeros(n)
        gas_volatility_a = np.zeros(n)

        # Per-team state: attack and defence strengths in log-space
        attack: dict[str, float] = {}   # log(attack rate)
        defence: dict[str, float] = {}  # log(defence rate) — higher = worse defence
        match_count: dict[str, int] = {}
        # Track recent score magnitudes for volatility
        recent_scores: dict[str, list[float]] = {}

        # Per-competition parameter learning from data
        _comp_goals: dict[str, list[int]] = {}
        _learned_params: dict[str, dict] = {}

        omega, A, B = self.OMEGA, self.A, self.B

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            # Select per-competition learned params or defaults
            comp = getattr(r, 'competition', 'default')
            if comp in _learned_params:
                omega = _learned_params[comp]['omega']
                A = _learned_params[comp]['A']
                B = _learned_params[comp]['B']
            else:
                omega, A, B = self.OMEGA, self.A, self.B

            for t in (h, a):
                if t not in attack:
                    attack[t] = 0.0
                    defence[t] = 0.0
                    match_count[t] = 0
                    recent_scores[t] = []

            # Current strength = attack - opponent's defence + home advantage
            strength_h = attack[h] - defence[a] + self.HOME_ADV
            strength_a = attack[a] - defence[h]

            gas_strength_h[i] = strength_h
            gas_strength_a[i] = strength_a
            gas_diff[i] = strength_h - strength_a
            gas_home_adv[i] = self.HOME_ADV

            # Expected goals from GAS strengths
            lambda_h = math.exp(max(-3, min(3, strength_h)))  # clamp for numerical safety
            lambda_a = math.exp(max(-3, min(3, strength_a)))

            # Compute 1X2 probabilities from Poisson with these lambdas
            p_h, p_d, p_a = 0.0, 0.0, 0.0
            for hg in range(8):
                for ag in range(8):
                    p_hg = math.exp(-lambda_h) * (lambda_h ** hg) / math.factorial(hg)
                    p_ag = math.exp(-lambda_a) * (lambda_a ** ag) / math.factorial(ag)
                    p = p_hg * p_ag
                    if hg > ag:
                        p_h += p
                    elif hg == ag:
                        p_d += p
                    else:
                        p_a += p

            probs[i] = _norm3(p_h, p_d, p_a)

            # Volatility from recent score deviations
            if len(recent_scores[h]) >= 3:
                gas_volatility_h[i] = float(np.std(recent_scores[h][-10:]))
            if len(recent_scores[a]) >= 3:
                gas_volatility_a[i] = float(np.std(recent_scores[a][-10:]))

            # Confidence from match count
            min_mc = min(match_count[h], match_count[a])
            conf[i] = min(0.85, min_mc * 0.02)

            # GAS update: use the SCORE of the Poisson log-likelihood
            if _is_finished(r):
                hg, ag = int(r.home_goals), int(r.away_goals)

                # Collect goals for parameter learning
                _comp_goals.setdefault(comp, []).append(hg)
                _comp_goals[comp].append(ag)
                # Learn params after warmup (100+ goals per competition)
                if len(_comp_goals[comp]) >= 100 and comp not in _learned_params:
                    from footy.models.parameter_learner import learn_gas_params
                    _learned_params[comp] = learn_gas_params(
                        np.array(_comp_goals[comp])
                    )
                    omega = _learned_params[comp]['omega']
                    A = _learned_params[comp]['A']
                    B = _learned_params[comp]['B']

                # Poisson score: s_t = y_t - exp(f_t) (gradient of Poisson log-lik)
                score_h = hg - lambda_h  # positive = scored more than expected
                score_a = ag - lambda_a

                # GAS update: f_{t+1} = omega + A * s_t + B * f_t
                # Update attack strengths
                attack[h] = omega + A * score_h + B * attack[h]
                attack[a] = omega + A * score_a + B * attack[a]

                # Update defence strengths (opponent scored more than expected = worse defence)
                defence[a] = omega + A * score_h + B * defence[a]  # away team's defence
                defence[h] = omega + A * score_a + B * defence[h]  # home team's defence

                match_count[h] += 1
                match_count[a] += 1

                # Track score magnitudes for volatility
                recent_scores[h].append(float(abs(score_h)))
                recent_scores[a].append(float(abs(score_a)))

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "gas_strength_h": gas_strength_h,
                "gas_strength_a": gas_strength_a,
                "gas_diff": gas_diff,
                "gas_home_adv": gas_home_adv,
                "gas_volatility_h": gas_volatility_h,
                "gas_volatility_a": gas_volatility_a,
            },
        )
