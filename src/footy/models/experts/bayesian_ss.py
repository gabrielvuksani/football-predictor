"""Bayesian State-Space Expert — Dynamic strength-based predictions.

Uses the Bayesian State-Space Engine to produce predictions with proper
uncertainty quantification, Rue-Salvesen psychological effects, and
zero-inflated score distributions.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished


class BayesianStateSpaceExpert(Expert):
    """Expert that uses the Bayesian State-Space model.

    Maintains a BayesianStateSpaceEngine internally, processes all historical
    matches to learn time-varying team strengths, then produces predictions
    for upcoming matches.
    """

    name = "bayesian_ss"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        from footy.models.bayesian_engine import BayesianStateSpaceEngine

        n = len(df)
        probs = np.full((n, 3), 1 / 3)
        confidence = np.full(n, 0.3)
        features: dict[str, np.ndarray] = {
            "bss_lambda_h": np.zeros(n),
            "bss_lambda_a": np.zeros(n),
            "bss_confidence": np.zeros(n),
            "bss_rs_gamma": np.zeros(n),
            "bss_zero_infl": np.zeros(n),
            "bss_momentum_h": np.zeros(n),
            "bss_momentum_a": np.zeros(n),
            "bss_attack_h": np.zeros(n),
            "bss_attack_a": np.zeros(n),
            "bss_defense_h": np.zeros(n),
            "bss_defense_a": np.zeros(n),
        }

        engine = BayesianStateSpaceEngine(
            home_advantage=0.27,
            rue_salvesen_gamma=0.08,
            zero_inflation_base=0.03,
            momentum_decay=0.85,
            learning_rate_adapt=True,
            max_goals=8,
        )

        for i, row in enumerate(df.itertuples(index=False)):
            home = str(row.home_team)
            away = str(row.away_team)

            # Generate prediction BEFORE updating (so we don't leak this match's result)
            pred = engine.predict(home, away)
            probs[i] = [pred.p_home, pred.p_draw, pred.p_away]
            confidence[i] = pred.confidence

            features["bss_lambda_h"][i] = pred.lambda_home
            features["bss_lambda_a"][i] = pred.lambda_away
            features["bss_confidence"][i] = pred.confidence
            features["bss_rs_gamma"][i] = pred.rue_salvesen_gamma
            features["bss_zero_infl"][i] = pred.zero_inflation

            # Team-level features
            h_state = engine._get_team(home)
            a_state = engine._get_team(away)
            features["bss_momentum_h"][i] = h_state.momentum
            features["bss_momentum_a"][i] = a_state.momentum
            features["bss_attack_h"][i] = h_state.attack_mean
            features["bss_attack_a"][i] = a_state.attack_mean
            features["bss_defense_h"][i] = h_state.defense_mean
            features["bss_defense_a"][i] = a_state.defense_mean

            # Update engine state AFTER prediction (only for finished matches)
            if _is_finished(row):
                hg = int(row.home_goals)
                ag = int(row.away_goals)
                ts = float(row.utc_date.timestamp()) if hasattr(row.utc_date, "timestamp") else 0
                engine.update(home, away, hg, ag, timestamp=ts)

        return ExpertResult(probs=probs, confidence=confidence, features=features)
