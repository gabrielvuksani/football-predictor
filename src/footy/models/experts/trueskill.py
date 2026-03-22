"""TrueSkillExpert — Bayesian skill rating expert for the council.

Implements the Expert interface for the council ensemble, using the TrueSkill
rating system to produce probability predictions and rich feature extraction.

Features exported:
    - skill_diff: Overall team strength difference
    - attack_diff: Attack skill difference
    - defense_diff: Defense skill difference
    - uncertainty: Total posterior uncertainty (σ₁² + σ₂²)^0.5
    - confidence: Inverse of uncertainty (higher = more confident)
    - momentum: Recent form signal (positive = improving)
    - home_adv: Home team's ground effect advantage
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3
from footy.models.trueskill import TrueSkillEngine


class TrueSkillExpert(Expert):
    """TrueSkill-based expert for the council.

    Maintains Bayesian beliefs about team attack/defense skills, updates them
    after each match, and produces well-calibrated probability predictions.

    Name: "trueskill"
    """

    name = "trueskill"

    # Match the TrueSkill defaults
    INITIAL_MU = 1500.0
    INITIAL_SIGMA = 350.0
    BETA = 100.0
    TAU = 10.0
    DRAW_MARGIN = 45.0

    # How many recent strength snapshots to keep for momentum calculation
    MOMENTUM_WINDOW = 6

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        """Compute TrueSkill ratings and probabilities over matches."""
        n = len(df)
        engine = TrueSkillEngine(
            initial_mu=self.INITIAL_MU,
            initial_sigma=self.INITIAL_SIGMA,
            beta=self.BETA,
            tau=self.TAU,
            draw_margin=self.DRAW_MARGIN,
        )

        # Track strength history for momentum computation
        strength_history: dict[str, list[float]] = {}

        # Output arrays
        out_ph = np.zeros(n)
        out_pd = np.zeros(n)
        out_pa = np.zeros(n)
        out_conf = np.zeros(n)
        out_skill_diff = np.zeros(n)
        out_attack_diff = np.zeros(n)
        out_defense_diff = np.zeros(n)
        out_uncertainty = np.zeros(n)
        out_momentum_h = np.zeros(n)
        out_momentum_a = np.zeros(n)
        out_home_adv = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            # Predict before state update
            ph, pd, pa = engine.predict_probs(h, a)

            # Extract features pre-match
            h_skills = engine._get_team(h)
            a_skills = engine._get_team(a)

            skill_diff = h_skills.overall_strength() - a_skills.overall_strength()
            attack_diff = h_skills.attack.mu - a_skills.attack.mu
            defense_diff = h_skills.defense.mu - a_skills.defense.mu
            uncertainty = h_skills.uncertainty() + a_skills.uncertainty()
            confidence = 1.0 / (1.0 + uncertainty / 1000.0)
            home_adv = h_skills.home_advantage

            # Compute momentum from strength history (direction of recent change)
            hist_h = strength_history.get(h, [])
            hist_a = strength_history.get(a, [])
            w = self.MOMENTUM_WINDOW
            if len(hist_h) >= w:
                momentum_h = (hist_h[-1] - hist_h[-w]) / w
            elif len(hist_h) >= 2:
                momentum_h = (hist_h[-1] - hist_h[0]) / len(hist_h)
            else:
                momentum_h = 0.0
            if len(hist_a) >= w:
                momentum_a = (hist_a[-1] - hist_a[-w]) / w
            elif len(hist_a) >= 2:
                momentum_a = (hist_a[-1] - hist_a[0]) / len(hist_a)
            else:
                momentum_a = 0.0

            out_ph[i] = ph
            out_pd[i] = pd
            out_pa[i] = pa
            out_conf[i] = confidence
            out_skill_diff[i] = skill_diff
            out_attack_diff[i] = attack_diff
            out_defense_diff[i] = defense_diff
            out_uncertainty[i] = uncertainty
            out_momentum_h[i] = momentum_h
            out_momentum_a[i] = momentum_a
            out_home_adv[i] = home_adv

            # Update state AFTER recording pre-match values
            if _is_finished(r):
                hg, ag = int(r.home_goals), int(r.away_goals)
                engine.update(
                    home=h,
                    away=a,
                    home_goals=hg,
                    away_goals=ag,
                    match_idx=i,
                    league=getattr(r, "competition", None),
                )
                # Record post-update strength for momentum tracking
                h_post = engine._get_team(h)
                a_post = engine._get_team(a)
                strength_history.setdefault(h, []).append(h_post.overall_strength())
                strength_history.setdefault(a, []).append(a_post.overall_strength())

        return ExpertResult(
            probs=np.column_stack([out_ph, out_pd, out_pa]),
            confidence=out_conf,
            features={
                "ts_skill_diff": out_skill_diff,
                "ts_attack_diff": out_attack_diff,
                "ts_defense_diff": out_defense_diff,
                "ts_uncertainty": out_uncertainty,
                "ts_confidence": out_conf,
                "ts_momentum_h": out_momentum_h,
                "ts_momentum_a": out_momentum_a,
                "ts_home_adv": out_home_adv,
            },
        )
