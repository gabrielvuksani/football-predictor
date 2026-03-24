"""MoraleExpert — Team confidence, morale, and psychological state.

Models the psychological dimension of football that pure stats miss:

1. Confidence after big wins (beat a top team → morale boost)
2. Fragility after embarrassing losses (heavy defeat → crisis)
3. Recovery speed (how quickly does a team bounce back from a loss?)
4. Pressure handling (performance in "must-win" situations)
5. Consistency (low variance = reliable; high variance = unpredictable)
6. Second-half collapses (teams that fade → fitness/mental issues)

All derived from match results patterns — no external data needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3


class MoraleExpert(Expert):
    """Track team morale and psychological signals."""

    name = "morale"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        ml_confidence_h = np.zeros(n)    # morale/confidence level home
        ml_confidence_a = np.zeros(n)
        ml_fragility_h = np.zeros(n)     # vulnerability after bad results
        ml_fragility_a = np.zeros(n)
        ml_recovery_h = np.zeros(n)      # bounce-back ability
        ml_recovery_a = np.zeros(n)
        ml_consistency_h = np.zeros(n)   # result consistency (low = predictable)
        ml_consistency_a = np.zeros(n)
        ml_big_win_boost_h = np.zeros(n) # recent big win effect
        ml_big_win_boost_a = np.zeros(n)

        # Per-team state
        team_results: dict[str, list[int]] = {}  # +3/+1/0 per match
        team_gd: dict[str, list[int]] = {}       # goal difference per match
        team_last_loss_ago: dict[str, int] = {}   # matches since last loss
        team_last_big_win: dict[str, int] = {}    # matches since last 3+ goal win
        team_last_heavy_loss: dict[str, int] = {} # matches since last 3+ goal loss

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            for t in (h, a):
                if t not in team_results:
                    team_results[t] = []
                    team_gd[t] = []
                    team_last_loss_ago[t] = 99
                    team_last_big_win[t] = 99
                    team_last_heavy_loss[t] = 99

            # Compute features from history
            for t, conf_arr, frag_arr, rec_arr, cons_arr, bw_arr in [
                (h, ml_confidence_h, ml_fragility_h, ml_recovery_h, ml_consistency_h, ml_big_win_boost_h),
                (a, ml_confidence_a, ml_fragility_a, ml_recovery_a, ml_consistency_a, ml_big_win_boost_a),
            ]:
                res = team_results[t]
                gd = team_gd[t]

                if len(res) >= 5:
                    recent = res[-5:]
                    recent_gd = gd[-5:]

                    # Confidence: points per game recently + goal difference trend
                    ppg = sum(recent) / (3.0 * len(recent))  # 0-1 scale
                    gd_trend = sum(recent_gd) / len(recent_gd)
                    conf_arr[i] = ppg * 0.7 + np.clip(gd_trend / 3.0, -0.3, 0.3)

                    # Fragility: were any recent results heavy losses?
                    heavy = sum(1 for g in recent_gd if g <= -3)
                    frag_arr[i] = heavy / len(recent)

                    # Recovery: if last result was a loss, did they bounce back before?
                    bounces = 0
                    losses = 0
                    for j in range(1, len(res)):
                        if res[j - 1] == 0:  # previous was a loss
                            losses += 1
                            if res[j] == 3:  # bounced back with a win
                                bounces += 1
                    rec_arr[i] = bounces / max(losses, 1)

                    # Consistency: std of points (low = reliable)
                    cons_arr[i] = 1.0 - min(1.0, np.std(res[-10:]) / 1.5)

                    # Big win boost: recent big win = confidence boost
                    if team_last_big_win[t] <= 3:
                        bw_arr[i] = 1.0 - team_last_big_win[t] / 4.0
                    # Heavy loss fragility amplifier
                    if team_last_heavy_loss[t] <= 2:
                        frag_arr[i] += 0.3 * (1.0 - team_last_heavy_loss[t] / 3.0)

            # Probability adjustment
            morale_diff = ml_confidence_h[i] - ml_confidence_a[i]
            frag_signal = ml_fragility_h[i] - ml_fragility_a[i]  # positive = home more fragile

            adj = morale_diff * 0.08 - frag_signal * 0.05
            if abs(adj) > 0.01:
                p_h = 0.36 + adj
                p_a = 0.36 - adj
                p_d = 0.28
                s = p_h + p_d + p_a
                if s > 0:
                    probs[i] = _norm3(p_h / s, p_d / s, p_a / s)

            min_g = min(len(team_results.get(h, [])), len(team_results.get(a, [])))
            if min_g >= 5:
                conf[i] = min(0.85, min_g * 0.015 + abs(morale_diff) * 0.4)

            # Update state
            if _is_finished(r):
                hg, ag = int(r.home_goals), int(r.away_goals)

                for t, gf, ga in [(h, hg, ag), (a, ag, hg)]:
                    pts = 3 if gf > ga else (1 if gf == ga else 0)
                    team_results[t].append(pts)
                    team_gd[t].append(gf - ga)

                    # Track special events
                    if pts == 0:
                        team_last_loss_ago[t] = 0
                    else:
                        team_last_loss_ago[t] += 1

                    if gf - ga >= 3:
                        team_last_big_win[t] = 0
                    else:
                        team_last_big_win[t] += 1

                    if ga - gf >= 3:
                        team_last_heavy_loss[t] = 0
                    else:
                        team_last_heavy_loss[t] += 1

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "ml_confidence_h": ml_confidence_h,
                "ml_confidence_a": ml_confidence_a,
                "ml_fragility_h": ml_fragility_h,
                "ml_fragility_a": ml_fragility_a,
                "ml_recovery_h": ml_recovery_h,
                "ml_recovery_a": ml_recovery_a,
                "ml_consistency_h": ml_consistency_h,
                "ml_consistency_a": ml_consistency_a,
                "ml_big_win_boost_h": ml_big_win_boost_h,
                "ml_big_win_boost_a": ml_big_win_boost_a,
            },
        )
