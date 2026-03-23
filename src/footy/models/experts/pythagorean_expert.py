"""PythagoreanExpert — Pythagorean expectation for regression-to-mean detection.

Theory (adapted from baseball analytics):
    Expected_Points% = GF^γ / (GF^γ + GA^γ), γ ≈ 1.35 for football
    Luck = Actual_Points - Expected_Points

Teams with high positive luck tend to regress (they've been winning close games
at an unsustainable rate). Teams with negative luck are better than their record
suggests. This is one of the strongest regression-to-mean signals in sports.

References:
- StatsBomb: Pythagorean for soccer, γ ≈ 1.2-1.7
- RMSE: ~4.35 points per 38-game season (~30% within ±1 point)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _pts


class PythagoreanExpert(Expert):
    """Detect regression-to-mean via Pythagorean expectation."""

    name = "pythagorean"

    GAMMA = 1.35  # Optimal exponent for football
    MIN_GAMES = 5  # Minimum games before Pythagorean is meaningful

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        # Feature arrays
        pyth_luck_h = np.zeros(n)
        pyth_luck_a = np.zeros(n)
        pyth_expected_ppg_h = np.zeros(n)
        pyth_expected_ppg_a = np.zeros(n)
        pyth_regression_signal = np.zeros(n)
        pyth_efficiency_h = np.zeros(n)
        pyth_efficiency_a = np.zeros(n)

        # Per-team rolling state
        team_gf: dict[str, float] = {}  # goals for (cumulative)
        team_ga: dict[str, float] = {}  # goals against (cumulative)
        team_pts: dict[str, float] = {}  # actual points (cumulative)
        team_games: dict[str, int] = {}

        g = self.GAMMA

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            # Ensure teams exist
            for t in (h, a):
                if t not in team_gf:
                    team_gf[t] = 0.0
                    team_ga[t] = 0.0
                    team_pts[t] = 0.0
                    team_games[t] = 0

            games_h = team_games[h]
            games_a = team_games[a]

            # Compute Pythagorean expected points for each team
            if games_h >= self.MIN_GAMES and team_gf[h] > 0 and team_ga[h] > 0:
                gf_h, ga_h = team_gf[h], team_ga[h]
                # Pythagorean win%: GF^γ / (GF^γ + GA^γ)
                pyth_win_pct_h = (gf_h ** g) / (gf_h ** g + ga_h ** g)
                # Expected PPG: win% * 3 + draw_component
                # Approximate: expected_ppg ≈ pyth_win_pct * 3 * draw_adjustment
                # Simpler: use the ratio directly
                expected_ppg_h = pyth_win_pct_h * 3.0 * 0.85  # ~85% of max accounts for draws
                actual_ppg_h = team_pts[h] / games_h
                luck_h = actual_ppg_h - expected_ppg_h

                pyth_luck_h[i] = luck_h
                pyth_expected_ppg_h[i] = expected_ppg_h
                pyth_efficiency_h[i] = actual_ppg_h / max(expected_ppg_h, 0.1)

            if games_a >= self.MIN_GAMES and team_gf[a] > 0 and team_ga[a] > 0:
                gf_a, ga_a = team_gf[a], team_ga[a]
                pyth_win_pct_a = (gf_a ** g) / (gf_a ** g + ga_a ** g)
                expected_ppg_a = pyth_win_pct_a * 3.0 * 0.85
                actual_ppg_a = team_pts[a] / games_a
                luck_a = actual_ppg_a - expected_ppg_a

                pyth_luck_a[i] = luck_a
                pyth_expected_ppg_a[i] = expected_ppg_a
                pyth_efficiency_a[i] = actual_ppg_a / max(expected_ppg_a, 0.1)

            # Regression signal: positive = home team likely to regress down,
            # away team likely to regress up (upset-prone)
            pyth_regression_signal[i] = pyth_luck_h[i] - pyth_luck_a[i]

            # Adjust probabilities based on regression signal
            reg = pyth_regression_signal[i]
            if abs(reg) > 0.05:
                # Positive reg = home overperforming → reduce home win prob
                adj = np.clip(reg * 0.15, -0.10, 0.10)
                p_h = 0.36 - adj
                p_a = 0.36 + adj
                p_d = 0.28
                s = p_h + p_d + p_a
                if s > 0:
                    probs[i] = [p_h / s, p_d / s, p_a / s]

            # Confidence based on data availability
            min_games = min(games_h, games_a)
            if min_games >= self.MIN_GAMES:
                conf[i] = min(0.85, 0.15 + min_games * 0.01 + abs(reg) * 0.5)

            # Update state for finished matches
            if _is_finished(r):
                hg, ag = int(r.home_goals), int(r.away_goals)
                team_gf[h] += hg
                team_ga[h] += ag
                team_gf[a] += ag
                team_ga[a] += hg
                team_pts[h] += _pts(hg, ag)
                team_pts[a] += _pts(ag, hg)
                team_games[h] += 1
                team_games[a] += 1

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "pyth_luck_h": pyth_luck_h,
                "pyth_luck_a": pyth_luck_a,
                "pyth_expected_ppg_h": pyth_expected_ppg_h,
                "pyth_expected_ppg_a": pyth_expected_ppg_a,
                "pyth_regression_signal": pyth_regression_signal,
                "pyth_efficiency_h": pyth_efficiency_h,
                "pyth_efficiency_a": pyth_efficiency_a,
            },
        )
