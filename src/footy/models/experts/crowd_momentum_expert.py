"""CrowdMomentumExpert — Stadium atmosphere and fan impact signals.

Uses stadium capacity, venue data, and historical home/away performance
to model the "12th man" effect. Key signals:

1. Stadium fill rate — packed stadiums amplify home advantage
2. Large stadium effect — bigger crowds = more pressure on away team
3. Home fortress rating — % of home points won
4. Away fortress rating — teams that travel well
5. Altitude advantage — high-altitude stadiums affect visiting teams
6. Synthetic pitch disadvantage — artificial surfaces change play style

Data from: stadiums table (capacity, altitude, surface), venue_stats table,
and historical home/away results from matches table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3


class CrowdMomentumExpert(Expert):
    """Model stadium atmosphere and crowd impact on match outcomes."""

    name = "crowd_momentum"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        cm_home_fortress_h = np.zeros(n)
        cm_away_fortress_a = np.zeros(n)
        cm_capacity = np.zeros(n)
        cm_altitude = np.zeros(n)
        cm_synthetic = np.zeros(n)
        cm_home_dominance = np.zeros(n)
        cm_travel_factor = np.zeros(n)

        # Track home/away records per team
        team_home_w: dict[str, int] = {}
        team_home_d: dict[str, int] = {}
        team_home_l: dict[str, int] = {}
        team_away_w: dict[str, int] = {}
        team_away_d: dict[str, int] = {}
        team_away_l: dict[str, int] = {}
        team_home_games: dict[str, int] = {}
        team_away_games: dict[str, int] = {}

        has_capacity = "venue_capacity" in df.columns
        has_altitude = "venue_altitude_m" in df.columns
        has_surface = "venue_surface" in df.columns

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            for t in (h, a):
                if t not in team_home_games:
                    team_home_w[t] = 0
                    team_home_d[t] = 0
                    team_home_l[t] = 0
                    team_away_w[t] = 0
                    team_away_d[t] = 0
                    team_away_l[t] = 0
                    team_home_games[t] = 0
                    team_away_games[t] = 0

            # Home fortress: what % of home games does this team win?
            hg = team_home_games[h]
            if hg >= 5:
                cm_home_fortress_h[i] = team_home_w[h] / hg

            # Away fortress: what % of away games does this team win?
            ag = team_away_games[a]
            if ag >= 5:
                cm_away_fortress_a[i] = team_away_w[a] / ag

            # Stadium capacity
            if has_capacity:
                cap = getattr(r, "venue_capacity", None)
                try:
                    if cap is not None and not pd.isna(cap) and float(cap) > 0:
                        cm_capacity[i] = min(1.0, float(cap) / 80000.0)  # normalize
                except (TypeError, ValueError):
                    pass

            # Altitude advantage
            if has_altitude:
                alt = getattr(r, "venue_altitude_m", None)
                try:
                    if alt is not None and not pd.isna(alt) and float(alt) > 500:
                        cm_altitude[i] = min(1.0, float(alt) / 3000.0)
                except (TypeError, ValueError):
                    pass

            # Synthetic surface
            if has_surface:
                surface = getattr(r, "venue_surface", None)
                if surface and isinstance(surface, str) and "artificial" in surface.lower():
                    cm_synthetic[i] = 1.0

            # Home dominance: fortress rating differential
            cm_home_dominance[i] = cm_home_fortress_h[i] - cm_away_fortress_a[i]

            # Probability adjustment
            fortress_diff = cm_home_dominance[i]
            if abs(fortress_diff) > 0.05:
                adj = fortress_diff * 0.10
                p_h = 0.36 + adj + cm_capacity[i] * 0.02 + cm_altitude[i] * 0.03
                p_a = 0.36 - adj
                p_d = 0.28
                s = p_h + p_d + p_a
                if s > 0:
                    probs[i] = _norm3(p_h / s, p_d / s, p_a / s)

            min_g = min(team_home_games.get(h, 0), team_away_games.get(a, 0))
            if min_g >= 3:
                conf[i] = min(0.85, min_g * 0.02)

            # Update state
            if _is_finished(r):
                hg_v = int(r.home_goals)
                ag_v = int(r.away_goals)

                team_home_games[h] += 1
                team_away_games[a] += 1

                if hg_v > ag_v:
                    team_home_w[h] += 1
                    team_away_l[a] += 1
                elif hg_v == ag_v:
                    team_home_d[h] += 1
                    team_away_d[a] += 1
                else:
                    team_home_l[h] += 1
                    team_away_w[a] += 1

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "cm_home_fortress_h": cm_home_fortress_h,
                "cm_away_fortress_a": cm_away_fortress_a,
                "cm_capacity": cm_capacity,
                "cm_altitude": cm_altitude,
                "cm_synthetic": cm_synthetic,
                "cm_home_dominance": cm_home_dominance,
                "cm_travel_factor": cm_travel_factor,
            },
        )
