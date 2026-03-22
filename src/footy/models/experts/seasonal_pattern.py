"""SeasonalPatternExpert — season phase, matchday buckets, and cyclical calendar features.

Extracts meaningful seasonal signals:
- Season progress (continuous 0.0 to 1.0)
- Early/late season flags (first/last 6 matchdays — historically more upsets)
- Matchday bucket (early/mid/late)
- Month-of-year cyclical encoding (sin/cos)
- Season phase x league position interaction
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _f


class SeasonalPatternExpert(Expert):
    """Capture seasonal structure: phase, matchday bucket, cyclical calendar,
    and phase-position interactions."""

    name = "seasonal_pattern"

    # Typical European season: matchday 1 in Aug, ~38 matchdays ending May
    TYPICAL_SEASON_LENGTH = 38
    EARLY_CUTOFF = 6   # first N matchdays = early season
    LATE_CUTOFF = 6    # last N matchdays = late season

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        # Feature arrays
        season_progress = np.zeros(n)      # continuous 0..1
        is_early = np.zeros(n)             # first 6 matchdays flag
        is_late = np.zeros(n)              # last 6 matchdays flag
        matchday_bucket = np.zeros(n)      # 0=early, 1=mid, 2=late
        month_sin = np.zeros(n)            # cyclical month encoding
        month_cos = np.zeros(n)
        phase_x_pos_h = np.zeros(n)        # season phase x league position interaction
        phase_x_pos_a = np.zeros(n)
        fixture_congestion = np.zeros(n)

        # Track league standings internally for position interaction
        team_points: dict[str, int] = {}
        team_games: dict[str, int] = {}
        league_team_count: int = 0

        for i, r in enumerate(df.itertuples(index=False)):
            home = str(r.home_team)
            away = str(r.away_team)

            # Track unique teams for league size estimation
            for t in (home, away):
                if t not in team_points:
                    team_points[t] = 0
                    team_games[t] = 0
                    league_team_count = len(team_points)

            # ── Matchday-based features ──
            md = _f(getattr(r, "matchday", None))
            season_len = self.TYPICAL_SEASON_LENGTH

            if md > 0:
                progress = min(1.0, md / season_len)
                season_progress[i] = progress

                if md <= self.EARLY_CUTOFF:
                    is_early[i] = 1.0
                    matchday_bucket[i] = 0.0
                elif md >= season_len - self.LATE_CUTOFF + 1:
                    is_late[i] = 1.0
                    matchday_bucket[i] = 2.0
                else:
                    matchday_bucket[i] = 1.0
            else:
                # Fallback: estimate from month if matchday missing
                utc_date = getattr(r, "utc_date", None)
                if utc_date is not None:
                    try:
                        month = utc_date.month if hasattr(utc_date, "month") else int(str(utc_date).split("-")[1])
                    except Exception:
                        month = 6
                    # Map month to approximate progress
                    # Aug=0.05, Sep=0.13, Oct=0.21, Nov=0.29, Dec=0.37
                    # Jan=0.47, Feb=0.55, Mar=0.66, Apr=0.79, May=0.92
                    month_progress = {
                        8: 0.05, 9: 0.13, 10: 0.21, 11: 0.29, 12: 0.37,
                        1: 0.47, 2: 0.55, 3: 0.66, 4: 0.79, 5: 0.92,
                        6: 0.97, 7: 0.02,
                    }
                    progress = month_progress.get(month, 0.5)
                    season_progress[i] = progress
                    if progress <= 0.16:
                        is_early[i] = 1.0
                        matchday_bucket[i] = 0.0
                    elif progress >= 0.84:
                        is_late[i] = 1.0
                        matchday_bucket[i] = 2.0
                    else:
                        matchday_bucket[i] = 1.0

            # ── Month cyclical encoding (sin/cos) ──
            utc_date = getattr(r, "utc_date", None)
            if utc_date is not None:
                try:
                    month = utc_date.month if hasattr(utc_date, "month") else int(str(utc_date).split("-")[1])
                except Exception:
                    month = 6
                angle = 2.0 * math.pi * (month - 1) / 12.0
                month_sin[i] = math.sin(angle)
                month_cos[i] = math.cos(angle)

            # ── Fixture congestion ──
            rest_days = _f(getattr(r, "rest_days", None))
            if rest_days > 0:
                if rest_days <= 3:
                    fixture_congestion[i] = 1.0
                elif rest_days <= 5:
                    fixture_congestion[i] = 0.5
                else:
                    fixture_congestion[i] = 0.0

            # ── Season phase x league position interaction ──
            progress_val = season_progress[i]
            if league_team_count >= 4 and progress_val > 0.05:
                # Compute position proxy: points per game relative to league
                h_ppg = team_points.get(home, 0) / max(team_games.get(home, 0), 1)
                a_ppg = team_points.get(away, 0) / max(team_games.get(away, 0), 1)
                # Normalize ppg to [-1, 1] range (3.0 = max ppg, 0 = min)
                h_pos_signal = (h_ppg - 1.5) / 1.5
                a_pos_signal = (a_ppg - 1.5) / 1.5
                # Interaction: late season amplifies position differences
                # Top teams fight for title, bottom teams fight relegation
                phase_x_pos_h[i] = progress_val * h_pos_signal
                phase_x_pos_a[i] = progress_val * a_pos_signal

            # ── Probability adjustments ──
            draw_boost = 0.0
            home_boost = 0.0

            # Early season: more upsets, higher draw rate (teams still gelling)
            if is_early[i] > 0.5:
                draw_boost += 0.03
                home_boost -= 0.01  # weaker home advantage early on

            # Late season: more decisive results (desperation & motivation)
            if is_late[i] > 0.5:
                draw_boost -= 0.02
                # Late season: strong teams push harder at home
                if phase_x_pos_h[i] > 0.3:
                    home_boost += 0.02
                # Relegation-threatened away teams fight harder
                if phase_x_pos_a[i] < -0.3:
                    home_boost -= 0.015

            # Fixture congestion: tired legs favor draws and upsets
            if fixture_congestion[i] > 0.5:
                draw_boost += 0.015

            p_h = 0.36 + home_boost
            p_d = 0.28 + draw_boost
            p_a = 0.36 - home_boost
            s = p_h + p_d + p_a
            if s > 0:
                probs[i] = [p_h / s, p_d / s, p_a / s]

            conf[i] = min(0.35, 0.12 + abs(draw_boost) * 2.5 + abs(home_boost) * 3.0)

            # ── Update internal standings (after prediction, only finished) ──
            if _is_finished(r):
                hg = int(r.home_goals)
                ag = int(r.away_goals)
                if hg > ag:
                    team_points[home] = team_points.get(home, 0) + 3
                elif hg == ag:
                    team_points[home] = team_points.get(home, 0) + 1
                    team_points[away] = team_points.get(away, 0) + 1
                else:
                    team_points[away] = team_points.get(away, 0) + 3
                team_games[home] = team_games.get(home, 0) + 1
                team_games[away] = team_games.get(away, 0) + 1

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "sp_season_progress": season_progress,
                "sp_is_early": is_early,
                "sp_is_late": is_late,
                "sp_matchday_bucket": matchday_bucket,
                "sp_month_sin": month_sin,
                "sp_month_cos": month_cos,
                "sp_phase_x_pos_h": phase_x_pos_h,
                "sp_phase_x_pos_a": phase_x_pos_a,
                "sp_fixture_congestion": fixture_congestion,
            },
        )
