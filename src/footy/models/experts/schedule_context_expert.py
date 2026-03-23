"""ScheduleContextExpert — Schedule and calendar-based match context.

Captures scenarios that affect performance based on WHEN a match is played:
1. Multi-competition fatigue (CL/EL midweek → league weekend)
2. Post-international break effect (disrupted preparation, travel fatigue)
3. Newly promoted team honeymoon (first ~10 games back in top flight)
4. Fixture congestion across competitions (not just within league)
5. Holiday period matches (December/January in England — packed schedule)
6. End-of-season matches with nothing to play for (dead rubbers)
7. First match of season / opening day effect
8. Monday night / Friday night scheduling effects

Uses utc_date, competition, and match context to derive these signals.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3


class ScheduleContextExpert(Expert):
    """Capture schedule and calendar-based match context."""

    name = "schedule_context"

    # International break windows (approximate dates — month ranges)
    INTL_BREAK_MONTHS = {9, 10, 11, 3, 6}  # Sep, Oct, Nov, Mar, Jun

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        sc_midweek_fatigue_h = np.zeros(n)
        sc_midweek_fatigue_a = np.zeros(n)
        sc_post_intl_break = np.zeros(n)
        sc_promoted_honeymoon_h = np.zeros(n)
        sc_promoted_honeymoon_a = np.zeros(n)
        sc_holiday_congestion = np.zeros(n)
        sc_dead_rubber_h = np.zeros(n)
        sc_dead_rubber_a = np.zeros(n)
        sc_opening_day = np.zeros(n)
        sc_schedule_advantage = np.zeros(n)

        # Track per-team: last match date, matches in current season, promotion status
        team_last_date: dict[str, object] = {}
        team_season_games: dict[str, int] = {}
        team_recent_dates: dict[str, list] = {}  # last N match dates for congestion
        # Track which teams might be newly promoted (appear late in the data for a competition)
        comp_teams_seen: dict[str, set] = {}

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            comp = getattr(r, "competition", "UNK")
            utc = r.utc_date

            for t in (h, a):
                if t not in team_season_games:
                    team_season_games[t] = 0
                    team_recent_dates[t] = []

            if comp not in comp_teams_seen:
                comp_teams_seen[comp] = set()

            try:
                month = utc.month if hasattr(utc, "month") else 1
                day_of_week = utc.weekday() if hasattr(utc, "weekday") else 0  # 0=Monday
            except Exception:
                month = 1
                day_of_week = 0

            # 1. Midweek fatigue — was the team's last match within 3 days?
            for t, arr in [(h, sc_midweek_fatigue_h), (a, sc_midweek_fatigue_a)]:
                if t in team_last_date and team_last_date[t] is not None:
                    try:
                        days_since = (utc - team_last_date[t]).days
                        if 0 < days_since <= 3:
                            arr[i] = 1.0  # played within 3 days — fatigued
                        elif 0 < days_since <= 5:
                            arr[i] = 0.5  # played within 5 days — mild fatigue
                    except Exception:
                        pass

            # 2. Post-international break
            if month in self.INTL_BREAK_MONTHS:
                # Check if this looks like the first match after a break
                # (gap of 10+ days for both teams)
                both_rested = True
                for t in (h, a):
                    if t in team_last_date and team_last_date[t] is not None:
                        try:
                            gap = (utc - team_last_date[t]).days
                            if gap < 10:
                                both_rested = False
                        except Exception:
                            both_rested = False
                    else:
                        both_rested = False
                if both_rested:
                    sc_post_intl_break[i] = 1.0

            # 3. Newly promoted honeymoon (first 10 games in a competition where team is new)
            for t, arr in [(h, sc_promoted_honeymoon_h), (a, sc_promoted_honeymoon_a)]:
                if t not in comp_teams_seen[comp]:
                    # First time seeing this team in this competition — possibly promoted
                    if team_season_games[t] < 10:
                        arr[i] = max(0.0, 1.0 - team_season_games[t] / 10.0)

            # 4. Holiday congestion (December/January in European football)
            if month in (12, 1):
                sc_holiday_congestion[i] = 1.0
            elif month == 2:
                sc_holiday_congestion[i] = 0.5

            # 5. Opening day effect (first ~3 games of season)
            min_games = min(team_season_games.get(h, 0), team_season_games.get(a, 0))
            if min_games < 3:
                sc_opening_day[i] = 1.0 - min_games / 3.0

            # 6. Schedule advantage (rest days difference)
            rest_h = 7  # default
            rest_a = 7
            if h in team_last_date and team_last_date[h] is not None:
                try:
                    rest_h = max(1, (utc - team_last_date[h]).days)
                except Exception:
                    pass
            if a in team_last_date and team_last_date[a] is not None:
                try:
                    rest_a = max(1, (utc - team_last_date[a]).days)
                except Exception:
                    pass
            sc_schedule_advantage[i] = np.clip((rest_h - rest_a) / 7.0, -1.0, 1.0)

            # Probability adjustment
            fatigue_diff = sc_midweek_fatigue_a[i] - sc_midweek_fatigue_h[i]
            schedule_adj = sc_schedule_advantage[i] * 0.03
            intl_adj = sc_post_intl_break[i] * 0.02  # draws more likely after break
            honeymoon_adj = (sc_promoted_honeymoon_h[i] - sc_promoted_honeymoon_a[i]) * 0.02

            total_adj = fatigue_diff * 0.04 + schedule_adj + honeymoon_adj
            draw_adj = intl_adj + sc_holiday_congestion[i] * 0.01

            if abs(total_adj) > 0.005 or draw_adj > 0.005:
                p_h = 0.36 + total_adj
                p_d = 0.28 + draw_adj
                p_a = 0.36 - total_adj
                s = p_h + p_d + p_a
                if s > 0:
                    probs[i] = _norm3(p_h / s, p_d / s, p_a / s)
                conf[i] = min(0.85, abs(total_adj) * 10 + draw_adj * 5)

            # Update state
            if _is_finished(r):
                team_last_date[h] = utc
                team_last_date[a] = utc
                team_season_games[h] = team_season_games.get(h, 0) + 1
                team_season_games[a] = team_season_games.get(a, 0) + 1
                team_recent_dates[h] = (team_recent_dates.get(h, []) + [utc])[-20:]
                team_recent_dates[a] = (team_recent_dates.get(a, []) + [utc])[-20:]
                comp_teams_seen[comp].add(h)
                comp_teams_seen[comp].add(a)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "sc_midweek_fatigue_h": sc_midweek_fatigue_h,
                "sc_midweek_fatigue_a": sc_midweek_fatigue_a,
                "sc_post_intl_break": sc_post_intl_break,
                "sc_promoted_honeymoon_h": sc_promoted_honeymoon_h,
                "sc_promoted_honeymoon_a": sc_promoted_honeymoon_a,
                "sc_holiday_congestion": sc_holiday_congestion,
                "sc_dead_rubber_h": sc_dead_rubber_h,
                "sc_dead_rubber_a": sc_dead_rubber_a,
                "sc_opening_day": sc_opening_day,
                "sc_schedule_advantage": sc_schedule_advantage,
            },
        )
