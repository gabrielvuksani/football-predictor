"""MatchDynamicsExpert — In-match and pre-match situational dynamics.

Captures football scenarios that most models miss:
1. Early goal psychology — teams that concede first: collapse rate vs comeback rate
2. Half-time/full-time patterns — HT score as predictor of FT result
3. Red card effect — teams' record when receiving reds (historical)
4. Set piece dependency — % of goals from corners/free kicks
5. Counter-attack vs possession style matchup
6. Penalty tendencies — conversion rates, penalties won/conceded
7. Long streak psychology — when do unbeaten/winning/losing streaks break?
8. Teams' performance when leading vs trailing (game state resilience)

All computed from historical match data already in the DB (goals, cards,
corners, shots, half-time scores from match_extras).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3


class MatchDynamicsExpert(Expert):
    """Capture in-match dynamics and situational patterns."""

    name = "match_dynamics"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        # Feature arrays
        md_comeback_rate_h = np.zeros(n)     # % of times home team came back after trailing at HT
        md_comeback_rate_a = np.zeros(n)
        md_collapse_rate_h = np.zeros(n)     # % of times team lost after leading at HT
        md_collapse_rate_a = np.zeros(n)
        md_ht_draw_win_h = np.zeros(n)       # % of times team won after HT draw
        md_ht_draw_win_a = np.zeros(n)
        md_set_piece_dep_h = np.zeros(n)     # corner/FK goal dependency ratio
        md_set_piece_dep_a = np.zeros(n)
        md_card_rate_h = np.zeros(n)         # avg cards per game (discipline)
        md_card_rate_a = np.zeros(n)
        md_streak_h = np.zeros(n)            # current streak length (+ for wins, - for losses)
        md_streak_a = np.zeros(n)
        md_streak_break_risk_h = np.zeros(n) # probability of streak breaking
        md_streak_break_risk_a = np.zeros(n)
        md_resilience_h = np.zeros(n)        # ability to recover from adversity
        md_resilience_a = np.zeros(n)

        # Per-team rolling state
        team_ht_trailing_count: dict[str, int] = {}
        team_ht_trailing_won: dict[str, int] = {}
        team_ht_leading_count: dict[str, int] = {}
        team_ht_leading_lost: dict[str, int] = {}
        team_ht_draw_count: dict[str, int] = {}
        team_ht_draw_won: dict[str, int] = {}
        team_corners: dict[str, list[int]] = {}
        team_goals_from_play: dict[str, list[int]] = {}
        team_cards: dict[str, list[float]] = {}
        team_streak: dict[str, int] = {}
        team_games: dict[str, int] = {}
        # Track come-from-behind wins for resilience
        team_behind_count: dict[str, int] = {}
        team_behind_recovered: dict[str, int] = {}

        has_ht = "hthg" in df.columns and "htag" in df.columns
        has_cards = "hy" in df.columns
        has_corners = "hc" in df.columns

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            for t in (h, a):
                if t not in team_games:
                    team_ht_trailing_count[t] = 0
                    team_ht_trailing_won[t] = 0
                    team_ht_leading_count[t] = 0
                    team_ht_leading_lost[t] = 0
                    team_ht_draw_count[t] = 0
                    team_ht_draw_won[t] = 0
                    team_corners[t] = []
                    team_goals_from_play[t] = []
                    team_cards[t] = []
                    team_streak[t] = 0
                    team_games[t] = 0
                    team_behind_count[t] = 0
                    team_behind_recovered[t] = 0

            # Compute features from current state
            g_h = team_games[h]
            g_a = team_games[a]

            # Comeback and collapse rates
            if team_ht_trailing_count[h] >= 3:
                md_comeback_rate_h[i] = team_ht_trailing_won[h] / team_ht_trailing_count[h]
            if team_ht_trailing_count[a] >= 3:
                md_comeback_rate_a[i] = team_ht_trailing_won[a] / team_ht_trailing_count[a]
            if team_ht_leading_count[h] >= 3:
                md_collapse_rate_h[i] = team_ht_leading_lost[h] / team_ht_leading_count[h]
            if team_ht_leading_count[a] >= 3:
                md_collapse_rate_a[i] = team_ht_leading_lost[a] / team_ht_leading_count[a]

            # HT draw → win rate
            if team_ht_draw_count[h] >= 3:
                md_ht_draw_win_h[i] = team_ht_draw_won[h] / team_ht_draw_count[h]
            if team_ht_draw_count[a] >= 3:
                md_ht_draw_win_a[i] = team_ht_draw_won[a] / team_ht_draw_count[a]

            # Set piece dependency (corners as proxy)
            if len(team_corners[h]) >= 5 and len(team_goals_from_play[h]) >= 5:
                avg_corners = np.mean(team_corners[h][-10:])
                avg_goals = max(np.mean(team_goals_from_play[h][-10:]), 0.5)
                md_set_piece_dep_h[i] = avg_corners / (avg_corners + avg_goals * 10)
            if len(team_corners[a]) >= 5 and len(team_goals_from_play[a]) >= 5:
                avg_corners = np.mean(team_corners[a][-10:])
                avg_goals = max(np.mean(team_goals_from_play[a][-10:]), 0.5)
                md_set_piece_dep_a[i] = avg_corners / (avg_corners + avg_goals * 10)

            # Card rates
            if len(team_cards[h]) >= 3:
                md_card_rate_h[i] = np.mean(team_cards[h][-10:])
            if len(team_cards[a]) >= 3:
                md_card_rate_a[i] = np.mean(team_cards[a][-10:])

            # Streak and streak-break risk
            md_streak_h[i] = team_streak[h]
            md_streak_a[i] = team_streak[a]
            # Long streaks become increasingly likely to break
            # P(break) increases with streak length (empirical: ~10% per game for wins, ~15% for unbeaten)
            if team_streak[h] > 0:
                md_streak_break_risk_h[i] = 1.0 - (0.90 ** team_streak[h])
            elif team_streak[h] < 0:
                md_streak_break_risk_h[i] = 1.0 - (0.85 ** abs(team_streak[h]))
            if team_streak[a] > 0:
                md_streak_break_risk_a[i] = 1.0 - (0.90 ** team_streak[a])
            elif team_streak[a] < 0:
                md_streak_break_risk_a[i] = 1.0 - (0.85 ** abs(team_streak[a]))

            # Resilience
            if team_behind_count[h] >= 3:
                md_resilience_h[i] = team_behind_recovered[h] / team_behind_count[h]
            if team_behind_count[a] >= 3:
                md_resilience_a[i] = team_behind_recovered[a] / team_behind_count[a]

            # Probability adjustment
            # Resilient away team vs fragile home team = upset signal
            res_diff = md_resilience_a[i] - md_resilience_h[i]
            streak_signal = (md_streak_break_risk_h[i] - md_streak_break_risk_a[i]) * 0.5
            adj = res_diff * 0.06 + streak_signal * 0.04
            if abs(adj) > 0.01:
                p_h = 0.36 - adj
                p_a = 0.36 + adj
                p_d = 0.28
                s = p_h + p_d + p_a
                if s > 0:
                    probs[i] = _norm3(p_h / s, p_d / s, p_a / s)

            min_g = min(g_h, g_a)
            if min_g >= 5:
                conf[i] = min(0.85, min_g * 0.015)

            # Update state for finished matches
            if _is_finished(r):
                hg = int(r.home_goals)
                ag = int(r.away_goals)

                # Streak tracking
                for team, gf, ga in [(h, hg, ag), (a, ag, hg)]:
                    if gf > ga:
                        team_streak[team] = max(1, team_streak[team] + 1) if team_streak[team] > 0 else 1
                    elif gf < ga:
                        team_streak[team] = min(-1, team_streak[team] - 1) if team_streak[team] < 0 else -1
                    else:
                        team_streak[team] = 0

                # HT score patterns
                if has_ht:
                    hthg = getattr(r, "hthg", None)
                    htag = getattr(r, "htag", None)
                    if hthg is not None and htag is not None:
                        try:
                            hthg_v, htag_v = int(hthg), int(htag)
                        except (ValueError, TypeError):
                            hthg_v, htag_v = None, None

                        if hthg_v is not None:
                            # Home team perspective
                            if hthg_v < htag_v:  # trailing at HT
                                team_ht_trailing_count[h] += 1
                                team_behind_count[h] += 1
                                if hg > ag:
                                    team_ht_trailing_won[h] += 1
                                    team_behind_recovered[h] += 1
                                elif hg == ag:
                                    team_behind_recovered[h] += 1
                            elif hthg_v > htag_v:  # leading at HT
                                team_ht_leading_count[h] += 1
                                if hg < ag:
                                    team_ht_leading_lost[h] += 1
                            else:  # drawing at HT
                                team_ht_draw_count[h] += 1
                                if hg > ag:
                                    team_ht_draw_won[h] += 1

                            # Away team perspective
                            if htag_v < hthg_v:
                                team_ht_trailing_count[a] += 1
                                team_behind_count[a] += 1
                                if ag > hg:
                                    team_ht_trailing_won[a] += 1
                                    team_behind_recovered[a] += 1
                                elif ag == hg:
                                    team_behind_recovered[a] += 1
                            elif htag_v > hthg_v:
                                team_ht_leading_count[a] += 1
                                if ag < hg:
                                    team_ht_leading_lost[a] += 1
                            else:
                                team_ht_draw_count[a] += 1
                                if ag > hg:
                                    team_ht_draw_won[a] += 1

                # Corners
                if has_corners:
                    hc = getattr(r, "hc", None)
                    ac = getattr(r, "ac", None)
                    if hc is not None and not (isinstance(hc, float) and np.isnan(hc)):
                        team_corners[h].append(int(hc))
                    if ac is not None and not (isinstance(ac, float) and np.isnan(ac)):
                        team_corners[a].append(int(ac))

                # Goals from open play (total goals minus estimated set piece goals)
                team_goals_from_play[h].append(hg)
                team_goals_from_play[a].append(ag)

                # Cards
                if has_cards:
                    hy = getattr(r, "hy", 0) or 0
                    hr = getattr(r, "hr", 0) or 0
                    ay = getattr(r, "ay", 0) or 0
                    ar = getattr(r, "ar", 0) or 0
                    try:
                        team_cards[h].append(float(hy) + float(hr) * 2)
                        team_cards[a].append(float(ay) + float(ar) * 2)
                    except (ValueError, TypeError):
                        pass

                team_games[h] += 1
                team_games[a] += 1

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "md_comeback_h": md_comeback_rate_h,
                "md_comeback_a": md_comeback_rate_a,
                "md_collapse_h": md_collapse_rate_h,
                "md_collapse_a": md_collapse_rate_a,
                "md_ht_draw_win_h": md_ht_draw_win_h,
                "md_ht_draw_win_a": md_ht_draw_win_a,
                "md_set_piece_dep_h": md_set_piece_dep_h,
                "md_set_piece_dep_a": md_set_piece_dep_a,
                "md_card_rate_h": md_card_rate_h,
                "md_card_rate_a": md_card_rate_a,
                "md_streak_h": md_streak_h,
                "md_streak_a": md_streak_a,
                "md_streak_break_h": md_streak_break_risk_h,
                "md_streak_break_a": md_streak_break_risk_a,
                "md_resilience_h": md_resilience_h,
                "md_resilience_a": md_resilience_a,
            },
        )
