"""OpponentAdjustedExpert — Normalize all team stats by opponent strength.

Research finding: opponent-adjusted metrics are in the top 3 feature categories
for prediction accuracy. A team beating weak opponents looks strong on paper
but isn't — adjusting for opponent quality reveals true strength.

Method: For each team, weight their stats by the inverse of their opponents'
Elo rating. Beating a 2000-rated team counts more than beating a 1200-rated team.

Also computes:
- PPDA (Passes Per Defensive Action) rolling average as pressing intensity proxy
- Goal scoring style (what % from set pieces vs open play)
- Points-to-safety and points-to-title calculations
- Team-specific post-international-break record
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3, _pts


class OpponentAdjustedExpert(Expert):
    """Opponent-quality-adjusted team metrics and advanced derived stats."""

    name = "opponent_adjusted"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        # Features
        oa_ppg_h = np.zeros(n)         # opponent-adjusted PPG home team
        oa_ppg_a = np.zeros(n)         # opponent-adjusted PPG away team
        oa_gf_h = np.zeros(n)          # opponent-adjusted goals for
        oa_gf_a = np.zeros(n)
        oa_ga_h = np.zeros(n)          # opponent-adjusted goals against
        oa_ga_a = np.zeros(n)
        oa_strength_diff = np.zeros(n) # overall strength differential
        ppda_h = np.zeros(n)           # pressing intensity (proxy)
        ppda_a = np.zeros(n)
        style_set_piece_h = np.zeros(n)  # % goals likely from set pieces
        style_set_piece_a = np.zeros(n)
        pts_to_safety_h = np.zeros(n)  # points above relegation
        pts_to_safety_a = np.zeros(n)
        pts_to_title_h = np.zeros(n)   # points below leader
        pts_to_title_a = np.zeros(n)
        post_intl_record_h = np.zeros(n) # team-specific post-break PPG
        post_intl_record_a = np.zeros(n)

        # State tracking
        team_elo: dict[str, float] = {}         # simple Elo for opponent weighting
        team_results: dict[str, list] = {}      # (pts, gf, ga, opp_elo) per match
        team_corners: dict[str, list[int]] = {} # rolling corners (pressing proxy)
        team_fouls: dict[str, list[int]] = {}   # rolling fouls committed
        team_shots: dict[str, list[int]] = {}   # rolling shots
        team_pts: dict[str, int] = {}           # cumulative points
        team_games: dict[str, int] = {}
        team_post_intl: dict[str, list[float]] = {}  # PPG in post-intl matches
        comp_teams: dict[str, dict[str, dict]] = {}  # comp -> team -> {pts, games}

        has_corners = "hc" in df.columns
        has_fouls = "hf" in df.columns
        has_shots = "hs" in df.columns

        INTL_BREAK_MONTHS = {9, 10, 11, 3}

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            comp = getattr(r, "competition", "UNK")

            for t in (h, a):
                if t not in team_elo:
                    team_elo[t] = 1500.0
                    team_results[t] = []
                    team_corners[t] = []
                    team_fouls[t] = []
                    team_shots[t] = []
                    team_pts[t] = 0
                    team_games[t] = 0
                    team_post_intl[t] = []
            if comp not in comp_teams:
                comp_teams[comp] = {}
            for t in (h, a):
                if t not in comp_teams[comp]:
                    comp_teams[comp][t] = {"pts": 0, "games": 0}

            # Compute opponent-adjusted stats
            for t, opp, feat_ppg, feat_gf, feat_ga in [
                (h, a, oa_ppg_h, oa_gf_h, oa_ga_h),
                (a, h, oa_ppg_a, oa_gf_a, oa_ga_a)
            ]:
                results = team_results[t][-20:]  # last 20 matches
                if len(results) >= 5:
                    weighted_pts = 0.0
                    weighted_gf = 0.0
                    weighted_ga = 0.0
                    total_weight = 0.0
                    for pts, gf, ga, opp_elo in results:
                        # Weight by opponent strength relative to average (1500)
                        weight = opp_elo / 1500.0
                        weighted_pts += pts * weight
                        weighted_gf += gf * weight
                        weighted_ga += ga * weight
                        total_weight += weight
                    if total_weight > 0:
                        feat_ppg[i] = weighted_pts / total_weight
                        feat_gf[i] = weighted_gf / total_weight
                        feat_ga[i] = weighted_ga / total_weight

            oa_strength_diff[i] = oa_ppg_h[i] - oa_ppg_a[i]

            # PPDA proxy: corners + fouls indicate pressing intensity
            for t, arr in [(h, ppda_h), (a, ppda_a)]:
                corners = team_corners[t][-10:]
                fouls = team_fouls[t][-10:]
                shots = team_shots[t][-10:]
                if len(corners) >= 5 and len(fouls) >= 5:
                    # Higher corners + fouls = more aggressive pressing
                    avg_c = np.mean(corners)
                    avg_f = np.mean(fouls)
                    avg_s = np.mean(shots) if shots else 10
                    arr[i] = (avg_c + avg_f) / max(avg_s, 5)

            # Set piece style proxy: corners per goal
            for t, arr in [(h, style_set_piece_h), (a, style_set_piece_a)]:
                results = team_results[t][-15:]
                corners = team_corners[t][-15:]
                if len(results) >= 5 and len(corners) >= 5:
                    total_goals = sum(gf for _, gf, _, _ in results)
                    total_corners = sum(corners)
                    if total_goals > 0:
                        # Rough estimate: ~3% of corners become goals
                        set_piece_goals_est = total_corners * 0.03
                        arr[i] = min(1.0, set_piece_goals_est / max(total_goals, 1))

            # Points to safety / title
            ct = comp_teams.get(comp, {})
            if len(ct) >= 10:
                all_pts = sorted([v["pts"] for v in ct.values()], reverse=True)
                n_teams = len(all_pts)
                rel_zone = max(0, n_teams - 3)  # bottom 3 relegated

                for t, arr_safety, arr_title in [
                    (h, pts_to_safety_h, pts_to_title_h),
                    (a, pts_to_safety_a, pts_to_title_a)
                ]:
                    if t in ct:
                        t_pts = ct[t]["pts"]
                        rel_pts = all_pts[min(rel_zone, len(all_pts) - 1)] if rel_zone < len(all_pts) else 0
                        leader_pts = all_pts[0]
                        arr_safety[i] = max(0, t_pts - rel_pts)
                        arr_title[i] = max(0, leader_pts - t_pts)

            # Post-international-break record
            try:
                month = r.utc_date.month if hasattr(r.utc_date, "month") else 0
            except Exception:
                month = 0
            if month in INTL_BREAK_MONTHS:
                for t, arr in [(h, post_intl_record_h), (a, post_intl_record_a)]:
                    records = team_post_intl[t]
                    if len(records) >= 3:
                        arr[i] = np.mean(records[-10:])

            # Probability adjustment
            diff = oa_strength_diff[i]
            if abs(diff) > 0.1:
                adj = np.clip(diff * 0.08, -0.10, 0.10)
                p_h = 0.36 + adj
                p_a = 0.36 - adj
                p_d = 0.28
                s = p_h + p_d + p_a
                if s > 0:
                    probs[i] = _norm3(p_h / s, p_d / s, p_a / s)

            min_g = min(team_games.get(h, 0), team_games.get(a, 0))
            if min_g >= 5:
                conf[i] = min(0.85, min_g * 0.012 + abs(diff) * 0.3)

            # Update state
            if _is_finished(r):
                hg, ag = int(r.home_goals), int(r.away_goals)
                h_pts = _pts(hg, ag)
                a_pts = _pts(ag, hg)

                team_results[h].append((h_pts, hg, ag, team_elo[a]))
                team_results[a].append((a_pts, ag, hg, team_elo[h]))

                # Simple Elo update for opponent weighting
                exp_h = 1.0 / (1.0 + 10.0 ** ((team_elo[a] - team_elo[h]) / 400.0))
                actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
                team_elo[h] += 20.0 * (actual_h - exp_h)
                team_elo[a] += 20.0 * ((1 - actual_h) - (1 - exp_h))

                team_pts[h] = team_pts.get(h, 0) + h_pts
                team_pts[a] = team_pts.get(a, 0) + a_pts
                team_games[h] = team_games.get(h, 0) + 1
                team_games[a] = team_games.get(a, 0) + 1

                if comp in comp_teams:
                    if h in comp_teams[comp]:
                        comp_teams[comp][h]["pts"] += h_pts
                        comp_teams[comp][h]["games"] += 1
                    if a in comp_teams[comp]:
                        comp_teams[comp][a]["pts"] += a_pts
                        comp_teams[comp][a]["games"] += 1

                if has_corners:
                    hc = getattr(r, "hc", 0)
                    ac = getattr(r, "ac", 0)
                    try:
                        team_corners[h].append(int(hc or 0))
                        team_corners[a].append(int(ac or 0))
                    except (ValueError, TypeError):
                        pass
                if has_fouls:
                    hf = getattr(r, "hf", 0)
                    af = getattr(r, "af", 0)
                    try:
                        team_fouls[h].append(int(hf or 0))
                        team_fouls[a].append(int(af or 0))
                    except (ValueError, TypeError):
                        pass
                if has_shots:
                    hs = getattr(r, "hs", 0)
                    as_ = getattr(r, "as_", 0)
                    try:
                        team_shots[h].append(int(hs or 0))
                        team_shots[a].append(int(as_ or 0))
                    except (ValueError, TypeError):
                        pass

                # Track post-international-break performance
                if month in INTL_BREAK_MONTHS:
                    team_post_intl[h].append(float(h_pts))
                    team_post_intl[a].append(float(a_pts))

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "oa_ppg_h": oa_ppg_h,
                "oa_ppg_a": oa_ppg_a,
                "oa_gf_h": oa_gf_h,
                "oa_gf_a": oa_gf_a,
                "oa_ga_h": oa_ga_h,
                "oa_ga_a": oa_ga_a,
                "oa_strength_diff": oa_strength_diff,
                "ppda_h": ppda_h,
                "ppda_a": ppda_a,
                "style_set_piece_h": style_set_piece_h,
                "style_set_piece_a": style_set_piece_a,
                "pts_to_safety_h": pts_to_safety_h,
                "pts_to_safety_a": pts_to_safety_a,
                "pts_to_title_h": pts_to_title_h,
                "pts_to_title_a": pts_to_title_a,
                "post_intl_record_h": post_intl_record_h,
                "post_intl_record_a": post_intl_record_a,
            },
        )
