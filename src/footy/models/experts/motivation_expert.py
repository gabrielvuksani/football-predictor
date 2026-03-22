"""MotivationExpert — Match importance and team motivation from league context.

Uses shared LeagueTableTracker for consistent league position data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3, _pts
from footy.models.experts._league_table_tracker import LeagueTableTracker


# Objective encoding constants
OBJ_RELEGATION = 0
OBJ_MIDTABLE = 1
OBJ_EUROPA = 2
OBJ_UCL = 3
OBJ_TITLE = 4


class MotivationExpert(Expert):
    """
    Analyses match importance and team motivation based on league position,
    seasonal objectives, and context.

    Uses the shared LeagueTableTracker instead of maintaining its own table.
    """

    name = "motivation"

    RECENT_WINDOW = 5
    GAP_THRESHOLD = 6
    TYPICAL_SEASON_GAMES = 38

    def __init__(self, tracker: LeagueTableTracker | None = None):
        self._tracker = tracker

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        tracker = self._tracker or LeagueTableTracker()

        # Recent results per team for desperation tracking
        recent_results: dict[str, list[int]] = {}
        # Games played per team per season_key
        games_played: dict[str, dict[str, int]] = {}

        mot_home_obj = np.full(n, OBJ_MIDTABLE, dtype=float)
        mot_away_obj = np.full(n, OBJ_MIDTABLE, dtype=float)
        mot_home_mot = np.zeros(n)
        mot_away_mot = np.zeros(n)
        mot_diff = np.zeros(n)
        mot_season_prog = np.zeros(n)
        mot_home_pts_obj = np.zeros(n)
        mot_away_pts_obj = np.zeros(n)
        mot_both_high = np.zeros(n)
        mot_pressure_diff = np.zeros(n)

        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)

        def _classify_objective(team: str, sk: str) -> tuple[int, float, float]:
            entry = tracker.get_entry(sk, team)
            n_teams = tracker.n_teams(sk)
            if n_teams < 4 or entry.played < 3:
                return OBJ_MIDTABLE, 0.0, 0.3

            pos = entry.pos
            pts = entry.pts

            all_pts = tracker.all_pts_sorted(sk)
            leader_pts = all_pts[0] if all_pts else 0

            ucl_idx = min(3, n_teams - 1)
            ucl_boundary_pts = all_pts[ucl_idx] if len(all_pts) > ucl_idx else 0

            europa_idx = min(6, n_teams - 1)
            europa_boundary_pts = all_pts[europa_idx] if len(all_pts) > europa_idx else 0

            rel_count = 3 if n_teams >= 20 else 2
            safety_idx = max(0, n_teams - rel_count - 1)
            safety_pts = all_pts[safety_idx] if len(all_pts) > safety_idx else 0

            gap = self.GAP_THRESHOLD

            if pos <= 2 and (leader_pts - pts) <= gap:
                pts_gap = leader_pts - pts
                pressure = 0.9 - 0.05 * pts_gap if pts_gap > 0 else 0.95
                return OBJ_TITLE, pts_gap, min(1.0, pressure)

            if pos <= 6 and (ucl_boundary_pts - pts) <= gap and pos > 2:
                pts_gap = max(0.0, ucl_boundary_pts - pts) if pos > ucl_idx + 1 else 0.0
                pressure = 0.7 + 0.03 * max(0, gap - pts_gap)
                return OBJ_UCL, pts_gap, min(1.0, pressure)

            if pos > n_teams - rel_count - 4 and (pts - safety_pts) <= gap:
                pts_gap = safety_pts - pts
                pressure = 0.85 + 0.05 * max(0, -pts_gap)
                return OBJ_RELEGATION, abs(pts_gap), min(1.0, pressure)

            if pos <= europa_idx + 3 and abs(europa_boundary_pts - pts) <= gap:
                pts_gap = max(0.0, europa_boundary_pts - pts)
                pressure = 0.5 + 0.03 * max(0, gap - pts_gap)
                return OBJ_EUROPA, pts_gap, min(1.0, pressure)

            return OBJ_MIDTABLE, 0.0, 0.25

        def _motivation_score(objective, pressure, season_progress, pts_gap, team):
            obj_base = {
                OBJ_TITLE: 0.85, OBJ_UCL: 0.70, OBJ_EUROPA: 0.55,
                OBJ_MIDTABLE: 0.35, OBJ_RELEGATION: 0.80,
            }
            base = obj_base.get(objective, 0.4)
            progress_mult = 0.6 + 0.8 * season_progress
            gap_factor = max(0.5, 1.0 - pts_gap / 12.0)

            desperation = 0.0
            recent = recent_results.get(team, [])
            if len(recent) >= 3:
                avg_pts = sum(recent[-3:]) / 3.0
                if avg_pts < 1.0:
                    if objective in (OBJ_RELEGATION, OBJ_TITLE, OBJ_UCL):
                        desperation = 0.10
                    else:
                        desperation = -0.05

            raw = base * progress_mult * gap_factor + desperation
            return float(np.clip(raw, 0.0, 1.0))

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            sk = tracker.season_key(r)

            # Season progress
            gp = games_played.get(sk, {})
            h_played = gp.get(h, 0)
            a_played = gp.get(a, 0)
            avg_played = (h_played + a_played) / 2.0 if (h_played + a_played) > 0 else 0.0
            max_played = max(gp.values()) if gp else 0
            season_len = max(max_played, self.TYPICAL_SEASON_GAMES)
            s_progress = min(1.0, avg_played / season_len)
            mot_season_prog[i] = s_progress

            h_obj, h_pts_gap, h_pressure = _classify_objective(h, sk)
            a_obj, a_pts_gap, a_pressure = _classify_objective(a, sk)

            mot_home_obj[i] = float(h_obj)
            mot_away_obj[i] = float(a_obj)
            mot_home_pts_obj[i] = h_pts_gap
            mot_away_pts_obj[i] = a_pts_gap

            h_mot = _motivation_score(h_obj, h_pressure, s_progress, h_pts_gap, h)
            a_mot = _motivation_score(a_obj, a_pressure, s_progress, a_pts_gap, a)

            mot_home_mot[i] = h_mot
            mot_away_mot[i] = a_mot
            mot_diff[i] = h_mot - a_mot
            mot_both_high[i] = 1.0 if (h_mot > 0.65 and a_mot > 0.65) else 0.0
            mot_pressure_diff[i] = h_pressure - a_pressure

            # 1X2 probabilities
            m_diff = h_mot - a_mot
            p_h = 0.42
            p_d = 0.28
            p_a = 0.30
            p_h += m_diff * 0.12
            p_a -= m_diff * 0.08
            if h_obj == OBJ_MIDTABLE and a_obj == OBJ_MIDTABLE:
                p_d += 0.04; p_h -= 0.02; p_a -= 0.02
            if h_obj == OBJ_RELEGATION or a_obj == OBJ_RELEGATION:
                p_d += 0.03
                if h_obj == OBJ_RELEGATION and a_obj == OBJ_RELEGATION:
                    p_d += 0.03
            if mot_both_high[i] > 0.5:
                p_d += 0.02
            probs[i] = _norm3(max(0.05, p_h), max(0.10, p_d), max(0.05, p_a))

            total_played = h_played + a_played
            conf[i] = min(1.0, max(0.0, (total_played - 10) / 30.0))
            n_teams = tracker.n_teams(sk)
            if n_teams < 8:
                conf[i] *= 0.5

            # Update state for finished matches
            if not _is_finished(r):
                continue
            hg, ag = int(r.home_goals), int(r.away_goals)
            tracker.update(sk, h, a, hg, ag)

            if sk not in games_played:
                games_played[sk] = {}
            games_played[sk][h] = games_played[sk].get(h, 0) + 1
            games_played[sk][a] = games_played[sk].get(a, 0) + 1

            h_pts_val = _pts(hg, ag)
            a_pts_val = _pts(ag, hg)
            recent_results.setdefault(h, []).append(h_pts_val)
            recent_results.setdefault(a, []).append(a_pts_val)
            if len(recent_results[h]) > self.RECENT_WINDOW * 2:
                recent_results[h] = recent_results[h][-self.RECENT_WINDOW * 2:]
            if len(recent_results[a]) > self.RECENT_WINDOW * 2:
                recent_results[a] = recent_results[a][-self.RECENT_WINDOW * 2:]

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "mot_home_objective": mot_home_obj,
                "mot_away_objective": mot_away_obj,
                "mot_home_motivation": mot_home_mot,
                "mot_away_motivation": mot_away_mot,
                "mot_motivation_diff": mot_diff,
                "mot_season_progress": mot_season_prog,
                "mot_home_pts_to_obj": mot_home_pts_obj,
                "mot_away_pts_to_obj": mot_away_pts_obj,
                "mot_both_high": mot_both_high,
                "mot_pressure_diff": mot_pressure_diff,
            },
        )
