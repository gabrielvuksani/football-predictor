"""LeagueTableExpert — League position, home/away tables, zone proximity.

Uses shared LeagueTableTracker to avoid duplicate table maintenance
across Context, LeagueTable, and Motivation experts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3
from footy.models.experts._league_table_tracker import LeagueTableTracker


class LeagueTableExpert(Expert):
    """
    Simulates the live league table and derives features:
    - Overall league position + PPG + GD
    - **Home-only and away-only table** positions (critical for upset prediction)
    - Points gap to leader & relegation zone
    - **Relegation zone proximity** flag (within 3 pts of drop zone)
    - **European zone proximity** flag (within 3 pts of UEFA spots)
    - **Form table** (last 6 games position relative to overall)
    - Combined table quality index
    """
    name = "league_table"

    def __init__(self, tracker: LeagueTableTracker | None = None):
        self._tracker = tracker

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        tracker = self._tracker or LeagueTableTracker()

        pos_h = np.zeros(n); pos_a = np.zeros(n)
        rel_pos_h = np.zeros(n); rel_pos_a = np.zeros(n)
        ppg_h = np.zeros(n); ppg_a = np.zeros(n)
        gd_h = np.zeros(n); gd_a = np.zeros(n)
        pts_h = np.zeros(n); pts_a = np.zeros(n)
        gap_top_h = np.zeros(n); gap_top_a = np.zeros(n)
        gap_bot_h = np.zeros(n); gap_bot_a = np.zeros(n)
        home_pos_h = np.zeros(n); away_pos_a = np.zeros(n)
        home_ppg_h = np.zeros(n); away_ppg_a = np.zeros(n)
        home_gd_h = np.zeros(n); away_gd_a = np.zeros(n)
        rel_zone_h = np.zeros(n); rel_zone_a = np.zeros(n)
        euro_zone_h = np.zeros(n); euro_zone_a = np.zeros(n)
        title_zone_h = np.zeros(n); title_zone_a = np.zeros(n)
        form_rel_h = np.zeros(n); form_rel_a = np.zeros(n)
        form_ppg_h = np.zeros(n); form_ppg_a = np.zeros(n)
        pos_form_gap_h = np.zeros(n); pos_form_gap_a = np.zeros(n)
        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            sk = tracker.season_key(r)

            tracker.get_table(sk)
            n_teams = tracker.n_teams(sk)

            h_entry = tracker.get_entry(sk, h)
            a_entry = tracker.get_entry(sk, a)

            pos_h[i] = h_entry.pos
            pos_a[i] = a_entry.pos
            rel_pos_h[i] = 1.0 - (pos_h[i] - 1) / max(n_teams - 1, 1) if n_teams > 1 else 0.5
            rel_pos_a[i] = 1.0 - (pos_a[i] - 1) / max(n_teams - 1, 1) if n_teams > 1 else 0.5
            ppg_h[i] = h_entry.pts / max(h_entry.played, 1)
            ppg_a[i] = a_entry.pts / max(a_entry.played, 1)
            gd_h[i] = h_entry.gf - h_entry.ga
            gd_a[i] = a_entry.gf - a_entry.ga
            pts_h[i] = h_entry.pts
            pts_a[i] = a_entry.pts

            all_pts = tracker.all_pts_sorted(sk)
            max_pts = all_pts[0] if all_pts else 0
            min_pts = all_pts[-1] if all_pts else 0
            gap_top_h[i] = max_pts - h_entry.pts
            gap_top_a[i] = max_pts - a_entry.pts
            gap_bot_h[i] = h_entry.pts - min_pts
            gap_bot_a[i] = a_entry.pts - min_pts

            # Home-only table for home team
            ht = tracker.get_home_table(sk)
            if h in ht:
                ht_e = ht[h]
                home_pos_h[i] = ht_e.pos
                home_ppg_h[i] = ht_e.pts / max(ht_e.played, 1)
                home_gd_h[i] = ht_e.gf - ht_e.ga
            else:
                home_pos_h[i] = n_teams
                home_ppg_h[i] = 0.0
                home_gd_h[i] = 0.0

            # Away-only table for away team
            at = tracker.get_away_table(sk)
            if a in at:
                at_e = at[a]
                away_pos_a[i] = at_e.pos
                away_ppg_a[i] = at_e.pts / max(at_e.played, 1)
                away_gd_a[i] = at_e.gf - at_e.ga
            else:
                away_pos_a[i] = n_teams
                away_ppg_a[i] = 0.0
                away_gd_a[i] = 0.0

            # Zone proximity
            rel_boundary_pts = tracker.relegation_boundary_pts(sk)
            euro_boundary_pts = tracker.euro_boundary_pts(sk)

            h_gap_rel = h_entry.pts - rel_boundary_pts
            a_gap_rel = a_entry.pts - rel_boundary_pts
            rel_zone_h[i] = max(0.0, min(1.0, 1.0 - h_gap_rel / 10.0))
            rel_zone_a[i] = max(0.0, min(1.0, 1.0 - a_gap_rel / 10.0))

            h_gap_euro = euro_boundary_pts - h_entry.pts
            a_gap_euro = euro_boundary_pts - a_entry.pts
            euro_zone_h[i] = max(0.0, min(1.0, 1.0 - abs(h_gap_euro) / 8.0))
            euro_zone_a[i] = max(0.0, min(1.0, 1.0 - abs(a_gap_euro) / 8.0))

            title_zone_h[i] = max(0.0, min(1.0, 1.0 - gap_top_h[i] / 10.0))
            title_zone_a[i] = max(0.0, min(1.0, 1.0 - gap_top_a[i] / 10.0))

            # Form table
            ft = tracker.form_table(sk)
            ft_h = ft.get(h, {})
            ft_a = ft.get(a, {})
            form_rel_h[i] = ft_h.get("rel_pos", 0.5)
            form_rel_a[i] = ft_a.get("rel_pos", 0.5)
            form_ppg_h[i] = ft_h.get("ppg", 0.0)
            form_ppg_a[i] = ft_a.get("ppg", 0.0)

            pos_form_gap_h[i] = rel_pos_h[i] - form_rel_h[i]
            pos_form_gap_a[i] = rel_pos_a[i] - form_rel_a[i]

            # Position-based probability
            adj_rel_pos_h = 0.6 * rel_pos_h[i] + 0.4 * (1.0 - (home_pos_h[i] - 1) / max(n_teams - 1, 1) if n_teams > 1 else 0.5)
            adj_rel_pos_a = 0.6 * rel_pos_a[i] + 0.4 * (1.0 - (away_pos_a[i] - 1) / max(n_teams - 1, 1) if n_teams > 1 else 0.5)
            pos_diff = adj_rel_pos_h - adj_rel_pos_a
            p_h = 0.44 + pos_diff * 0.3
            p_a = 0.28 - pos_diff * 0.2
            p_d = 1.0 - p_h - p_a
            probs[i] = _norm3(max(0.05, p_h), max(0.10, p_d), max(0.05, p_a))

            total_played = h_entry.played + a_entry.played
            conf[i] = min(1.0, total_played / 20.0)

            # Update tables for finished matches
            if _is_finished(r):
                hg, ag = int(r.home_goals), int(r.away_goals)
                tracker.update(sk, h, a, hg, ag)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "lt_pos_h": pos_h, "lt_pos_a": pos_a,
                "lt_pos_diff": pos_h - pos_a,
                "lt_rel_pos_h": rel_pos_h, "lt_rel_pos_a": rel_pos_a,
                "lt_ppg_h": ppg_h, "lt_ppg_a": ppg_a,
                "lt_ppg_diff": ppg_h - ppg_a,
                "lt_gd_h": gd_h, "lt_gd_a": gd_a,
                "lt_gd_diff": gd_h - gd_a,
                "lt_pts_h": pts_h, "lt_pts_a": pts_a,
                "lt_gap_top_h": gap_top_h, "lt_gap_top_a": gap_top_a,
                "lt_gap_bot_h": gap_bot_h, "lt_gap_bot_a": gap_bot_a,
                "lt_home_pos_h": home_pos_h, "lt_away_pos_a": away_pos_a,
                "lt_home_ppg_h": home_ppg_h, "lt_away_ppg_a": away_ppg_a,
                "lt_home_gd_h": home_gd_h, "lt_away_gd_a": away_gd_a,
                "lt_rel_zone_h": rel_zone_h, "lt_rel_zone_a": rel_zone_a,
                "lt_euro_zone_h": euro_zone_h, "lt_euro_zone_a": euro_zone_a,
                "lt_title_zone_h": title_zone_h, "lt_title_zone_a": title_zone_a,
                "lt_form_rel_h": form_rel_h, "lt_form_rel_a": form_rel_a,
                "lt_form_ppg_h": form_ppg_h, "lt_form_ppg_a": form_ppg_a,
                "lt_pos_form_gap_h": pos_form_gap_h, "lt_pos_form_gap_a": pos_form_gap_a,
            },
        )
