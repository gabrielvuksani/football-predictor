"""LeagueTableExpert — League position, home/away tables, zone proximity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _norm3


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

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        # Track per-season-competition tables: overall + home + away
        tables: dict[str, dict[str, dict]] = {}       # overall
        home_tables: dict[str, dict[str, dict]] = {}   # home-only
        away_tables: dict[str, dict[str, dict]] = {}   # away-only
        # Track last 6 games for form table
        recent_results: dict[str, dict[str, list]] = {}  # season -> team -> [pts]

        pos_h = np.zeros(n); pos_a = np.zeros(n)
        rel_pos_h = np.zeros(n); rel_pos_a = np.zeros(n)
        ppg_h = np.zeros(n); ppg_a = np.zeros(n)
        gd_h = np.zeros(n); gd_a = np.zeros(n)
        pts_h = np.zeros(n); pts_a = np.zeros(n)
        gap_top_h = np.zeros(n); gap_top_a = np.zeros(n)
        gap_bot_h = np.zeros(n); gap_bot_a = np.zeros(n)
        # NEW: home/away table positions
        home_pos_h = np.zeros(n); away_pos_a = np.zeros(n)
        home_ppg_h = np.zeros(n); away_ppg_a = np.zeros(n)
        home_gd_h = np.zeros(n); away_gd_a = np.zeros(n)
        # NEW: zone proximity
        rel_zone_h = np.zeros(n); rel_zone_a = np.zeros(n)     # 0-1: how close to relegation
        euro_zone_h = np.zeros(n); euro_zone_a = np.zeros(n)    # 0-1: how close to European spots
        title_zone_h = np.zeros(n); title_zone_a = np.zeros(n)  # 0-1: in title contention
        # NEW: form table (last 6 position vs overall)
        form_rel_h = np.zeros(n); form_rel_a = np.zeros(n)      # form-table relative position
        form_ppg_h = np.zeros(n); form_ppg_a = np.zeros(n)
        # NEW: position mismatch (team ranked higher overall but lower in form = regression candidate)
        pos_form_gap_h = np.zeros(n); pos_form_gap_a = np.zeros(n)
        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)

        def _rank(table_dict):
            """Rank a table dict, return sorted list with positions assigned."""
            if not table_dict:
                return {}
            ranked = sorted(table_dict.values(),
                            key=lambda x: (-x["pts"], -(x["gf"] - x["ga"]), -x["gf"]))
            for idx, entry in enumerate(ranked):
                entry["pos"] = idx + 1
            return table_dict

        def _get_table(tables_dict, comp):
            if comp not in tables_dict:
                return {}
            return _rank(tables_dict[comp])

        def _season_key(r):
            """Get competition + season key for table tracking."""
            comp = getattr(r, "competition", "UNK")
            dt = r.utc_date
            try:
                yr = dt.year
                m = dt.month
                season = yr if m >= 7 else yr - 1
            except Exception:
                season = 2024
            return f"{comp}_{season}"

        def _form_table(sk, n_recent=6):
            """Build a mini-table from each team's last N results."""
            rr = recent_results.get(sk, {})
            if not rr:
                return {}
            form_tbl = {}
            for team, pts_list in rr.items():
                recent = pts_list[-n_recent:]
                if recent:
                    form_tbl[team] = {
                        "pts": sum(recent),
                        "ppg": sum(recent) / len(recent),
                    }
            # rank by total points in form window
            ranked = sorted(form_tbl.items(), key=lambda x: -x[1]["pts"])
            for idx, (team, entry) in enumerate(ranked):
                entry["pos"] = idx + 1
                entry["rel_pos"] = 1.0 - idx / max(len(ranked) - 1, 1) if len(ranked) > 1 else 0.5
            return form_tbl

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            sk = _season_key(r)

            # Overall table
            table = _get_table(tables, sk)
            n_teams = max(len(table), 1)

            h_entry = table.get(h, {"pts": 0, "gf": 0, "ga": 0, "played": 0, "pos": n_teams})
            a_entry = table.get(a, {"pts": 0, "gf": 0, "ga": 0, "played": 0, "pos": n_teams})

            pos_h[i] = h_entry.get("pos", n_teams)
            pos_a[i] = a_entry.get("pos", n_teams)
            rel_pos_h[i] = 1.0 - (pos_h[i] - 1) / max(n_teams - 1, 1) if n_teams > 1 else 0.5
            rel_pos_a[i] = 1.0 - (pos_a[i] - 1) / max(n_teams - 1, 1) if n_teams > 1 else 0.5
            ppg_h[i] = h_entry["pts"] / max(h_entry.get("played", 1), 1)
            ppg_a[i] = a_entry["pts"] / max(a_entry.get("played", 1), 1)
            gd_h[i] = h_entry["gf"] - h_entry["ga"]
            gd_a[i] = a_entry["gf"] - a_entry["ga"]
            pts_h[i] = h_entry["pts"]
            pts_a[i] = a_entry["pts"]

            # Gap to top and bottom
            all_pts = [e["pts"] for e in table.values()] if table else [0]
            max_pts = max(all_pts) if all_pts else 0
            min_pts = min(all_pts) if all_pts else 0
            gap_top_h[i] = max_pts - h_entry["pts"]
            gap_top_a[i] = max_pts - a_entry["pts"]
            gap_bot_h[i] = h_entry["pts"] - min_pts
            gap_bot_a[i] = a_entry["pts"] - min_pts

            # NEW: Home-only table for home team
            ht = _get_table(home_tables, sk)
            ht_entry = ht.get(h, {"pts": 0, "gf": 0, "ga": 0, "played": 0, "pos": n_teams})
            home_pos_h[i] = ht_entry.get("pos", n_teams)
            home_ppg_h[i] = ht_entry["pts"] / max(ht_entry.get("played", 1), 1)
            home_gd_h[i] = ht_entry["gf"] - ht_entry["ga"]

            # NEW: Away-only table for away team
            at = _get_table(away_tables, sk)
            at_entry = at.get(a, {"pts": 0, "gf": 0, "ga": 0, "played": 0, "pos": n_teams})
            away_pos_a[i] = at_entry.get("pos", n_teams)
            away_ppg_a[i] = at_entry["pts"] / max(at_entry.get("played", 1), 1)
            away_gd_a[i] = at_entry["gf"] - at_entry["ga"]

            # NEW: Zone proximity
            all_pts_sorted = sorted(all_pts, reverse=True) if all_pts else [0]
            # relegation zone: bottom 3 in 20-team, bottom 2 in 18-team
            rel_boundary = 3 if n_teams >= 20 else 2
            # pts at relegation boundary position
            rel_pos_idx = max(0, n_teams - rel_boundary - 1)
            rel_boundary_pts = all_pts_sorted[min(rel_pos_idx, len(all_pts_sorted) - 1)]
            # European zone: position 4-6
            euro_pos_idx = min(max(3, n_teams // 5), len(all_pts_sorted) - 1)
            euro_boundary_pts = all_pts_sorted[euro_pos_idx]

            # Relegation proximity (1 = in zone, 0.5 = near, 0 = safe)
            h_gap_rel = h_entry["pts"] - rel_boundary_pts
            a_gap_rel = a_entry["pts"] - rel_boundary_pts
            rel_zone_h[i] = max(0.0, min(1.0, 1.0 - h_gap_rel / 10.0))  # within 10 pts = some signal
            rel_zone_a[i] = max(0.0, min(1.0, 1.0 - a_gap_rel / 10.0))

            # European zone proximity (1 = in zone, 0 = far)
            h_gap_euro = euro_boundary_pts - h_entry["pts"]
            a_gap_euro = euro_boundary_pts - a_entry["pts"]
            euro_zone_h[i] = max(0.0, min(1.0, 1.0 - abs(h_gap_euro) / 8.0))
            euro_zone_a[i] = max(0.0, min(1.0, 1.0 - abs(a_gap_euro) / 8.0))

            # Title zone (1 = title contender)
            title_zone_h[i] = max(0.0, min(1.0, 1.0 - gap_top_h[i] / 10.0))
            title_zone_a[i] = max(0.0, min(1.0, 1.0 - gap_top_a[i] / 10.0))

            # NEW: Form table (last 6 results)
            ft = _form_table(sk)
            ft_h = ft.get(h, {})
            ft_a = ft.get(a, {})
            form_rel_h[i] = ft_h.get("rel_pos", 0.5)
            form_rel_a[i] = ft_a.get("rel_pos", 0.5)
            form_ppg_h[i] = ft_h.get("ppg", 0.0)
            form_ppg_a[i] = ft_a.get("ppg", 0.0)

            # Position-form gap: positive = team higher overall than recent form suggests (regression risk)
            pos_form_gap_h[i] = rel_pos_h[i] - form_rel_h[i]
            pos_form_gap_a[i] = rel_pos_a[i] - form_rel_a[i]

            # Position-based probability — use VENUE-ADJUSTED position
            # Blend overall position with home/away position for more accurate picture
            adj_rel_pos_h = 0.6 * rel_pos_h[i] + 0.4 * (1.0 - (home_pos_h[i] - 1) / max(n_teams - 1, 1) if n_teams > 1 else 0.5)
            adj_rel_pos_a = 0.6 * rel_pos_a[i] + 0.4 * (1.0 - (away_pos_a[i] - 1) / max(n_teams - 1, 1) if n_teams > 1 else 0.5)
            pos_diff = adj_rel_pos_h - adj_rel_pos_a
            p_h = 0.44 + pos_diff * 0.3
            p_a = 0.28 - pos_diff * 0.2
            p_d = 1.0 - p_h - p_a
            probs[i] = _norm3(max(0.05, p_h), max(0.10, p_d), max(0.05, p_a))

            total_played = h_entry.get("played", 0) + a_entry.get("played", 0)
            conf[i] = min(1.0, total_played / 20.0)

            # --- Update tables ---
            hg, ag = int(r.home_goals), int(r.away_goals)
            for tbl_dict in [tables]:
                if sk not in tbl_dict:
                    tbl_dict[sk] = {}
                for team, gf, ga in [(h, hg, ag), (a, ag, hg)]:
                    if team not in tbl_dict[sk]:
                        tbl_dict[sk][team] = {"pts": 0, "gf": 0, "ga": 0, "played": 0, "pos": 1}
                    e = tbl_dict[sk][team]
                    e["gf"] += gf; e["ga"] += ga; e["played"] += 1
                    e["pts"] += 3 if gf > ga else (1 if gf == ga else 0)
            # Home-only table: only update home team
            if sk not in home_tables:
                home_tables[sk] = {}
            if h not in home_tables[sk]:
                home_tables[sk][h] = {"pts": 0, "gf": 0, "ga": 0, "played": 0, "pos": 1}
            he = home_tables[sk][h]
            he["gf"] += hg; he["ga"] += ag; he["played"] += 1
            he["pts"] += 3 if hg > ag else (1 if hg == ag else 0)
            # Away-only table: only update away team
            if sk not in away_tables:
                away_tables[sk] = {}
            if a not in away_tables[sk]:
                away_tables[sk][a] = {"pts": 0, "gf": 0, "ga": 0, "played": 0, "pos": 1}
            ae = away_tables[sk][a]
            ae["gf"] += ag; ae["ga"] += hg; ae["played"] += 1
            ae["pts"] += 3 if ag > hg else (1 if ag == hg else 0)
            # Recent results for form table
            if sk not in recent_results:
                recent_results[sk] = {}
            h_pts_val = 3 if hg > ag else (1 if hg == ag else 0)
            a_pts_val = 3 if ag > hg else (1 if ag == hg else 0)
            recent_results[sk].setdefault(h, []).append(h_pts_val)
            recent_results[sk].setdefault(a, []).append(a_pts_val)

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
                # NEW: home/away table positions
                "lt_home_pos_h": home_pos_h, "lt_away_pos_a": away_pos_a,
                "lt_home_ppg_h": home_ppg_h, "lt_away_ppg_a": away_ppg_a,
                "lt_home_gd_h": home_gd_h, "lt_away_gd_a": away_gd_a,
                # NEW: zone proximity
                "lt_rel_zone_h": rel_zone_h, "lt_rel_zone_a": rel_zone_a,
                "lt_euro_zone_h": euro_zone_h, "lt_euro_zone_a": euro_zone_a,
                "lt_title_zone_h": title_zone_h, "lt_title_zone_a": title_zone_a,
                # NEW: form table
                "lt_form_rel_h": form_rel_h, "lt_form_rel_a": form_rel_a,
                "lt_form_ppg_h": form_ppg_h, "lt_form_ppg_a": form_ppg_a,
                "lt_pos_form_gap_h": pos_form_gap_h, "lt_pos_form_gap_a": pos_form_gap_a,
            },
        )
