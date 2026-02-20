"""ContextExpert — Rest / congestion / motivation / calendar / season stage."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _norm3, _pts


class ContextExpert(Expert):
    """
    Contextual factors that influence match outcomes beyond team quality:
    - Rest days and fixture congestion (7/14/30 day windows)
    - Fatigue interactions (short rest × congestion)
    - Season calendar (early/late, midweek, weekend)
    - **Motivation modeling**: League position context — relegation battle,
      title race, European push, mid-table "nothing to play for"
    - **Congestion fatigue index**: Combined rest + recent games penalty
    - **Midweek European proxy**: Short rest + midweek flag detection
    """
    name = "context"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        last_date: dict[str, Any] = {}
        match_dates: dict[str, list] = {}   # for congestion calc
        # Track league table for motivation features
        tables: dict[str, dict[str, dict]] = {}  # season_comp -> team -> stats
        results_history: dict[str, list[float]] = {}  # team -> recent pts
        # Track fixture frequency for rivalry detection
        fixture_count: dict[frozenset, int] = {}  # how many times these two teams have met
        fixture_dates: dict[frozenset, list] = {}  # dates of each meeting

        rest_h = np.zeros(n); rest_a = np.zeros(n)
        cong7_h = np.zeros(n); cong7_a = np.zeros(n)
        cong14_h = np.zeros(n); cong14_a = np.zeros(n)
        cong30_h = np.zeros(n); cong30_a = np.zeros(n)
        # season calendar features
        season_prog = np.zeros(n)       # 0-1 season progress (Aug=0, May=1)
        is_early = np.zeros(n)          # first 6 GWs proxy
        is_late = np.zeros(n)           # final 6 GWs proxy
        dow = np.zeros(n)               # day of week (0=Mon, 6=Sun)
        is_weekend = np.zeros(n)        # Sat/Sun
        is_midweek = np.zeros(n)        # Tue/Wed (European nights)
        hour_utc = np.zeros(n)          # hour of kickoff
        # fatigue interactions
        rest_ratio = np.zeros(n)        # home/away rest ratio
        short_rest_h = np.zeros(n)      # home on <3 days rest
        short_rest_a = np.zeros(n)      # away on <3 days rest
        # NEW: fatigue index (combined rest + congestion penalty)
        fatigue_idx_h = np.zeros(n)
        fatigue_idx_a = np.zeros(n)
        # NEW: midweek turnaround flag (played midweek, now playing weekend = tired)
        midweek_turn_h = np.zeros(n)
        midweek_turn_a = np.zeros(n)
        # NEW: motivation features
        motivation_h = np.zeros(n)      # -1 to +1 motivation score
        motivation_a = np.zeros(n)
        motivation_diff = np.zeros(n)   # + = home more motivated
        is_relegation_h = np.zeros(n)   # in/near relegation zone
        is_relegation_a = np.zeros(n)
        is_title_race_h = np.zeros(n)   # title contender
        is_title_race_a = np.zeros(n)
        is_european_push_h = np.zeros(n)   # fighting for European spots
        is_european_push_a = np.zeros(n)
        is_safe_h = np.zeros(n)          # mid-table, nothing to play for
        is_safe_a = np.zeros(n)
        # NEW: derby/rivalry detection
        is_derby = np.zeros(n)           # 0-1: fixture intensity score
        # NEW: high-stakes composite
        high_stakes = np.zeros(n)        # 0-1: combined urgency of the match

        probs = np.full((n, 3), [0.44, 0.28, 0.28])  # mild home prior
        conf = np.full(n, 0.3)  # context is informative but never decisive

        def _season_key(r):
            comp = getattr(r, "competition", "UNK")
            dt = r.utc_date
            try:
                yr = dt.year
                m = dt.month
                season = yr if m >= 7 else yr - 1
            except Exception:
                season = 2024
            return f"{comp}_{season}"

        def _get_motivation(team, sk, season_progress):
            """Calculate motivation score [-1, +1] from league position context.

            +1 = maximum motivation (title race, relegation battle)
            -1 = no motivation (safe mid-table, nothing to play for)
            0  = neutral
            """
            table = tables.get(sk, {})
            entry = table.get(team)
            if not entry or entry["played"] < 5:
                return 0.0, {}

            n_teams = max(len(table), 1)
            pos = entry.get("pos", n_teams // 2)
            pts = entry["pts"]

            ranked = sorted(table.values(),
                            key=lambda x: (-x["pts"], -(x["gf"] - x["ga"])))
            all_pts = [e["pts"] for e in ranked]
            if not all_pts:
                return 0.0, {}
            max_pts = all_pts[0]
            # relegation zone: bottom 3 in 20-team, bottom 2 in 18-team
            rel_zone = 3 if n_teams >= 20 else 2
            rel_boundary_pos = n_teams - rel_zone
            rel_pts = all_pts[rel_boundary_pos] if len(all_pts) > rel_boundary_pos else 0
            # European spots: roughly positions 4-7 in 20-team league
            euro_boundary = max(4, n_teams // 5)
            euro_pts = all_pts[min(euro_boundary, len(all_pts) - 1)] if all_pts else 0

            flags = {}
            motivation = 0.0

            # Title race: within 6 points of top AND in top 3
            if pts >= max_pts - 6 and pos <= 3:
                flags["title_race"] = 1.0
                motivation += 0.5

            # Relegation battle: within 4 points of relegation zone boundary
            if pts <= rel_pts + 4 and pos > n_teams - rel_zone - 3:
                flags["relegation"] = 1.0
                motivation += 0.6  # relegation fear is strongest motivator

            # European push: positions 4-7, within 4 pts of European spots
            if euro_boundary <= pos <= euro_boundary + 4 and pts >= euro_pts - 4:
                flags["european_push"] = 1.0
                motivation += 0.3

            # Safe mid-table: not in any race, no threat, boring season
            if not flags:
                # Check if gap to relegation is large AND gap to European is large
                gap_to_rel = pts - rel_pts
                gap_to_euro = euro_pts - pts
                if gap_to_rel > 10 and gap_to_euro > 10:
                    flags["safe"] = 1.0
                    motivation -= 0.4  # demotivated
                elif gap_to_rel > 6:
                    motivation -= 0.1  # somewhat comfortable

            # Late-season amplification: motivation effects stronger as season ends
            if season_progress > 0.7:
                motivation *= (1.0 + (season_progress - 0.7) * 1.5)

            return min(1.0, max(-1.0, motivation)), flags

        def _update_table(sk, team, gf, ga):
            if sk not in tables:
                tables[sk] = {}
            if team not in tables[sk]:
                tables[sk][team] = {"pts": 0, "gf": 0, "ga": 0, "played": 0, "pos": 1}
            e = tables[sk][team]
            e["pts"] += _pts(gf, ga)
            e["gf"] += gf
            e["ga"] += ga
            e["played"] += 1
            # Re-rank
            ranked = sorted(tables[sk].values(),
                            key=lambda x: (-x["pts"], -(x["gf"] - x["ga"]), -x["gf"]))
            for idx, entry in enumerate(ranked):
                entry["pos"] = idx + 1

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            dt = r.utc_date
            sk = _season_key(r)

            # rest days
            lh = last_date.get(h)
            la = last_date.get(a)
            rh = 7.0 if lh is None else max(0.0, (dt - lh).total_seconds() / 86400.0)
            ra = 7.0 if la is None else max(0.0, (dt - la).total_seconds() / 86400.0)
            rest_h[i] = rh; rest_a[i] = ra
            rest_ratio[i] = rh / max(ra, 0.5)
            short_rest_h[i] = 1.0 if rh < 3.0 else 0.0
            short_rest_a[i] = 1.0 if ra < 3.0 else 0.0

            # congestion: how many games in last 7/14/30 days
            for team, arr7, arr14, arr30, idx in [
                (h, cong7_h, cong14_h, cong30_h, i),
                (a, cong7_a, cong14_a, cong30_a, i),
            ]:
                dates = match_dates.get(team, [])
                c7 = sum(1 for d in dates if (dt - d).total_seconds() / 86400 <= 7)
                c14 = sum(1 for d in dates if (dt - d).total_seconds() / 86400 <= 14)
                c30 = sum(1 for d in dates if (dt - d).total_seconds() / 86400 <= 30)
                arr7[idx] = c7; arr14[idx] = c14; arr30[idx] = c30

            # NEW: fatigue index = f(rest_days, congestion_7, congestion_14)
            # Lower rest + higher congestion = higher fatigue
            fatigue_idx_h[i] = max(0.0,
                (3.0 - rh) * 0.3 +     # short rest penalty
                cong7_h[i] * 0.25 +     # recent game load
                cong14_h[i] * 0.1       # medium-term load
            )
            fatigue_idx_a[i] = max(0.0,
                (3.0 - ra) * 0.3 +
                cong7_a[i] * 0.25 +
                cong14_a[i] * 0.1
            )

            # season calendar
            try:
                month = dt.month
                # season runs Aug(8) → May(5)
                if month >= 8:
                    prog = (month - 8) / 10.0  # Aug=0, Dec=0.4
                else:
                    prog = (month + 4) / 10.0  # Jan=0.5, May=0.9
                season_prog[i] = min(1.0, prog)
                is_early[i] = 1.0 if (month in (8, 9)) else 0.0
                is_late[i] = 1.0 if (month in (4, 5)) else 0.0
                d = dt.weekday()  # 0=Monday
                dow[i] = d
                is_weekend[i] = 1.0 if d >= 5 else 0.0  # Sat/Sun
                is_midweek[i] = 1.0 if d in (1, 2) else 0.0  # Tue/Wed
                hour_utc[i] = dt.hour
            except Exception:
                pass

            # NEW: midweek turnaround detection
            # If last match was Tue/Wed (d=1,2) and this is Sat/Sun, short turnaround
            if lh is not None and rh <= 4.0:
                try:
                    last_dow_h = lh.weekday()
                    if last_dow_h in (1, 2) and dt.weekday() >= 5:
                        midweek_turn_h[i] = 1.0
                except Exception:
                    pass
            if la is not None and ra <= 4.0:
                try:
                    last_dow_a = la.weekday()
                    if last_dow_a in (1, 2) and dt.weekday() >= 5:
                        midweek_turn_a[i] = 1.0
                except Exception:
                    pass

            # NEW: motivation modeling from league position
            sp = season_prog[i]
            mot_h, flags_h = _get_motivation(h, sk, sp)
            mot_a, flags_a = _get_motivation(a, sk, sp)
            motivation_h[i] = mot_h
            motivation_a[i] = mot_a
            motivation_diff[i] = mot_h - mot_a
            is_relegation_h[i] = flags_h.get("relegation", 0.0)
            is_relegation_a[i] = flags_a.get("relegation", 0.0)
            is_title_race_h[i] = flags_h.get("title_race", 0.0)
            is_title_race_a[i] = flags_a.get("title_race", 0.0)
            is_european_push_h[i] = flags_h.get("european_push", 0.0)
            is_european_push_a[i] = flags_a.get("european_push", 0.0)
            is_safe_h[i] = flags_h.get("safe", 0.0)
            is_safe_a[i] = flags_a.get("safe", 0.0)

            # NEW: derby/rivalry detection — data-driven from fixture frequency
            fix_key = frozenset([h, a])
            n_meetings = fixture_count.get(fix_key, 0)
            # Derby intensity: high meeting frequency + close league positions = rivalry
            if n_meetings >= 4:
                # Frequency factor: teams meeting 4+ times are likely same league
                freq_factor = min(1.0, n_meetings / 12.0)
                # Position proximity factor: closer positions = more intense derby
                pos_h_ctx = tables.get(sk, {}).get(h, {}).get("pos", 10)
                pos_a_ctx = tables.get(sk, {}).get(a, {}).get("pos", 10)
                pos_gap = abs(pos_h_ctx - pos_a_ctx)
                proximity_factor = max(0.0, 1.0 - pos_gap / 10.0)
                is_derby[i] = freq_factor * 0.6 + proximity_factor * 0.4
            elif n_meetings >= 2:
                is_derby[i] = min(0.3, n_meetings / 10.0)

            # NEW: high-stakes composite — captures overall match intensity
            stakes = max(abs(mot_h), abs(mot_a))  # highest motivation of either team
            if is_late[i]:
                stakes *= 1.3  # late-season amplifier
            if is_derby[i] > 0.3:
                stakes = max(stakes, 0.4)  # derbies always somewhat high-stakes
            high_stakes[i] = min(1.0, stakes)

            # context-adjusted probs — enhanced
            rest_adv = (rh - ra) * 0.01
            season_adj = 0.02 if is_late[i] else 0.0  # home advantage stronger late season
            fatigue_diff = (fatigue_idx_a[i] - fatigue_idx_h[i]) * 0.015  # tired away = home benefit
            motivation_adj = motivation_diff[i] * 0.03   # motivation differential
            # Derby effect: reduces home advantage (away team more fired up in derbies)
            derby_adj = -is_derby[i] * 0.015 if is_derby[i] > 0.3 else 0.0
            total_adj = rest_adv + season_adj + fatigue_diff + motivation_adj + derby_adj
            probs[i] = _norm3(
                0.44 + total_adj,
                0.28,
                0.28 - total_adj,
            )

            # adjust confidence based on available signals
            base_conf = 0.3
            if short_rest_h[i] or short_rest_a[i]:
                base_conf += 0.15
            if abs(motivation_diff[i]) > 0.3:
                base_conf += 0.1   # strong motivation asymmetry is informative
            if midweek_turn_h[i] or midweek_turn_a[i]:
                base_conf += 0.05
            if is_derby[i] > 0.3:
                base_conf += 0.05  # derbies are informative context
            if high_stakes[i] > 0.5:
                base_conf += 0.05
            conf[i] = min(0.7, base_conf)

            # --- update ---
            hg, ag = int(r.home_goals), int(r.away_goals)
            last_date[h] = dt; last_date[a] = dt
            match_dates.setdefault(h, []).append(dt)
            match_dates.setdefault(a, []).append(dt)
            _update_table(sk, h, hg, ag)
            _update_table(sk, a, ag, hg)
            # Track fixture frequency for rivalry detection
            fixture_count[fix_key] = fixture_count.get(fix_key, 0) + 1
            fixture_dates.setdefault(fix_key, []).append(dt)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "ctx_rest_h": rest_h, "ctx_rest_a": rest_a,
                "ctx_rest_diff": rest_h - rest_a,
                "ctx_rest_ratio": rest_ratio,
                "ctx_short_rest_h": short_rest_h, "ctx_short_rest_a": short_rest_a,
                "ctx_cong7_h": cong7_h, "ctx_cong7_a": cong7_a,
                "ctx_cong14_h": cong14_h, "ctx_cong14_a": cong14_a,
                "ctx_cong30_h": cong30_h, "ctx_cong30_a": cong30_a,
                "ctx_cong7_diff": cong7_h - cong7_a,
                "ctx_cong14_diff": cong14_h - cong14_a,
                "ctx_fatigue_h": fatigue_idx_h, "ctx_fatigue_a": fatigue_idx_a,
                "ctx_fatigue_diff": fatigue_idx_h - fatigue_idx_a,
                "ctx_midweek_turn_h": midweek_turn_h, "ctx_midweek_turn_a": midweek_turn_a,
                "ctx_season_prog": season_prog,
                "ctx_is_early": is_early, "ctx_is_late": is_late,
                "ctx_dow": dow, "ctx_is_weekend": is_weekend,
                "ctx_is_midweek": is_midweek,
                "ctx_hour_utc": hour_utc,
                "ctx_motivation_h": motivation_h, "ctx_motivation_a": motivation_a,
                "ctx_motivation_diff": motivation_diff,
                "ctx_is_relegation_h": is_relegation_h, "ctx_is_relegation_a": is_relegation_a,
                "ctx_is_title_race_h": is_title_race_h, "ctx_is_title_race_a": is_title_race_a,
                "ctx_is_euro_push_h": is_european_push_h, "ctx_is_euro_push_a": is_european_push_a,
                "ctx_is_safe_h": is_safe_h, "ctx_is_safe_a": is_safe_a,
                "ctx_is_derby": is_derby,
                "ctx_high_stakes": high_stakes,
            },
        )
