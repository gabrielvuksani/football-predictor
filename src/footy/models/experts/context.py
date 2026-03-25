"""ContextExpert — Rest / congestion / motivation / calendar / season stage.

Uses shared LeagueTableTracker for league position context, eliminating
duplicate table maintenance.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3
from footy.models.experts._league_table_tracker import LeagueTableTracker


class ContextExpert(Expert):
    """
    Contextual factors that influence match outcomes beyond team quality:
    - Rest days and fixture congestion (7/14/30 day windows)
    - Fatigue interactions (short rest x congestion)
    - Season calendar (early/late, midweek, weekend)
    - Motivation modeling from shared league table
    - Congestion fatigue index
    - Midweek European proxy
    """
    name = "context"

    def __init__(self, tracker: LeagueTableTracker | None = None):
        self._tracker = tracker

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        tracker = self._tracker or LeagueTableTracker()
        last_date: dict[str, Any] = {}
        match_dates: dict[str, list] = {}
        fixture_count: dict[frozenset, int] = {}
        fixture_dates: dict[frozenset, list] = {}

        rest_h = np.zeros(n); rest_a = np.zeros(n)
        cong7_h = np.zeros(n); cong7_a = np.zeros(n)
        cong14_h = np.zeros(n); cong14_a = np.zeros(n)
        cong30_h = np.zeros(n); cong30_a = np.zeros(n)
        season_prog = np.zeros(n)
        is_early = np.zeros(n)
        is_late = np.zeros(n)
        dow = np.zeros(n)
        is_weekend = np.zeros(n)
        is_midweek = np.zeros(n)
        hour_utc = np.zeros(n)
        rest_ratio = np.zeros(n)
        short_rest_h = np.zeros(n)
        short_rest_a = np.zeros(n)
        fatigue_idx_h = np.zeros(n)
        fatigue_idx_a = np.zeros(n)
        midweek_turn_h = np.zeros(n)
        midweek_turn_a = np.zeros(n)
        motivation_h = np.zeros(n)
        motivation_a = np.zeros(n)
        # v15: FPL Fixture Difficulty Rating (PL only, free data)
        fdr_h = np.zeros(n)
        fdr_a = np.zeros(n)
        motivation_diff = np.zeros(n)
        is_relegation_h = np.zeros(n)
        is_relegation_a = np.zeros(n)
        is_title_race_h = np.zeros(n)
        is_title_race_a = np.zeros(n)
        is_european_push_h = np.zeros(n)
        is_european_push_a = np.zeros(n)
        is_safe_h = np.zeros(n)
        is_safe_a = np.zeros(n)
        is_derby = np.zeros(n)
        high_stakes = np.zeros(n)

        probs = np.full((n, 3), [0.44, 0.28, 0.28])
        conf = np.full(n, 0.3)

        def _get_motivation(team, sk, season_progress):
            """Calculate motivation score [-1, +1] from league position context."""
            entry = tracker.get_entry(sk, team)
            if entry.played < 5:
                return 0.0, {}

            n_teams = tracker.n_teams(sk)
            pos = entry.pos
            pts = entry.pts

            leader = tracker.leader_pts(sk)
            rel_pts = tracker.relegation_boundary_pts(sk)
            euro_pts = tracker.euro_boundary_pts(sk)

            flags = {}
            mot = 0.0

            if pts >= leader - 6 and pos <= 3:
                flags["title_race"] = 1.0
                mot += 0.5

            rel_zone = 3 if n_teams >= 20 else 2
            if pts <= rel_pts + 4 and pos > n_teams - rel_zone - 3:
                flags["relegation"] = 1.0
                mot += 0.6

            euro_boundary = max(4, n_teams // 5)
            if euro_boundary <= pos <= euro_boundary + 4 and pts >= euro_pts - 4:
                flags["european_push"] = 1.0
                mot += 0.3

            if not flags:
                gap_to_rel = pts - rel_pts
                gap_to_euro = euro_pts - pts
                if gap_to_rel > 10 and gap_to_euro > 10:
                    flags["safe"] = 1.0
                    mot -= 0.4
                elif gap_to_rel > 6:
                    mot -= 0.1

            if season_progress > 0.7:
                mot *= (1.0 + (season_progress - 0.7) * 1.5)

            return min(1.0, max(-1.0, mot)), flags

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            dt = r.utc_date
            sk = tracker.season_key(r)

            # rest days
            lh = last_date.get(h)
            la = last_date.get(a)
            rh = 7.0 if lh is None else max(0.0, (dt - lh).total_seconds() / 86400.0)
            ra = 7.0 if la is None else max(0.0, (dt - la).total_seconds() / 86400.0)
            rest_h[i] = rh; rest_a[i] = ra
            rest_ratio[i] = rh / max(ra, 0.5)
            short_rest_h[i] = 1.0 if rh < 3.0 else 0.0
            short_rest_a[i] = 1.0 if ra < 3.0 else 0.0

            # v15: FPL Fixture Difficulty Rating (PL only, default 3.0 = average)
            fdr_h[i] = _f(getattr(r, "fpl_fdr_h", 3.0)) or 3.0
            fdr_a[i] = _f(getattr(r, "fpl_fdr_a", 3.0)) or 3.0

            # congestion
            for team, arr7, arr14, arr30, idx in [
                (h, cong7_h, cong14_h, cong30_h, i),
                (a, cong7_a, cong14_a, cong30_a, i),
            ]:
                dates = match_dates.get(team, [])
                c7 = sum(1 for d in dates if (dt - d).total_seconds() / 86400 <= 7)
                c14 = sum(1 for d in dates if (dt - d).total_seconds() / 86400 <= 14)
                c30 = sum(1 for d in dates if (dt - d).total_seconds() / 86400 <= 30)
                arr7[idx] = c7; arr14[idx] = c14; arr30[idx] = c30

            # fatigue index
            fatigue_idx_h[i] = max(0.0,
                (3.0 - rh) * 0.3 + cong7_h[i] * 0.25 + cong14_h[i] * 0.1
            )
            fatigue_idx_a[i] = max(0.0,
                (3.0 - ra) * 0.3 + cong7_a[i] * 0.25 + cong14_a[i] * 0.1
            )

            # season calendar
            try:
                month = dt.month
                if month >= 8:
                    prog = (month - 8) / 10.0
                else:
                    prog = (month + 4) / 10.0
                season_prog[i] = min(1.0, prog)
                is_early[i] = 1.0 if month in (8, 9) else 0.0
                is_late[i] = 1.0 if month in (4, 5) else 0.0
                d = dt.weekday()
                dow[i] = d
                is_weekend[i] = 1.0 if d >= 5 else 0.0
                is_midweek[i] = 1.0 if d in (1, 2) else 0.0
                hour_utc[i] = dt.hour
            except Exception:
                pass

            # midweek turnaround detection
            if lh is not None and rh <= 4.0:
                try:
                    if lh.weekday() in (1, 2) and dt.weekday() >= 5:
                        midweek_turn_h[i] = 1.0
                except Exception:
                    pass
            if la is not None and ra <= 4.0:
                try:
                    if la.weekday() in (1, 2) and dt.weekday() >= 5:
                        midweek_turn_a[i] = 1.0
                except Exception:
                    pass

            # motivation from shared tracker
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

            # derby/rivalry detection
            fix_key = frozenset([h, a])
            n_meetings = fixture_count.get(fix_key, 0)
            if n_meetings >= 4:
                freq_factor = min(1.0, n_meetings / 12.0)
                pos_h_ctx = tracker.get_entry(sk, h).pos
                pos_a_ctx = tracker.get_entry(sk, a).pos
                pos_gap = abs(pos_h_ctx - pos_a_ctx)
                proximity_factor = max(0.0, 1.0 - pos_gap / 10.0)
                is_derby[i] = freq_factor * 0.6 + proximity_factor * 0.4
            elif n_meetings >= 2:
                is_derby[i] = min(0.3, n_meetings / 10.0)

            # high-stakes composite
            stakes = max(abs(mot_h), abs(mot_a))
            if is_late[i]:
                stakes *= 1.3
            if is_derby[i] > 0.3:
                stakes = max(stakes, 0.4)
            high_stakes[i] = min(1.0, stakes)

            # context-adjusted probs
            rest_adv = (rh - ra) * 0.01
            season_adj = 0.02 if is_late[i] else 0.0
            fatigue_diff = (fatigue_idx_a[i] - fatigue_idx_h[i]) * 0.015
            motivation_adj = motivation_diff[i] * 0.03
            derby_adj = -is_derby[i] * 0.015 if is_derby[i] > 0.3 else 0.0
            total_adj = rest_adv + season_adj + fatigue_diff + motivation_adj + derby_adj
            probs[i] = _norm3(0.44 + total_adj, 0.28, 0.28 - total_adj)

            base_conf = 0.3
            if short_rest_h[i] or short_rest_a[i]:
                base_conf += 0.15
            if abs(motivation_diff[i]) > 0.3:
                base_conf += 0.1
            if midweek_turn_h[i] or midweek_turn_a[i]:
                base_conf += 0.05
            if is_derby[i] > 0.3:
                base_conf += 0.05
            if high_stakes[i] > 0.5:
                base_conf += 0.05
            conf[i] = min(0.7, base_conf)

            # update state
            last_date[h] = dt; last_date[a] = dt
            match_dates.setdefault(h, []).append(dt)
            match_dates.setdefault(a, []).append(dt)
            fixture_dates.setdefault(fix_key, []).append(dt)
            if _is_finished(r):
                hg, ag = int(r.home_goals), int(r.away_goals)
                tracker.update(sk, h, a, hg, ag)
                fixture_count[fix_key] = fixture_count.get(fix_key, 0) + 1

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
                # v15: FPL Fixture Difficulty Rating
                "ctx_fdr_h": fdr_h, "ctx_fdr_a": fdr_a,
            },
        )
