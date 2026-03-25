"""FBrefAdvancedExpert — Advanced team stats from FBref (StatsBomb/Opta quality).

Uses xG, npxG, shot quality, passing effectiveness, pressing intensity,
and defensive actions to build a comprehensive team quality assessment.

Data source: fbref_team_stats table (populated by data_ingest.py)
Fallback: Synthetic stats from match_extras (shots, SoT, corners, cards)

Features capture PLAYING STYLE which correlates with outcomes:
- Attacking quality: npxG/90, shots on target/90, progressive passes
- Defensive quality: tackles won %, interceptions/90, errors
- Style metrics: shot accuracy, pass completion, shot distance
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _f, _is_finished, _norm3


class FBrefAdvancedExpert(Expert):
    """Uses FBref advanced stats for team quality assessment."""

    name = "fbref_advanced"

    WINDOW = 8  # rolling window for synthetic stats

    # FBref column names to check for
    FBREF_COLS = [
        "fb_xg", "fb_npxg", "fb_shots", "fb_sot",
        "fb_passes_completed", "fb_passes_attempted",
        "fb_tackles_won", "fb_interceptions", "fb_clean_sheets",
    ]

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)

        # Check if FBref advanced columns exist in the DataFrame
        has_fbref = all(c in df.columns for c in ["fb_xg", "fb_npxg"])

        # Per-team rolling stats (when FBref data unavailable)
        team_shots: dict[str, list[float]] = {}
        team_sot: dict[str, list[float]] = {}
        team_corners: dict[str, list[float]] = {}
        team_goals: dict[str, list[float]] = {}
        team_conceded: dict[str, list[float]] = {}
        team_yellows: dict[str, list[float]] = {}

        # Per-team rolling FBref stats (when available)
        team_fb_npxg: dict[str, list[float]] = {}
        team_fb_xg: dict[str, list[float]] = {}
        team_fb_passes_comp: dict[str, list[float]] = {}
        team_fb_passes_att: dict[str, list[float]] = {}
        team_fb_tackles: dict[str, list[float]] = {}
        team_fb_interceptions: dict[str, list[float]] = {}

        # Output arrays
        fb_att_quality_h = np.zeros(n)
        fb_att_quality_a = np.zeros(n)
        fb_def_quality_h = np.zeros(n)
        fb_def_quality_a = np.zeros(n)
        fb_shot_quality_h = np.zeros(n)
        fb_shot_quality_a = np.zeros(n)
        fb_shot_accuracy_h = np.zeros(n)
        fb_shot_accuracy_a = np.zeros(n)
        fb_shot_volume_h = np.zeros(n)
        fb_shot_volume_a = np.zeros(n)
        fb_corner_dom_h = np.zeros(n)
        fb_corner_dom_a = np.zeros(n)
        fb_discipline_h = np.zeros(n)
        fb_discipline_a = np.zeros(n)
        fb_quality_mismatch = np.zeros(n)
        fb_has_data = np.zeros(n)
        # FBref-specific output features
        fb_npxg_diff = np.zeros(n)
        fb_pass_accuracy_h = np.zeros(n)
        fb_pass_accuracy_a = np.zeros(n)
        fb_pressing_h = np.zeros(n)
        fb_pressing_a = np.zeros(n)
        fb_progressive_h = np.zeros(n)
        fb_progressive_a = np.zeros(n)
        fb_has_fbref = np.zeros(n)

        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            # Compute rolling stats for each team
            sh_h = team_shots.get(h, [])
            sh_a = team_shots.get(a, [])
            sot_h = team_sot.get(h, [])
            sot_a = team_sot.get(a, [])
            cor_h = team_corners.get(h, [])
            cor_a = team_corners.get(a, [])
            gf_h = team_goals.get(h, [])
            gf_a = team_goals.get(a, [])
            gc_h = team_conceded.get(h, [])
            gc_a = team_conceded.get(a, [])
            yc_h = team_yellows.get(h, [])
            yc_a = team_yellows.get(a, [])

            # --- FBref advanced stats path (when fbref_team_stats joined) ---
            fb_npxg_h_list = team_fb_npxg.get(h, [])
            fb_npxg_a_list = team_fb_npxg.get(a, [])

            if has_fbref and len(fb_npxg_h_list) >= 3 and len(fb_npxg_a_list) >= 3:
                fb_has_fbref[i] = 1.0
                fb_has_data[i] = 1.0
                w = self.WINDOW

                # npxG-based attacking quality
                avg_npxg_h = np.mean(fb_npxg_h_list[-w:])
                avg_npxg_a = np.mean(fb_npxg_a_list[-w:])
                avg_xg_h = np.mean(team_fb_xg.get(h, [0.0])[-w:])
                avg_xg_a = np.mean(team_fb_xg.get(a, [0.0])[-w:])
                fb_att_quality_h[i] = avg_npxg_h * 0.6 + avg_xg_h * 0.4
                fb_att_quality_a[i] = avg_npxg_a * 0.6 + avg_xg_a * 0.4

                # npxG differential
                fb_npxg_diff[i] = avg_npxg_h - avg_npxg_a

                # Pass accuracy
                pc_h = team_fb_passes_comp.get(h, [])
                pa_h = team_fb_passes_att.get(h, [])
                pc_a = team_fb_passes_comp.get(a, [])
                pa_a = team_fb_passes_att.get(a, [])
                if len(pc_h) >= 3 and len(pa_h) >= 3:
                    avg_pc_h = np.mean(pc_h[-w:])
                    avg_pa_h = np.mean(pa_h[-w:])
                    fb_pass_accuracy_h[i] = avg_pc_h / max(avg_pa_h, 1.0)
                if len(pc_a) >= 3 and len(pa_a) >= 3:
                    avg_pc_a = np.mean(pc_a[-w:])
                    avg_pa_a = np.mean(pa_a[-w:])
                    fb_pass_accuracy_a[i] = avg_pc_a / max(avg_pa_a, 1.0)

                # Pressing intensity: (tackles_won + interceptions) per match
                tk_h = team_fb_tackles.get(h, [])
                int_h = team_fb_interceptions.get(h, [])
                tk_a = team_fb_tackles.get(a, [])
                int_a = team_fb_interceptions.get(a, [])
                if len(tk_h) >= 3 and len(int_h) >= 3:
                    fb_pressing_h[i] = np.mean(tk_h[-w:]) + np.mean(int_h[-w:])
                if len(tk_a) >= 3 and len(int_a) >= 3:
                    fb_pressing_a[i] = np.mean(tk_a[-w:]) + np.mean(int_a[-w:])

                # Defensive quality from pressing + low conceded xG
                fb_def_quality_h[i] = fb_pressing_h[i] * 0.05 + max(0.0, 2.0 - avg_npxg_a) * 0.5
                fb_def_quality_a[i] = fb_pressing_a[i] * 0.05 + max(0.0, 2.0 - avg_npxg_h) * 0.5

                # Progressive quality: pass accuracy * npxG (style indicator)
                fb_progressive_h[i] = fb_pass_accuracy_h[i] * avg_npxg_h
                fb_progressive_a[i] = fb_pass_accuracy_a[i] * avg_npxg_a

                # Shot quality and accuracy from FBref shots/SoT if available
                fb_sh_h = team_shots.get(h, [])
                fb_sh_a = team_shots.get(a, [])
                fb_sot_h_list = team_sot.get(h, [])
                fb_sot_a_list = team_sot.get(a, [])
                if len(fb_sh_h) >= 3:
                    avg_sh = np.mean(fb_sh_h[-w:])
                    avg_sot = np.mean(fb_sot_h_list[-w:]) if fb_sot_h_list else 0.0
                    fb_shot_volume_h[i] = avg_sh
                    fb_shot_accuracy_h[i] = avg_sot / max(avg_sh, 1.0)
                    fb_shot_quality_h[i] = avg_npxg_h / max(avg_sh, 1.0)
                if len(fb_sh_a) >= 3:
                    avg_sh = np.mean(fb_sh_a[-w:])
                    avg_sot = np.mean(fb_sot_a_list[-w:]) if fb_sot_a_list else 0.0
                    fb_shot_volume_a[i] = avg_sh
                    fb_shot_accuracy_a[i] = avg_sot / max(avg_sh, 1.0)
                    fb_shot_quality_a[i] = avg_npxg_a / max(avg_sh, 1.0)

                # Quality mismatch and probabilities
                quality_h = fb_att_quality_h[i] + fb_def_quality_h[i]
                quality_a = fb_att_quality_a[i] + fb_def_quality_a[i]
                fb_quality_mismatch[i] = quality_h - quality_a

                adj = np.clip(fb_quality_mismatch[i] * 0.06, -0.15, 0.15)
                p_h = 0.40 + adj
                p_d = 0.28 - abs(adj) * 0.2
                p_a = 0.32 - adj
                probs[i] = _norm3(max(0.05, p_h), max(0.10, p_d), max(0.05, p_a))
                conf[i] = min(0.6, 0.15 + abs(fb_quality_mismatch[i]) * 0.15)

            # --- Fallback: basic stats from match_extras ---
            elif len(sh_h) >= 3 and len(sh_a) >= 3:
                fb_has_data[i] = 1.0
                w = self.WINDOW

                # Attacking quality: goals + SoT + corners (offensive pressure)
                avg_gf_h = np.mean(gf_h[-w:]) if gf_h else 0.0
                avg_sot_h = np.mean(sot_h[-w:]) if sot_h else 0.0
                avg_cor_h = np.mean(cor_h[-w:]) if cor_h else 0.0
                fb_att_quality_h[i] = avg_gf_h * 0.4 + avg_sot_h * 0.04 + avg_cor_h * 0.02

                avg_gf_a = np.mean(gf_a[-w:]) if gf_a else 0.0
                avg_sot_a = np.mean(sot_a[-w:]) if sot_a else 0.0
                avg_cor_a = np.mean(cor_a[-w:]) if cor_a else 0.0
                fb_att_quality_a[i] = avg_gf_a * 0.4 + avg_sot_a * 0.04 + avg_cor_a * 0.02

                # Defensive quality: low goals conceded
                avg_gc_h = np.mean(gc_h[-w:]) if gc_h else 1.3
                avg_gc_a = np.mean(gc_a[-w:]) if gc_a else 1.3
                fb_def_quality_h[i] = max(0.0, 2.0 - avg_gc_h)  # higher = better defense
                fb_def_quality_a[i] = max(0.0, 2.0 - avg_gc_a)

                # Shot quality: goals per shot on target
                avg_sh_h = np.mean(sh_h[-w:]) if sh_h else 1.0
                avg_sh_a = np.mean(sh_a[-w:]) if sh_a else 1.0
                fb_shot_quality_h[i] = avg_gf_h / max(avg_sot_h, 1.0)
                fb_shot_quality_a[i] = avg_gf_a / max(avg_sot_a, 1.0)

                # Shot accuracy: SoT / total shots
                fb_shot_accuracy_h[i] = avg_sot_h / max(avg_sh_h, 1.0)
                fb_shot_accuracy_a[i] = avg_sot_a / max(avg_sh_a, 1.0)

                # Shot volume
                fb_shot_volume_h[i] = avg_sh_h
                fb_shot_volume_a[i] = avg_sh_a

                # Corner dominance
                total_cor = avg_cor_h + avg_cor_a
                if total_cor > 0:
                    fb_corner_dom_h[i] = avg_cor_h / total_cor
                    fb_corner_dom_a[i] = avg_cor_a / total_cor

                # Discipline (yellow cards per match — lower = more disciplined)
                avg_yc_h = np.mean(yc_h[-w:]) if yc_h else 2.0
                avg_yc_a = np.mean(yc_a[-w:]) if yc_a else 2.0
                fb_discipline_h[i] = avg_yc_h
                fb_discipline_a[i] = avg_yc_a

                # Quality mismatch (composite)
                quality_h = fb_att_quality_h[i] + fb_def_quality_h[i]
                quality_a = fb_att_quality_a[i] + fb_def_quality_a[i]
                fb_quality_mismatch[i] = quality_h - quality_a

                # Derive probabilities from quality mismatch
                adj = np.clip(fb_quality_mismatch[i] * 0.06, -0.15, 0.15)
                p_h = 0.40 + adj
                p_d = 0.28 - abs(adj) * 0.2
                p_a = 0.32 - adj
                probs[i] = _norm3(max(0.05, p_h), max(0.10, p_d), max(0.05, p_a))
                conf[i] = min(0.5, 0.1 + abs(fb_quality_mismatch[i]) * 0.15)

            # Update state
            if not _is_finished(r):
                continue
            hg, ag = int(r.home_goals), int(r.away_goals)

            hs = _f(getattr(r, "hs", None))
            as_ = _f(getattr(r, "as_", None))
            hst = _f(getattr(r, "hst", None))
            ast = _f(getattr(r, "ast", None))
            hc = _f(getattr(r, "hc", None))
            ac = _f(getattr(r, "ac", None))
            hy = _f(getattr(r, "hy", None))
            ay = _f(getattr(r, "ay", None))

            # Store basic stats for home team (when playing at home)
            team_shots.setdefault(h, []).append(hs)
            team_sot.setdefault(h, []).append(hst)
            team_corners.setdefault(h, []).append(hc)
            team_goals.setdefault(h, []).append(float(hg))
            team_conceded.setdefault(h, []).append(float(ag))
            team_yellows.setdefault(h, []).append(hy)

            # Store basic stats for away team (when playing away)
            team_shots.setdefault(a, []).append(as_)
            team_sot.setdefault(a, []).append(ast)
            team_corners.setdefault(a, []).append(ac)
            team_goals.setdefault(a, []).append(float(ag))
            team_conceded.setdefault(a, []).append(float(hg))
            team_yellows.setdefault(a, []).append(ay)

            # Store FBref advanced stats (if columns exist)
            if has_fbref:
                fb_xg_h = _f(getattr(r, "fb_xg", None))
                fb_npxg_h_val = _f(getattr(r, "fb_npxg", None))
                _f(getattr(r, "fb_shots", None))
                _f(getattr(r, "fb_sot", None))
                fb_pc_h = _f(getattr(r, "fb_passes_completed", None))
                fb_pa_h = _f(getattr(r, "fb_passes_attempted", None))
                fb_tk_h = _f(getattr(r, "fb_tackles_won", None))
                fb_int_h = _f(getattr(r, "fb_interceptions", None))

                # Home team FBref stats
                team_fb_xg.setdefault(h, []).append(fb_xg_h)
                team_fb_npxg.setdefault(h, []).append(fb_npxg_h_val)
                team_fb_passes_comp.setdefault(h, []).append(fb_pc_h)
                team_fb_passes_att.setdefault(h, []).append(fb_pa_h)
                team_fb_tackles.setdefault(h, []).append(fb_tk_h)
                team_fb_interceptions.setdefault(h, []).append(fb_int_h)

                # Away team: use same row values (FBref data is per-match for home team,
                # so away team gets opponent's defensive stats inverted)
                # For a proper implementation, we'd need separate away columns;
                # here we track what we have for the away team from their home matches
                team_fb_xg.setdefault(a, [])
                team_fb_npxg.setdefault(a, [])
                team_fb_passes_comp.setdefault(a, [])
                team_fb_passes_att.setdefault(a, [])
                team_fb_tackles.setdefault(a, [])
                team_fb_interceptions.setdefault(a, [])

        # v17: Emit aliased keys expected by _build_v13_features in council.py
        # Map our computed features to the keys the stacking model looks for
        feat = {
            "fb_att_quality_h": fb_att_quality_h,
            "fb_att_quality_a": fb_att_quality_a,
            "fb_def_quality_h": fb_def_quality_h,
            "fb_def_quality_a": fb_def_quality_a,
            "fb_shot_quality_h": fb_shot_quality_h,
            "fb_shot_quality_a": fb_shot_quality_a,
            "fb_shot_accuracy_h": fb_shot_accuracy_h,
            "fb_shot_accuracy_a": fb_shot_accuracy_a,
            "fb_shot_volume_h": fb_shot_volume_h,
            "fb_shot_volume_a": fb_shot_volume_a,
            "fb_corner_dom_h": fb_corner_dom_h,
            "fb_corner_dom_a": fb_corner_dom_a,
            "fb_discipline_h": fb_discipline_h,
            "fb_discipline_a": fb_discipline_a,
            "fb_quality_mismatch": fb_quality_mismatch,
            "fb_has_data": fb_has_data,
            # FBref advanced features (populated when fbref_team_stats joined)
            "fb_npxg_diff": fb_npxg_diff,
            "fb_pass_accuracy_h": fb_pass_accuracy_h,
            "fb_pass_accuracy_a": fb_pass_accuracy_a,
            "fb_pressing_h": fb_pressing_h,
            "fb_pressing_a": fb_pressing_a,
            "fb_progressive_h": fb_progressive_h,
            "fb_progressive_a": fb_progressive_a,
            "fb_has_fbref": fb_has_fbref,
        }

        # Aliases: council.py's _build_v13_features looks for these keys
        # fb_possession_h/a → use pass_accuracy as closest proxy (no raw possession column)
        feat["fb_possession_h"] = fb_pass_accuracy_h
        feat["fb_possession_a"] = fb_pass_accuracy_a
        # fb_xg_h/a → use att_quality (weighted blend of npxG and xG)
        feat["fb_xg_h"] = fb_att_quality_h
        feat["fb_xg_a"] = fb_att_quality_a
        # fb_shots_h/a → use shot_volume (rolling avg total shots)
        feat["fb_shots_h"] = fb_shot_volume_h
        feat["fb_shots_a"] = fb_shot_volume_a
        # fb_pass_pct_h/a → use pass_accuracy (completed / attempted)
        feat["fb_pass_pct_h"] = fb_pass_accuracy_h
        feat["fb_pass_pct_a"] = fb_pass_accuracy_a

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features=feat,
        )
