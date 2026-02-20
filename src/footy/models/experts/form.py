"""FormExpert — Opposition-adjusted form, venue-split, xG, EWMA."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _f, _norm3, _pts
from footy.models.advanced_math import beta_binomial_shrink, schedule_difficulty


class FormExpert(Expert):
    """
    Rolling form analysis with comprehensive venue-split and xG integration:
    - Standard rolling averages with exponential decay weighting
    - Opposition-Adjusted Form (OAF): weight each result by opponent Elo
    - **Complete venue-split** form: home-only and away-only for all stats
    - xG-based form (expected goals from API-Football data)
    - Possession-based performance indicator
    - Shot conversion, momentum, streak, CS/BTTS rates
    - Form against quality opposition (top-Elo opponents)
    """
    name = "form"

    ROLL = 5              # standard rolling window
    OAF_WINDOW = 10       # opponent-adjusted form window
    EXTENDED_WINDOW = 10  # for rates (CS, BTTS)
    DECAY_ALPHA = 0.3     # exponential decay weight for recency

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        roll: dict[str, list[dict]] = {}
        elo: dict[str, float] = {}  # track Elo for OAF weighting
        home_history: dict[str, list[dict]] = {}  # home-only history
        away_history: dict[str, list[dict]] = {}  # away-only history
        ELO_MED = 1500.0

        # outputs
        gf_h = np.zeros(n); ga_h = np.zeros(n); pp_h = np.zeros(n)
        gf_a = np.zeros(n); ga_a = np.zeros(n); pp_a = np.zeros(n)
        sh_h = np.zeros(n); sot_h = np.zeros(n); cor_h = np.zeros(n); card_h = np.zeros(n)
        sh_a = np.zeros(n); sot_a = np.zeros(n); cor_a = np.zeros(n); card_a = np.zeros(n)
        oaf_h = np.zeros(n); oaf_a = np.zeros(n)    # opposition-adjusted PPG
        hf_ppg_h = np.zeros(n); af_ppg_a = np.zeros(n)  # home-form PPG, away-form PPG
        cs_h = np.zeros(n); cs_a = np.zeros(n)
        btts_h = np.zeros(n); btts_a = np.zeros(n)
        mom_h = np.zeros(n); mom_a = np.zeros(n)
        strk_h = np.zeros(n); strk_a = np.zeros(n)
        gsup_h = np.zeros(n); gsup_a = np.zeros(n)
        sotr_h = np.zeros(n); sotr_a = np.zeros(n)
        # v10: Bayesian shrinkage rates, schedule difficulty, shot conversion
        shrunk_wr_h = np.zeros(n); shrunk_wr_a = np.zeros(n)
        shrunk_cs_h = np.zeros(n); shrunk_cs_a = np.zeros(n)
        shrunk_btts_h = np.zeros(n); shrunk_btts_a = np.zeros(n)
        sched_diff_h = np.zeros(n); sched_diff_a = np.zeros(n)
        conversion_h = np.zeros(n); conversion_a = np.zeros(n)
        # NEW: venue-split features (complete — goals, shots, conversion at venue)
        hf_gf_h = np.zeros(n)     # home team's goals scored in home games
        hf_ga_h = np.zeros(n)     # home team's goals conceded in home games
        af_gf_a = np.zeros(n)     # away team's goals scored in away games
        af_ga_a = np.zeros(n)     # away team's goals conceded in away games
        hf_sot_h = np.zeros(n)    # home team shots on target at home
        af_sot_a = np.zeros(n)    # away team shots on target at away
        hf_conv_h = np.zeros(n)   # home team conversion at home
        af_conv_a = np.zeros(n)   # away team conversion at away
        hf_cs_h = np.zeros(n)     # home team clean sheets at home
        af_cs_a = np.zeros(n)     # away team clean sheets at away
        # NEW: exponential decay weighted form
        decay_ppg_h = np.zeros(n)
        decay_ppg_a = np.zeros(n)
        # NEW: xG integration
        xg_h = np.zeros(n)       # home team expected goals from xG data
        xg_a = np.zeros(n)       # away team expected goals from xG data
        xg_diff_h = np.zeros(n)  # home xG - actual goals (overperformance)
        xg_diff_a = np.zeros(n)
        # NEW: form against quality opposition
        form_vs_top_h = np.zeros(n)  # home team PPG vs top-Elo opponents
        form_vs_top_a = np.zeros(n)
        # NEW: possession-based performance
        poss_h = np.zeros(n)          # home team rolling avg possession %
        poss_a = np.zeros(n)          # away team rolling avg possession %
        poss_eff_h = np.zeros(n)      # possession effectiveness: goals per % possession
        poss_eff_a = np.zeros(n)
        poss_dom_h = np.zeros(n)      # possession dominance: avg poss > 55%
        poss_dom_a = np.zeros(n)

        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)

        def _avg(team, key, w=None):
            xs = [d.get(key, 0.0) for d in roll.get(team, [])[-( w or self.ROLL):]]
            return float(np.mean(xs)) if xs else 0.0

        def _rate(team, key):
            xs = [d.get(key, 0.0) for d in roll.get(team, [])[-self.EXTENDED_WINDOW:]]
            return float(np.mean(xs)) if xs else 0.0

        def _oaf_ppg(team):
            """Opposition-adjusted PPG: weight points by opponent Elo."""
            recs = roll.get(team, [])[-self.OAF_WINDOW:]
            if not recs:
                return 0.0
            total_w = 0.0; total_p = 0.0
            for d in recs:
                opp_elo = d.get("opp_elo", ELO_MED)
                w = opp_elo / ELO_MED  # >1 for strong opponents, <1 for weak
                total_p += d["pts"] * w
                total_w += w
            return total_p / max(total_w, 1e-6)

        def _form_vs_top(team, elo_threshold=1600.0):
            """PPG against top-rated opponents (Elo > threshold)."""
            recs = roll.get(team, [])[-self.OAF_WINDOW:]
            top_recs = [d for d in recs if d.get("opp_elo", 1500) >= elo_threshold]
            if len(top_recs) < 2:
                return 0.0
            return float(np.mean([d["pts"] for d in top_recs]))

        def _venue_avg(hist, key, window=5):
            """Average from venue-specific history."""
            recs = hist[-window:]
            if not recs:
                return 0.0
            return float(np.mean([d.get(key, 0.0) for d in recs]))

        def _momentum(team):
            hist = roll.get(team, [])
            if len(hist) < 4:
                return 0.0
            recent = np.mean([d["pts"] for d in hist[-3:]])
            older = np.mean([d["pts"] for d in hist[-6:]])
            return float(recent - older)

        def _streak(team):
            hist = roll.get(team, [])
            if not hist:
                return 0.0
            s = 0
            last = hist[-1]["pts"]
            for d in reversed(hist):
                if last == 3 and d["pts"] == 3:
                    s += 1
                elif last == 0 and d["pts"] == 0:
                    s -= 1
                else:
                    break
            return float(s)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            # pre-match features
            gf_h[i] = _avg(h, "gf"); ga_h[i] = _avg(h, "ga"); pp_h[i] = _avg(h, "pts")
            gf_a[i] = _avg(a, "gf"); ga_a[i] = _avg(a, "ga"); pp_a[i] = _avg(a, "pts")
            sh_h[i] = _avg(h, "sh"); sot_h[i] = _avg(h, "sot"); cor_h[i] = _avg(h, "cor"); card_h[i] = _avg(h, "card")
            sh_a[i] = _avg(a, "sh"); sot_a[i] = _avg(a, "sot"); cor_a[i] = _avg(a, "cor"); card_a[i] = _avg(a, "card")
            oaf_h[i] = _oaf_ppg(h); oaf_a[i] = _oaf_ppg(a)

            # COMPLETE venue-split: home team's home-only form
            h_home = home_history.get(h, [])
            a_away = away_history.get(a, [])
            hf_ppg_h[i] = _venue_avg(h_home, "pts")
            af_ppg_a[i] = _venue_avg(a_away, "pts")
            hf_gf_h[i] = _venue_avg(h_home, "gf")
            hf_ga_h[i] = _venue_avg(h_home, "ga")
            af_gf_a[i] = _venue_avg(a_away, "gf")
            af_ga_a[i] = _venue_avg(a_away, "ga")
            hf_sot_h[i] = _venue_avg(h_home, "sot")
            af_sot_a[i] = _venue_avg(a_away, "sot")
            hf_cs_rate = _venue_avg(h_home, "cs")
            af_cs_rate = _venue_avg(a_away, "cs")
            hf_cs_h[i] = hf_cs_rate
            af_cs_a[i] = af_cs_rate
            # venue-split conversion
            h_home_sot_total = sum(d.get("sot", 0) for d in h_home[-5:])
            h_home_gf_total = sum(d.get("gf", 0) for d in h_home[-5:])
            a_away_sot_total = sum(d.get("sot", 0) for d in a_away[-5:])
            a_away_gf_total = sum(d.get("gf", 0) for d in a_away[-5:])
            hf_conv_h[i] = h_home_gf_total / max(h_home_sot_total, 1)
            af_conv_a[i] = a_away_gf_total / max(a_away_sot_total, 1)

            cs_h[i] = _rate(h, "cs"); cs_a[i] = _rate(a, "cs")
            btts_h[i] = _rate(h, "btts"); btts_a[i] = _rate(a, "btts")
            mom_h[i] = _momentum(h); mom_a[i] = _momentum(a)
            strk_h[i] = _streak(h); strk_a[i] = _streak(a)
            gsup_h[i] = _avg(h, "gsup"); gsup_a[i] = _avg(a, "gsup")
            sv_h = _avg(h, "sh"); sv_sot_h = _avg(h, "sot")
            sv_a = _avg(a, "sh"); sv_sot_a = _avg(a, "sot")
            sotr_h[i] = sv_sot_h / max(sv_h, 0.1)
            sotr_a[i] = sv_sot_a / max(sv_a, 0.1)

            # NEW: exponential decay weighted PPG
            h_recs_all = roll.get(h, [])[-self.ROLL:]
            a_recs_all = roll.get(a, [])[-self.ROLL:]
            if h_recs_all:
                alpha = self.DECAY_ALPHA
                weights_h = [(1 - alpha) ** (len(h_recs_all) - 1 - j) for j in range(len(h_recs_all))]
                ws = sum(weights_h)
                decay_ppg_h[i] = sum(d["pts"] * w for d, w in zip(h_recs_all, weights_h)) / max(ws, 1e-6)
            if a_recs_all:
                alpha = self.DECAY_ALPHA
                weights_a = [(1 - alpha) ** (len(a_recs_all) - 1 - j) for j in range(len(a_recs_all))]
                ws = sum(weights_a)
                decay_ppg_a[i] = sum(d["pts"] * w for d, w in zip(a_recs_all, weights_a)) / max(ws, 1e-6)

            # NEW: xG integration from API-Football
            xg_home_val = _f(getattr(r, 'af_xg_home', None))
            xg_away_val = _f(getattr(r, 'af_xg_away', None))
            # Use rolling xG from history (for pre-match prediction)
            xg_h[i] = _avg(h, "xg")
            xg_a[i] = _avg(a, "xg")
            # xG over/underperformance: positive = scoring more than xG suggests
            xg_diff_h[i] = _avg(h, "xg_diff")
            xg_diff_a[i] = _avg(a, "xg_diff")

            # NEW: form against quality opponents
            form_vs_top_h[i] = _form_vs_top(h)
            form_vs_top_a[i] = _form_vs_top(a)

            # NEW: possession-based features
            poss_h[i] = _avg(h, "poss")
            poss_a[i] = _avg(a, "poss")
            # Possession effectiveness: goals per possession % point
            h_poss_avg = poss_h[i]
            a_poss_avg = poss_a[i]
            poss_eff_h[i] = gf_h[i] / max(h_poss_avg, 1.0) * 50.0 if h_poss_avg > 0 else 0.0
            poss_eff_a[i] = gf_a[i] / max(a_poss_avg, 1.0) * 50.0 if a_poss_avg > 0 else 0.0
            # Possession dominance: how often they exceed 55%
            h_recs_poss = roll.get(h, [])[-self.ROLL:]
            a_recs_poss = roll.get(a, [])[-self.ROLL:]
            if h_recs_poss:
                poss_dom_h[i] = sum(1.0 for d in h_recs_poss if d.get("poss", 50) > 55) / len(h_recs_poss)
            if a_recs_poss:
                poss_dom_a[i] = sum(1.0 for d in a_recs_poss if d.get("poss", 50) > 55) / len(a_recs_poss)

            # v10: Bayesian shrinkage for noisy rates
            h_recs = roll.get(h, [])
            a_recs = roll.get(a, [])
            n_h_recs = len(h_recs)
            n_a_recs = len(a_recs)
            # Win rate shrinkage
            h_wins = sum(1 for d in h_recs if d["pts"] == 3)
            a_wins = sum(1 for d in a_recs if d["pts"] == 3)
            shrunk_wr_h[i] = beta_binomial_shrink(h_wins, n_h_recs, 4.5, 5.5)
            shrunk_wr_a[i] = beta_binomial_shrink(a_wins, n_a_recs, 4.5, 5.5)
            # CS rate shrinkage
            h_cs = sum(1 for d in h_recs if d.get("cs", 0) > 0)
            a_cs = sum(1 for d in a_recs if d.get("cs", 0) > 0)
            shrunk_cs_h[i] = beta_binomial_shrink(h_cs, n_h_recs, 3.5, 6.5)
            shrunk_cs_a[i] = beta_binomial_shrink(a_cs, n_a_recs, 3.5, 6.5)
            # BTTS rate shrinkage
            h_btts = sum(1 for d in h_recs if d.get("btts", 0) > 0)
            a_btts = sum(1 for d in a_recs if d.get("btts", 0) > 0)
            shrunk_btts_h[i] = beta_binomial_shrink(h_btts, n_h_recs, 5.0, 5.0)
            shrunk_btts_a[i] = beta_binomial_shrink(a_btts, n_a_recs, 5.0, 5.0)
            # Schedule difficulty
            h_opp_elos = [d.get("opp_elo", 1500) for d in h_recs[-5:]]
            a_opp_elos = [d.get("opp_elo", 1500) for d in a_recs[-5:]]
            sched_diff_h[i] = schedule_difficulty(h_opp_elos) if h_opp_elos else 1500.0
            sched_diff_a[i] = schedule_difficulty(a_opp_elos) if a_opp_elos else 1500.0
            # Shot conversion (goals per shot on target)
            h_total_sot = sum(d.get("sot", 0) for d in h_recs[-10:])
            h_total_gf = sum(d.get("gf", 0) for d in h_recs[-10:])
            a_total_sot = sum(d.get("sot", 0) for d in a_recs[-10:])
            a_total_gf = sum(d.get("gf", 0) for d in a_recs[-10:])
            conversion_h[i] = h_total_gf / max(h_total_sot, 1)
            conversion_a[i] = a_total_gf / max(a_total_sot, 1)

            # form-based probs: blend OAF and venue-split
            oaf_diff = oaf_h[i] - oaf_a[i]
            # Venue-adjusted OAF: home-form PPG vs away-form PPG
            venue_diff = hf_ppg_h[i] - af_ppg_a[i]
            # Blend overall form with venue-specific form (60/40 when venue data exists)
            blend_diff = oaf_diff
            if h_home and a_away:
                blend_diff = 0.6 * oaf_diff + 0.4 * (venue_diff - 1.0)  # venue_diff normalized around ~1.5 PPG
            p_h = 1 / (1 + math.exp(-0.8 * blend_diff))
            p_a = 1 - p_h
            p_d = max(0.18, 0.28 - abs(blend_diff) * 0.05)
            probs[i] = _norm3(p_h * (1 - p_d), p_d, p_a * (1 - p_d))

            # confidence
            n_h = len(roll.get(h, []))
            n_a = len(roll.get(a, []))
            conf[i] = min(1.0, (n_h + n_a) / 20.0)

            # --- update state ---
            hg, ag = int(r.home_goals), int(r.away_goals)
            hs = _f(getattr(r, "hs", 0)); hst = _f(getattr(r, "hst", 0))
            hc = _f(getattr(r, "hc", 0)); hy = _f(getattr(r, "hy", 0))
            hr_ = _f(getattr(r, "hr", 0))
            as_ = _f(getattr(r, "as_", 0)); ast = _f(getattr(r, "ast", 0))
            ac = _f(getattr(r, "ac", 0)); ay = _f(getattr(r, "ay", 0))
            ar_ = _f(getattr(r, "ar", 0))

            # Read possession data from API-Football
            poss_home_val = _f(getattr(r, 'af_possession_home', None))
            poss_away_val = _f(getattr(r, 'af_possession_away', None))

            opp_elo_a = elo.get(a, 1500.0)
            opp_elo_h = elo.get(h, 1500.0)

            h_rec = {
                "gf": hg, "ga": ag, "pts": _pts(hg, ag),
                "sh": hs, "sot": hst, "cor": hc, "card": hy + 2.5 * hr_,
                "cs": 1.0 if ag == 0 else 0.0,
                "btts": 1.0 if (hg > 0 and ag > 0) else 0.0,
                "gsup": hg - ag, "opp_elo": opp_elo_a,
                "xg": xg_home_val, "xg_diff": hg - xg_home_val if xg_home_val > 0 else 0.0,
                "poss": poss_home_val if poss_home_val > 0 else 50.0,
            }
            a_rec = {
                "gf": ag, "ga": hg, "pts": _pts(ag, hg),
                "sh": as_, "sot": ast, "cor": ac, "card": ay + 2.5 * ar_,
                "cs": 1.0 if hg == 0 else 0.0,
                "btts": 1.0 if (hg > 0 and ag > 0) else 0.0,
                "gsup": ag - hg, "opp_elo": opp_elo_h,
                "xg": xg_away_val, "xg_diff": ag - xg_away_val if xg_away_val > 0 else 0.0,
                "poss": poss_away_val if poss_away_val > 0 else 50.0,
            }
            roll.setdefault(h, []).append(h_rec)
            roll.setdefault(a, []).append(a_rec)
            # venue-specific history
            home_history.setdefault(h, []).append(h_rec)
            away_history.setdefault(a, []).append(a_rec)
            # lightweight Elo update for OAF weighting
            rh = elo.get(h, 1500.0); ra = elo.get(a, 1500.0)
            e = 1 / (1 + 10 ** (-(rh + 40 - ra) / 400))
            s = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            delta = 16 * (s - e)
            elo[h] = rh + delta; elo[a] = ra - delta

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "form_gf_h": gf_h, "form_ga_h": ga_h, "form_pts_h": pp_h,
                "form_gf_a": gf_a, "form_ga_a": ga_a, "form_pts_a": pp_a,
                "form_sh_h": sh_h, "form_sot_h": sot_h, "form_cor_h": cor_h, "form_card_h": card_h,
                "form_sh_a": sh_a, "form_sot_a": sot_a, "form_cor_a": cor_a, "form_card_a": card_a,
                "form_oaf_h": oaf_h, "form_oaf_a": oaf_a,
                "form_home_ppg_h": hf_ppg_h, "form_away_ppg_a": af_ppg_a,
                "form_cs_h": cs_h, "form_cs_a": cs_a,
                "form_btts_h": btts_h, "form_btts_a": btts_a,
                "form_momentum_h": mom_h, "form_momentum_a": mom_a,
                "form_streak_h": strk_h, "form_streak_a": strk_a,
                "form_gsup_h": gsup_h, "form_gsup_a": gsup_a,
                "form_sotr_h": sotr_h, "form_sotr_a": sotr_a,
                # v10: Bayesian shrinkage + schedule difficulty + shot conversion
                "form_shrunk_wr_h": shrunk_wr_h, "form_shrunk_wr_a": shrunk_wr_a,
                "form_shrunk_cs_h": shrunk_cs_h, "form_shrunk_cs_a": shrunk_cs_a,
                "form_shrunk_btts_h": shrunk_btts_h, "form_shrunk_btts_a": shrunk_btts_a,
                "form_sched_diff_h": sched_diff_h, "form_sched_diff_a": sched_diff_a,
                "form_conversion_h": conversion_h, "form_conversion_a": conversion_a,
                # NEW: complete venue-split form features
                "form_hf_gf_h": hf_gf_h, "form_hf_ga_h": hf_ga_h,
                "form_af_gf_a": af_gf_a, "form_af_ga_a": af_ga_a,
                "form_hf_sot_h": hf_sot_h, "form_af_sot_a": af_sot_a,
                "form_hf_conv_h": hf_conv_h, "form_af_conv_a": af_conv_a,
                "form_hf_cs_h": hf_cs_h, "form_af_cs_a": af_cs_a,
                # NEW: exponential decay weighted PPG
                "form_decay_ppg_h": decay_ppg_h, "form_decay_ppg_a": decay_ppg_a,
                # NEW: xG features
                "form_xg_h": xg_h, "form_xg_a": xg_a,
                "form_xg_diff_h": xg_diff_h, "form_xg_diff_a": xg_diff_a,
                # NEW: form against quality opponents
                "form_vs_top_h": form_vs_top_h, "form_vs_top_a": form_vs_top_a,
                # NEW: possession-based features
                "form_poss_h": poss_h, "form_poss_a": poss_a,
                "form_poss_eff_h": poss_eff_h, "form_poss_eff_a": poss_eff_a,
                "form_poss_dom_h": poss_dom_h, "form_poss_dom_a": poss_dom_a,
            },
        )
