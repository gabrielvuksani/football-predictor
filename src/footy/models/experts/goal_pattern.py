"""GoalPatternExpert — First-goal advantage, comeback rate, opponent-quality-adjusted patterns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _f, _norm3


class GoalPatternExpert(Expert):
    """
    Analyses goal-scoring patterns that other experts miss:
    - First-goal advantage (teams that score first win X% of the time)
    - Comeback rate (how often a team recovers from conceding first)
    - Scoring/conceding distributions by half (HT proxy from hthg/htag)
    - **Opponent-quality-adjusted patterns** (performance vs strong/weak teams)
    - **Late-goal tendency** (second-half scoring relative to first-half)
    - **Venue-split first-goal rate** (home vs away)
    - **Defensive resilience against quality** (CS rate vs top opponents)
    - Score-first probability from Poisson model
    """
    name = "goal_pattern"

    WINDOW = 15  # matches to analyze

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        team_history: dict[str, list[dict]] = {}
        home_fg_history: dict[str, list[float]] = {}   # venue-split first-goal
        away_fg_history: dict[str, list[float]] = {}
        elo: dict[str, float] = {}   # for opponent quality

        # outputs
        first_goal_rate_h = np.zeros(n)    # how often home team scores first
        first_goal_rate_a = np.zeros(n)
        comeback_rate_h = np.zeros(n)      # how often they come back from behind
        comeback_rate_a = np.zeros(n)
        ht_goals_rate_h = np.zeros(n)      # fraction of goals scored in first half
        ht_goals_rate_a = np.zeros(n)
        ht_concede_rate_h = np.zeros(n)    # fraction of goals conceded in first half
        ht_concede_rate_a = np.zeros(n)
        first_goal_win_h = np.zeros(n)     # win rate when scoring first
        first_goal_win_a = np.zeros(n)
        multi_goal_rate_h = np.zeros(n)    # rate of scoring 2+ goals
        multi_goal_rate_a = np.zeros(n)
        nil_rate_h = np.zeros(n)           # rate of scoring 0
        nil_rate_a = np.zeros(n)
        lead_hold_rate_h = np.zeros(n)     # rate of holding HT lead
        lead_hold_rate_a = np.zeros(n)
        # NEW: late-goal tendency (2nd half scoring rate relative to 1st)
        late_goal_h = np.zeros(n)
        late_goal_a = np.zeros(n)
        # NEW: venue-split first-goal rate
        venue_fg_h = np.zeros(n)           # home team's first-goal rate at home
        venue_fg_a = np.zeros(n)           # away team's first-goal rate away
        # NEW: opponent-quality-adjusted patterns
        fg_vs_top_h = np.zeros(n)          # first-goal rate vs strong opponents
        fg_vs_top_a = np.zeros(n)
        cs_vs_top_h = np.zeros(n)          # clean-sheet rate vs strong opponents
        cs_vs_top_a = np.zeros(n)
        comeback_vs_top_h = np.zeros(n)    # comeback rate vs strong opponents
        comeback_vs_top_a = np.zeros(n)
        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)

        def _team_stats(team):
            hist = team_history.get(team, [])[-self.WINDOW:]
            if not hist:
                return {}
            fg = [h.get("scored_first", 0) for h in hist]
            cb = [h.get("comeback", 0) for h in hist]
            ht_frac = [h.get("ht_goal_frac", 0.5) for h in hist]
            ht_conc = [h.get("ht_concede_frac", 0.5) for h in hist]
            fg_win = [h.get("first_goal_win", 0) for h in hist if h.get("scored_first")]
            multi = [1.0 if h.get("gf", 0) >= 2 else 0.0 for h in hist]
            nil = [1.0 if h.get("gf", 0) == 0 else 0.0 for h in hist]
            lead_hold = [h.get("held_lead", 0) for h in hist if h.get("had_ht_lead")]
            # late-goal tendency: 2nd half goals / total goals
            late = []
            for h_rec in hist:
                gf = h_rec.get("gf", 0)
                ht_gf = h_rec.get("ht_gf", 0)
                if gf > 0:
                    late.append((gf - ht_gf) / gf)  # fraction of goals in 2nd half
            return {
                "fg_rate": float(np.mean(fg)),
                "cb_rate": float(np.mean(cb)),
                "ht_frac": float(np.mean(ht_frac)),
                "ht_conc_frac": float(np.mean(ht_conc)),
                "fg_win": float(np.mean(fg_win)) if fg_win else 0.5,
                "multi": float(np.mean(multi)),
                "nil": float(np.mean(nil)),
                "lead_hold": float(np.mean(lead_hold)) if lead_hold else 0.5,
                "late_goal": float(np.mean(late)) if late else 0.5,
            }

        def _quality_stats(team, elo_threshold=1600.0):
            """Stats against top-rated opponents only."""
            hist = team_history.get(team, [])[-self.WINDOW:]
            top = [h for h in hist if h.get("opp_elo", 1500) >= elo_threshold]
            if len(top) < 2:
                return {}
            fg_top = [h.get("scored_first", 0) for h in top]
            cs_top = [1.0 if h.get("ga", 1) == 0 else 0.0 for h in top]
            cb_top = [h.get("comeback", 0) for h in top]
            return {
                "fg_vs_top": float(np.mean(fg_top)),
                "cs_vs_top": float(np.mean(cs_top)),
                "cb_vs_top": float(np.mean(cb_top)),
            }

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            # Pre-match features
            hs = _team_stats(h)
            as_ = _team_stats(a)

            first_goal_rate_h[i] = hs.get("fg_rate", 0.5)
            first_goal_rate_a[i] = as_.get("fg_rate", 0.5)
            comeback_rate_h[i] = hs.get("cb_rate", 0.1)
            comeback_rate_a[i] = as_.get("cb_rate", 0.1)
            ht_goals_rate_h[i] = hs.get("ht_frac", 0.5)
            ht_goals_rate_a[i] = as_.get("ht_frac", 0.5)
            ht_concede_rate_h[i] = hs.get("ht_conc_frac", 0.5)
            ht_concede_rate_a[i] = as_.get("ht_conc_frac", 0.5)
            first_goal_win_h[i] = hs.get("fg_win", 0.5)
            first_goal_win_a[i] = as_.get("fg_win", 0.5)
            multi_goal_rate_h[i] = hs.get("multi", 0.4)
            multi_goal_rate_a[i] = as_.get("multi", 0.4)
            nil_rate_h[i] = hs.get("nil", 0.2)
            nil_rate_a[i] = as_.get("nil", 0.2)
            lead_hold_rate_h[i] = hs.get("lead_hold", 0.5)
            lead_hold_rate_a[i] = as_.get("lead_hold", 0.5)

            # NEW: late-goal tendency
            late_goal_h[i] = hs.get("late_goal", 0.5)
            late_goal_a[i] = as_.get("late_goal", 0.5)

            # NEW: venue-split first-goal rate
            h_home_fg = home_fg_history.get(h, [])
            a_away_fg = away_fg_history.get(a, [])
            venue_fg_h[i] = float(np.mean(h_home_fg[-10:])) if h_home_fg else 0.5
            venue_fg_a[i] = float(np.mean(a_away_fg[-10:])) if a_away_fg else 0.5

            # NEW: opponent-quality-adjusted patterns
            qh = _quality_stats(h)
            qa = _quality_stats(a)
            fg_vs_top_h[i] = qh.get("fg_vs_top", 0.5)
            fg_vs_top_a[i] = qa.get("fg_vs_top", 0.5)
            cs_vs_top_h[i] = qh.get("cs_vs_top", 0.3)
            cs_vs_top_a[i] = qa.get("cs_vs_top", 0.3)
            comeback_vs_top_h[i] = qh.get("cb_vs_top", 0.1)
            comeback_vs_top_a[i] = qa.get("cb_vs_top", 0.1)

            # Probability: combine first-goal rates with win-when-first patterns
            fg_h = hs.get("fg_rate", 0.5)
            fg_w_h = hs.get("fg_win", 0.5)
            fg_a = as_.get("fg_rate", 0.5)
            fg_w_a = as_.get("fg_win", 0.5)
            p_h = fg_h * fg_w_h + (1 - fg_a) * 0.3  # score first + win | opponent doesn't score first
            p_a = fg_a * fg_w_a + (1 - fg_h) * 0.3
            p_d = 1.0 - p_h - p_a
            if p_d < 0.15:
                p_d = 0.25
            probs[i] = _norm3(p_h, p_d, p_a)

            n_h = len(team_history.get(h, []))
            n_a = len(team_history.get(a, []))
            conf[i] = min(1.0, (n_h + n_a) / 24.0)

            # --- Update state ---
            hg, ag = int(r.home_goals), int(r.away_goals)
            hthg = _f(getattr(r, "hthg", None))
            htag = _f(getattr(r, "htag", None))
            opp_elo_a = elo.get(a, 1500.0)
            opp_elo_h = elo.get(h, 1500.0)

            # Determine first goal, HT fractions
            scored_first_h = 1.0 if (hthg > 0 and htag == 0) or (hthg > htag) else (0.5 if hthg == htag and hg > 0 else 0.0)
            scored_first_a = 1.0 if (htag > 0 and hthg == 0) or (htag > hthg) else (0.5 if hthg == htag and ag > 0 else 0.0)
            # If no HT data, estimate from final score
            if hthg == 0 and htag == 0 and (hg > 0 or ag > 0):
                scored_first_h = 0.55 if hg > ag else (0.45 if hg < ag else 0.5)
                scored_first_a = 1.0 - scored_first_h

            ht_goal_frac_h = hthg / max(hg, 1) if hg > 0 else 0.0
            ht_goal_frac_a = htag / max(ag, 1) if ag > 0 else 0.0
            ht_concede_frac_h = htag / max(ag, 1) if ag > 0 else 0.0
            ht_concede_frac_a = hthg / max(hg, 1) if hg > 0 else 0.0

            # Comeback: behind at HT but won/drew
            comeback_h = 1.0 if (hthg < htag and hg >= ag) else 0.0
            comeback_a = 1.0 if (htag < hthg and ag >= hg) else 0.0

            had_ht_lead_h = hthg > htag
            held_lead_h = 1.0 if (had_ht_lead_h and hg > ag) else 0.0
            had_ht_lead_a = htag > hthg
            held_lead_a = 1.0 if (had_ht_lead_a and ag > hg) else 0.0

            first_goal_win_val_h = 1.0 if (scored_first_h > 0.5 and hg > ag) else 0.0
            first_goal_win_val_a = 1.0 if (scored_first_a > 0.5 and ag > hg) else 0.0

            team_history.setdefault(h, []).append({
                "gf": hg, "ga": ag,
                "scored_first": scored_first_h,
                "comeback": comeback_h,
                "ht_goal_frac": ht_goal_frac_h,
                "ht_concede_frac": ht_concede_frac_h,
                "ht_gf": int(hthg),   # raw HT goals for late-goal calc
                "first_goal_win": first_goal_win_val_h,
                "had_ht_lead": had_ht_lead_h,
                "held_lead": held_lead_h,
                "opp_elo": opp_elo_a,
            })
            team_history.setdefault(a, []).append({
                "gf": ag, "ga": hg,
                "scored_first": scored_first_a,
                "comeback": comeback_a,
                "ht_goal_frac": ht_goal_frac_a,
                "ht_concede_frac": ht_concede_frac_a,
                "ht_gf": int(htag),
                "first_goal_win": first_goal_win_val_a,
                "had_ht_lead": had_ht_lead_a,
                "held_lead": held_lead_a,
                "opp_elo": opp_elo_h,
            })
            # Venue-split first-goal
            home_fg_history.setdefault(h, []).append(scored_first_h)
            away_fg_history.setdefault(a, []).append(scored_first_a)
            # Lightweight Elo update
            rh = elo.get(h, 1500.0); ra = elo.get(a, 1500.0)
            e_val = 1 / (1 + 10 ** (-(rh + 40 - ra) / 400))
            s_val = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            delta = 16 * (s_val - e_val)
            elo[h] = rh + delta; elo[a] = ra - delta

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "gp_first_goal_h": first_goal_rate_h, "gp_first_goal_a": first_goal_rate_a,
                "gp_comeback_h": comeback_rate_h, "gp_comeback_a": comeback_rate_a,
                "gp_ht_frac_h": ht_goals_rate_h, "gp_ht_frac_a": ht_goals_rate_a,
                "gp_ht_conc_h": ht_concede_rate_h, "gp_ht_conc_a": ht_concede_rate_a,
                "gp_fg_win_h": first_goal_win_h, "gp_fg_win_a": first_goal_win_a,
                "gp_multi_h": multi_goal_rate_h, "gp_multi_a": multi_goal_rate_a,
                "gp_nil_h": nil_rate_h, "gp_nil_a": nil_rate_a,
                "gp_lead_hold_h": lead_hold_rate_h, "gp_lead_hold_a": lead_hold_rate_a,
                # NEW: late-goal tendency
                "gp_late_goal_h": late_goal_h, "gp_late_goal_a": late_goal_a,
                # NEW: venue-split first-goal rate
                "gp_venue_fg_h": venue_fg_h, "gp_venue_fg_a": venue_fg_a,
                # NEW: opponent-quality-adjusted patterns
                "gp_fg_vs_top_h": fg_vs_top_h, "gp_fg_vs_top_a": fg_vs_top_a,
                "gp_cs_vs_top_h": cs_vs_top_h, "gp_cs_vs_top_a": cs_vs_top_a,
                "gp_cb_vs_top_h": comeback_vs_top_h, "gp_cb_vs_top_a": comeback_vs_top_a,
            },
        )
