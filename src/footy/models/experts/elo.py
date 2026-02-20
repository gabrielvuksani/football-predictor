"""EloExpert — Rating dynamics, nonlinear transforms, momentum."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _f, _norm3, _pts
from footy.models.elo_core import elo_expected as _elo_exp, elo_draw_prob as _elo_dp, dynamic_k as _dk
from footy.models.advanced_math import tanh_transform, log_transform


class EloExpert(Expert):
    """
    Team Elo ratings with:
    - Team-specific home advantage (tracked from home/away records)
    - Dynamic K-factor (convergence + goal-diff scaling)
    - Elo momentum (direction of recent Elo change)
    - Elo volatility (std-dev of rating over last N games)
    """
    name = "elo"

    ELO_DEFAULT = 1500.0
    K_BASE = 20.0
    BLANKET_HOME = 40.0
    TRACK_N = 20          # home / away record window
    MOMENTUM_N = 6        # games for momentum calc

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        ratings: dict[str, float] = {}
        counts: dict[str, int] = {}
        home_rec: dict[str, list] = {}    # (pts, gf, ga)
        away_rec: dict[str, list] = {}
        elo_history: dict[str, list] = {} # last N ratings for volatility

        out_elo_h = np.zeros(n)
        out_elo_a = np.zeros(n)
        out_diff = np.zeros(n)
        out_ph = np.zeros(n)
        out_pd = np.zeros(n)
        out_pa = np.zeros(n)
        out_conf = np.zeros(n)
        out_home_adv = np.zeros(n)
        out_momentum_h = np.zeros(n)
        out_momentum_a = np.zeros(n)
        out_volatility_h = np.zeros(n)
        out_volatility_a = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            rh = ratings.get(h, self.ELO_DEFAULT)
            ra = ratings.get(a, self.ELO_DEFAULT)

            # team-specific home advantage
            h_rec = home_rec.get(h, [])
            a_rec_h = away_rec.get(h, [])
            if len(h_rec) >= 3 and len(a_rec_h) >= 3:
                h_ppg = np.mean([x[0] for x in h_rec[-self.TRACK_N:]])
                a_ppg = np.mean([x[0] for x in a_rec_h[-self.TRACK_N:]])
                h_gd = np.mean([x[1] - x[2] for x in h_rec[-self.TRACK_N:]])
                a_gd = np.mean([x[1] - x[2] for x in a_rec_h[-self.TRACK_N:]])
                adv = self.BLANKET_HOME + (h_ppg - a_ppg) * 50 + (h_gd - a_gd) * 10
                adv = max(0.0, min(150.0, adv))
            else:
                adv = self.BLANKET_HOME

            rh_adj = rh + adv

            # expected — use shared core
            e_h = _elo_exp(rh_adj, ra)
            p_draw = _elo_dp(rh_adj, ra)
            ph = e_h * (1 - p_draw)
            pa = (1 - e_h) * (1 - p_draw)
            ph, pd_, pa = _norm3(ph, p_draw, pa)

            # momentum & volatility
            hist_h = elo_history.get(h, [])
            hist_a = elo_history.get(a, [])
            mom_h = (hist_h[-1] - hist_h[-self.MOMENTUM_N]) / max(1, len(hist_h[-self.MOMENTUM_N:])) if len(hist_h) >= self.MOMENTUM_N else 0.0
            mom_a = (hist_a[-1] - hist_a[-self.MOMENTUM_N]) / max(1, len(hist_a[-self.MOMENTUM_N:])) if len(hist_a) >= self.MOMENTUM_N else 0.0
            vol_h = float(np.std(hist_h[-self.MOMENTUM_N:])) if len(hist_h) >= 3 else 30.0
            vol_a = float(np.std(hist_a[-self.MOMENTUM_N:])) if len(hist_a) >= 3 else 30.0

            # confidence — increases with games played (capped at 1)
            c_h = min(1.0, counts.get(h, 0) / 30.0)
            c_a = min(1.0, counts.get(a, 0) / 30.0)
            conf = (c_h + c_a) / 2.0

            out_elo_h[i] = rh; out_elo_a[i] = ra; out_diff[i] = rh - ra
            out_ph[i] = ph; out_pd[i] = pd_; out_pa[i] = pa
            out_conf[i] = conf; out_home_adv[i] = adv
            out_momentum_h[i] = mom_h; out_momentum_a[i] = mom_a
            out_volatility_h[i] = vol_h; out_volatility_a[i] = vol_a

            # --- update state AFTER recording pre-match values ---
            hg, ag = int(r.home_goals), int(r.away_goals)
            s_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            gd = abs(hg - ag)
            n_h = counts.get(h, 0)
            n_a = counts.get(a, 0)
            k_h = _dk(n_h, gd, k_base=self.K_BASE)
            k_a = _dk(n_a, gd, k_base=self.K_BASE)
            k = (k_h + k_a) / 2.0
            delta = k * (s_home - e_h)
            ratings[h] = rh + delta
            ratings[a] = ra - delta
            counts[h] = n_h + 1
            counts[a] = n_a + 1
            home_rec.setdefault(h, []).append((_pts(hg, ag), hg, ag))
            away_rec.setdefault(a, []).append((_pts(ag, hg), ag, hg))
            elo_history.setdefault(h, []).append(ratings[h])
            elo_history.setdefault(a, []).append(ratings[a])

        # Derived features using advanced math
        out_tanh_diff = np.array([tanh_transform(d, 400.0) for d in out_diff])
        out_log_diff = np.array([log_transform(d) for d in out_diff])
        out_elo_form_h = out_momentum_h * out_conf  # momentum weighted by confidence
        out_elo_form_a = out_momentum_a * out_conf

        return ExpertResult(
            probs=np.column_stack([out_ph, out_pd, out_pa]),
            confidence=out_conf,
            features={
                "elo_home": out_elo_h, "elo_away": out_elo_a, "elo_diff": out_diff,
                "elo_home_adv": out_home_adv,
                "elo_momentum_h": out_momentum_h, "elo_momentum_a": out_momentum_a,
                "elo_volatility_h": out_volatility_h, "elo_volatility_a": out_volatility_a,
                # v10: nonlinear transforms + form signals
                "elo_tanh_diff": out_tanh_diff,
                "elo_log_diff": out_log_diff,
                "elo_weighted_mom_h": out_elo_form_h,
                "elo_weighted_mom_a": out_elo_form_a,
            },
        )
