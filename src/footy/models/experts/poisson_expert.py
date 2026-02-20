"""PoissonExpert — Dixon-Coles + Skellam + Monte Carlo + Bivariate Poisson + Copula + COM-Poisson."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist

from footy.models.experts._base import Expert, ExpertResult, _f, _norm3
from footy.models.advanced_math import (
    build_dc_score_matrix, skellam_probs, monte_carlo_simulate,
    build_bivariate_poisson_matrix, build_copula_score_matrix,
    build_com_poisson_matrix, extract_match_probs,
    bradley_terry_probs,
)


class PoissonExpert(Expert):
    """
    Attack / defence strength via EMA tracking → Poisson goal distribution →
    analytical P(H, D, A) through score matrix convolution.
    """
    name = "poisson"

    ALPHA = 0.08        # EMA smoothing factor
    MAX_GOALS = 8       # max goals to consider in score matrix
    AVG = 1.35          # league average goals per team
    VENUE_ALPHA = 0.06  # slower EMA for venue-specific stats

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        attack: dict[str, float] = {}
        defense: dict[str, float] = {}
        # venue-specific attack/defense
        home_att: dict[str, float] = {}
        home_def: dict[str, float] = {}
        away_att: dict[str, float] = {}
        away_def: dict[str, float] = {}
        game_count: dict[str, int] = {}

        # MLE rho estimation: track per-competition history
        _rho_history: dict[str, dict] = {}  # comp → {hg, ag, lh, la lists}
        _rho_cache: dict[str, float] = {}   # comp → estimated ρ
        _RHO_MIN_SAMPLES = 200  # need at least 200 finished matches for MLE

        lam_h = np.zeros(n); lam_a = np.zeros(n)
        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)
        # new score-derived features
        pois_btts = np.zeros(n)        # P(both teams to score)
        pois_o25 = np.zeros(n)         # P(over 2.5 goals)
        pois_o15 = np.zeros(n)         # P(over 1.5 goals)
        pois_u25 = np.zeros(n)         # P(under 2.5 goals)
        pois_cs00 = np.zeros(n)        # P(0-0 clean sheet)
        pois_ml_hg = np.zeros(n)       # most likely home goals
        pois_ml_ag = np.zeros(n)       # most likely away goals
        pois_zero_h = np.zeros(n)      # P(home scores 0)
        pois_zero_a = np.zeros(n)      # P(away scores 0)
        pois_skew = np.zeros(n)        # skewness of goal diff distribution
        # v10: Dixon-Coles adjusted features
        dc_adj_ph = np.zeros(n)        # DC-adjusted P(home)
        dc_adj_pd = np.zeros(n)        # DC-adjusted P(draw)
        dc_adj_pa = np.zeros(n)        # DC-adjusted P(away)
        dc_cs00 = np.zeros(n)          # DC-adjusted P(0-0)
        dc_btts = np.zeros(n)          # DC-adjusted P(BTTS)
        # v10: Skellam distribution features
        sk_mean_gd = np.zeros(n)       # expected goal difference
        sk_var_gd = np.zeros(n)        # variance of goal difference
        sk_skew = np.zeros(n)          # skellam skewness
        # v10: Monte Carlo features (DC-correlated simulation)
        mc_ph = np.zeros(n)            # MC P(home)
        mc_pd = np.zeros(n)            # MC P(draw)
        mc_pa = np.zeros(n)            # MC P(away)
        mc_btts = np.zeros(n)          # MC P(BTTS)
        mc_o25 = np.zeros(n)           # MC P(O2.5)
        mc_o35 = np.zeros(n)           # MC P(O3.5)
        mc_var_total = np.zeros(n)     # MC variance of total goals
        # v11: Bivariate Poisson features (Karlis & Ntzoufras 2003)
        bp_ph = np.zeros(n)            # Bivariate Poisson P(home)
        bp_pd = np.zeros(n)            # Bivariate Poisson P(draw)
        bp_btts = np.zeros(n)          # Bivariate Poisson P(BTTS)
        bp_o25 = np.zeros(n)           # Bivariate Poisson P(O2.5)
        # v11: Frank Copula features (full-scoreline dependency)
        cop_ph = np.zeros(n)           # Copula P(home)
        cop_pd = np.zeros(n)           # Copula P(draw)
        cop_btts = np.zeros(n)         # Copula P(BTTS)
        cop_o25 = np.zeros(n)          # Copula P(O2.5)
        # v11: COM-Poisson features (over/under-dispersion)
        cmp_ph = np.zeros(n)           # COM-Poisson P(home)
        cmp_disp_h = np.zeros(n)       # COM-Poisson dispersion index home
        cmp_disp_a = np.zeros(n)       # COM-Poisson dispersion index away
        # v11: Bradley-Terry features
        bt_ph = np.zeros(n)            # BT P(home)
        bt_pd = np.zeros(n)            # BT P(draw)
        bt_pa = np.zeros(n)            # BT P(away)

        def _a(t): return attack.get(t, self.AVG)
        def _d(t): return defense.get(t, self.AVG * 0.9)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            ah, dh = _a(h), _d(h)
            aa, da = _a(a), _d(a)

            # blend overall + venue-specific if available
            ha_h = home_att.get(h, ah)
            hd_h = home_def.get(h, dh)
            aa_a = away_att.get(a, aa)
            ad_a = away_def.get(a, da)
            att_h = 0.7 * ah + 0.3 * ha_h  # weighted blend
            def_h = 0.7 * dh + 0.3 * hd_h
            att_a = 0.7 * aa + 0.3 * aa_a
            def_a = 0.7 * da + 0.3 * ad_a

            l_h = max(0.3, min(4.5, att_h * def_a / self.AVG * 1.05))
            l_a = max(0.3, min(4.5, att_a * def_h / self.AVG))
            lam_h[i] = l_h; lam_a[i] = l_a

            # score matrix
            goals_range = np.arange(self.MAX_GOALS + 1)
            ph_dist = poisson_dist.pmf(goals_range, l_h)
            pa_dist = poisson_dist.pmf(goals_range, l_a)
            score_mx = np.outer(ph_dist, pa_dist)
            p_home = float(np.sum(np.tril(score_mx, -1)))
            p_draw = float(np.sum(np.diag(score_mx)))
            p_away = float(np.sum(np.triu(score_mx, 1)))
            ph_, pd__, pa_ = _norm3(p_home, p_draw, p_away)
            probs[i] = [ph_, pd__, pa_]

            # score-derived features from the score matrix
            pois_cs00[i] = float(score_mx[0, 0])
            pois_zero_h[i] = float(ph_dist[0])
            pois_zero_a[i] = float(pa_dist[0])
            pois_btts[i] = float(1.0 - ph_dist[0] - pa_dist[0] + score_mx[0, 0])
            # over/under
            total_under25 = sum(
                float(score_mx[hg, ag])
                for hg in range(self.MAX_GOALS + 1)
                for ag in range(self.MAX_GOALS + 1)
                if hg + ag <= 2
            )
            total_under15 = sum(
                float(score_mx[hg, ag])
                for hg in range(self.MAX_GOALS + 1)
                for ag in range(self.MAX_GOALS + 1)
                if hg + ag <= 1
            )
            pois_u25[i] = total_under25
            pois_o25[i] = 1.0 - total_under25
            pois_o15[i] = 1.0 - total_under15
            # most likely score
            best_idx = np.unravel_index(np.argmax(score_mx), score_mx.shape)
            pois_ml_hg[i] = best_idx[0]; pois_ml_ag[i] = best_idx[1]
            # goal difference skewness (positive = home-heavy)
            gd_probs = np.zeros(2 * self.MAX_GOALS + 1)
            for hg_i in range(self.MAX_GOALS + 1):
                for ag_i in range(self.MAX_GOALS + 1):
                    gd_probs[hg_i - ag_i + self.MAX_GOALS] += score_mx[hg_i, ag_i]
            gd_range = np.arange(-self.MAX_GOALS, self.MAX_GOALS + 1)
            mu = np.sum(gd_range * gd_probs)
            var = np.sum(gd_probs * (gd_range - mu) ** 2)
            std = max(np.sqrt(var), 1e-6)
            pois_skew[i] = float(np.sum(gd_probs * ((gd_range - mu) / std) ** 3))

            # v10: Dixon-Coles adjusted score matrix (MLE-estimated ρ per competition)
            comp = getattr(r, 'competition', None) or 'default'
            if comp not in _rho_cache:
                _rho_cache[comp] = -0.13  # initial prior
            rho_est = _rho_cache[comp]

            dc_mx = build_dc_score_matrix(l_h, l_a, rho=rho_est, max_goals=self.MAX_GOALS)
            dc_ph = float(np.sum(np.tril(dc_mx, -1)))
            dc_pd_v = float(np.sum(np.diag(dc_mx)))
            dc_pa_v = float(np.sum(np.triu(dc_mx, 1)))
            dc_ph, dc_pd_v, dc_pa_v = _norm3(dc_ph, dc_pd_v, dc_pa_v)
            dc_adj_ph[i] = dc_ph; dc_adj_pd[i] = dc_pd_v; dc_adj_pa[i] = dc_pa_v
            dc_cs00[i] = float(dc_mx[0, 0])
            dc_btts[i] = float(1.0 - dc_mx[0, :].sum() - dc_mx[:, 0].sum() + dc_mx[0, 0])

            # v10: Skellam distribution features
            sk_res = skellam_probs(l_h, l_a)
            sk_mean_gd[i] = sk_res["mean_diff"]
            sk_var_gd[i] = sk_res["var_diff"]
            sk_skew[i] = sk_res["skewness"]

            # v10: Monte Carlo (DC-correlated, 2000 sims for speed)
            mc_res = monte_carlo_simulate(l_h, l_a, rho=rho_est, n_sims=2000)
            mc_ph[i] = mc_res["mc_p_home"]
            mc_pd[i] = mc_res["mc_p_draw"]
            mc_pa[i] = mc_res["mc_p_away"]
            mc_btts[i] = mc_res["mc_btts"]
            mc_o25[i] = mc_res["mc_o25"]
            mc_o35[i] = mc_res["mc_o35"]
            mc_var_total[i] = mc_res["mc_var_total"]

            # v11: Bivariate Poisson (Karlis & Ntzoufras 2003)
            # λ₃ = max(0, -ρ * √(λ_h * λ_a)) — approximate from DC rho
            bp_lambda3 = max(0.0, -rho_est * (l_h * l_a) ** 0.5 * 0.3)
            bp_mx = build_bivariate_poisson_matrix(l_h, l_a, lambda_3=bp_lambda3,
                                                    max_goals=self.MAX_GOALS)
            bp_probs = extract_match_probs(bp_mx)
            bp_ph[i] = bp_probs["p_home"]
            bp_pd[i] = bp_probs["p_draw"]
            bp_btts[i] = bp_probs["p_btts"]
            bp_o25[i] = bp_probs["p_o25"]

            # v11: Frank Copula (full-scoreline dependency)
            # θ ≈ 10 * ρ maps DC rho to copula parameter range
            cop_theta = rho_est * 10.0  # typically -1.3 for rho=-0.13
            cop_mx = build_copula_score_matrix(l_h, l_a, theta=cop_theta,
                                               max_goals=self.MAX_GOALS)
            cop_feats = extract_match_probs(cop_mx)
            cop_ph[i] = cop_feats["p_home"]
            cop_pd[i] = cop_feats["p_draw"]
            cop_btts[i] = cop_feats["p_btts"]
            cop_o25[i] = cop_feats["p_o25"]

            # v11: COM-Poisson (dispersion-aware Poisson)
            # Estimate ν from historical variance/mean ratio for the competition
            cmp_mx = build_com_poisson_matrix(l_h, l_a, nu_h=0.93, nu_a=0.93,
                                              max_goals=self.MAX_GOALS)
            cmp_feats = extract_match_probs(cmp_mx)
            cmp_ph[i] = cmp_feats["p_home"]
            cmp_disp_h[i] = cmp_feats["disp_home"]
            cmp_disp_a[i] = cmp_feats["disp_away"]

            # v11: Bradley-Terry probabilities from attack/defense strengths
            bt_str_h = (att_h - self.AVG) / max(self.AVG, 0.5)
            bt_str_a = (att_a - self.AVG) / max(self.AVG, 0.5)
            bt_p = bradley_terry_probs(bt_str_h, bt_str_a, home_adv=0.25,
                                       draw_factor=0.28)
            bt_ph[i], bt_pd[i], bt_pa[i] = bt_p

            # confidence — increases with number of known teams
            gc_h = game_count.get(h, 0)
            gc_a = game_count.get(a, 0)
            conf[i] = min(1.0, (gc_h + gc_a) / 20.0)

            # --- update ---
            hg, ag = int(r.home_goals), int(r.away_goals)
            if h in attack:
                attack[h] = (1 - self.ALPHA) * ah + self.ALPHA * hg
                defense[h] = (1 - self.ALPHA) * dh + self.ALPHA * ag
            else:
                attack[h] = float(hg) if hg > 0 else self.AVG
                defense[h] = float(ag) if ag > 0 else self.AVG * 0.9
            if a in attack:
                attack[a] = (1 - self.ALPHA) * aa + self.ALPHA * ag
                defense[a] = (1 - self.ALPHA) * da + self.ALPHA * hg
            else:
                attack[a] = float(ag) if ag > 0 else self.AVG
                defense[a] = float(hg) if hg > 0 else self.AVG * 0.9
            # venue-specific EMA
            if h in home_att:
                home_att[h] = (1 - self.VENUE_ALPHA) * home_att.get(h, ah) + self.VENUE_ALPHA * hg
                home_def[h] = (1 - self.VENUE_ALPHA) * home_def.get(h, dh) + self.VENUE_ALPHA * ag
            else:
                home_att[h] = float(hg) if hg > 0 else self.AVG
                home_def[h] = float(ag) if ag > 0 else self.AVG * 0.9
            if a in away_att:
                away_att[a] = (1 - self.VENUE_ALPHA) * away_att.get(a, aa) + self.VENUE_ALPHA * ag
                away_def[a] = (1 - self.VENUE_ALPHA) * away_def.get(a, da) + self.VENUE_ALPHA * hg
            else:
                away_att[a] = float(ag) if ag > 0 else self.AVG
                away_def[a] = float(hg) if hg > 0 else self.AVG * 0.9
            game_count[h] = game_count.get(h, 0) + 1
            game_count[a] = game_count.get(a, 0) + 1

            # --- MLE rho estimation: accumulate per-competition history ---
            if hg is not None and ag is not None:
                if comp not in _rho_history:
                    _rho_history[comp] = {"hg": [], "ag": [], "lh": [], "la": []}
                _rho_history[comp]["hg"].append(hg)
                _rho_history[comp]["ag"].append(ag)
                _rho_history[comp]["lh"].append(l_h)
                _rho_history[comp]["la"].append(l_a)
                # Re-estimate rho every 200 matches
                rh = _rho_history[comp]
                if len(rh["hg"]) >= _RHO_MIN_SAMPLES and len(rh["hg"]) % 200 == 0:
                    try:
                        from footy.models.advanced_math import estimate_rho_mle
                        _rho_cache[comp] = estimate_rho_mle(
                            np.array(rh["hg"]), np.array(rh["ag"]),
                            np.array(rh["lh"]), np.array(rh["la"]),
                        )
                    except Exception:
                        pass  # keep previous estimate

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "pois_lambda_h": lam_h, "pois_lambda_a": lam_a,
                "pois_lambda_diff": lam_h - lam_a,
                "pois_lambda_sum": lam_h + lam_a,
                "pois_lambda_prod": lam_h * lam_a,
                "pois_btts": pois_btts, "pois_o25": pois_o25,
                "pois_o15": pois_o15, "pois_u25": pois_u25,
                "pois_cs00": pois_cs00,
                "pois_zero_h": pois_zero_h, "pois_zero_a": pois_zero_a,
                "pois_ml_hg": pois_ml_hg, "pois_ml_ag": pois_ml_ag,
                "pois_skew": pois_skew,
                # v10: Dixon-Coles adjusted
                "pois_dc_ph": dc_adj_ph, "pois_dc_pd": dc_adj_pd, "pois_dc_pa": dc_adj_pa,
                "pois_dc_cs00": dc_cs00, "pois_dc_btts": dc_btts,
                # v10: Skellam distribution
                "pois_sk_mean_gd": sk_mean_gd, "pois_sk_var_gd": sk_var_gd,
                "pois_sk_skew": sk_skew,
                # v10: Monte Carlo simulation
                "pois_mc_ph": mc_ph, "pois_mc_pd": mc_pd, "pois_mc_pa": mc_pa,
                "pois_mc_btts": mc_btts, "pois_mc_o25": mc_o25,
                "pois_mc_o35": mc_o35, "pois_mc_var_total": mc_var_total,
                # v11: Bivariate Poisson (Karlis & Ntzoufras 2003)
                "pois_bp_ph": bp_ph, "pois_bp_pd": bp_pd,
                "pois_bp_btts": bp_btts, "pois_bp_o25": bp_o25,
                # v11: Frank Copula (full dependency)
                "pois_cop_ph": cop_ph, "pois_cop_pd": cop_pd,
                "pois_cop_btts": cop_btts, "pois_cop_o25": cop_o25,
                # v11: COM-Poisson (dispersion-aware)
                "pois_cmp_ph": cmp_ph,
                "pois_cmp_disp_h": cmp_disp_h, "pois_cmp_disp_a": cmp_disp_a,
                # v11: Bradley-Terry
                "pois_bt_ph": bt_ph, "pois_bt_pd": bt_pd, "pois_bt_pa": bt_pa,
            },
        )
