"""BayesianRateExpert — Beta-Binomial shrinkage for all team rates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _norm3, _pts
from footy.models.advanced_math import beta_binomial_shrink, league_specific_prior


class BayesianRateExpert(Expert):
    """
    Systematic Beta-Binomial shrinkage for all team rates.

    For each team, computes shrunk estimates of:
    - Win rate (overall + home/away venue split)
    - Clean sheet rate
    - BTTS rate (both teams to score)
    - Over 2.5 goals rate
    - Scoring first rate
    - Goals per match (Gamma-Poisson shrinkage approximation)

    Uses league-specific priors so that (e.g.) Bundesliga's higher scoring
    rate doesn't contaminate La Liga estimates.
    """
    name = "bayesian_rate"

    WINDOW = 20  # matches to consider for rate estimation

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)

        # Track per-team match history
        team_history: dict[str, list[dict]] = {}  # team -> list of match dicts

        # Output arrays
        bayes_wr_h = np.zeros(n);  bayes_wr_a = np.zeros(n)
        bayes_cs_h = np.zeros(n);  bayes_cs_a = np.zeros(n)
        bayes_btts_h = np.zeros(n); bayes_btts_a = np.zeros(n)
        bayes_o25_h = np.zeros(n);  bayes_o25_a = np.zeros(n)
        bayes_sf_h = np.zeros(n);   bayes_sf_a = np.zeros(n)   # scoring first
        bayes_gpm_h = np.zeros(n);  bayes_gpm_a = np.zeros(n)  # goals per match
        bayes_home_wr = np.zeros(n)  # home team's home-only win rate
        bayes_away_wr = np.zeros(n)  # away team's away-only win rate
        conf = np.zeros(n)
        probs = np.full((n, 3), 1 / 3)

        for i, r in enumerate(df.itertuples(index=False)):
            ht, at = r.home_team, r.away_team
            comp = getattr(r, "competition", "PL")

            for team, is_home, out_wr, out_cs, out_btts, out_o25, out_sf, out_gpm in [
                (ht, True, bayes_wr_h, bayes_cs_h, bayes_btts_h, bayes_o25_h, bayes_sf_h, bayes_gpm_h),
                (at, False, bayes_wr_a, bayes_cs_a, bayes_btts_a, bayes_o25_a, bayes_sf_a, bayes_gpm_a),
            ]:
                hist = team_history.get(team, [])[-self.WINDOW:]
                n_matches = len(hist)

                if n_matches >= 3:
                    wins = sum(1 for m in hist if m["pts"] == 3)
                    cs = sum(1 for m in hist if m["ga"] == 0)
                    btts = sum(1 for m in hist if m["gf"] > 0 and m["ga"] > 0)
                    o25 = sum(1 for m in hist if m["gf"] + m["ga"] > 2)
                    sf = sum(1 for m in hist if m["gf"] > 0)
                    total_goals = sum(m["gf"] for m in hist)

                    # League-specific priors
                    wr_prior = league_specific_prior(comp, "home_win")
                    cs_prior = league_specific_prior(comp, "cs")
                    btts_prior = league_specific_prior(comp, "btts")
                    o25_prior = league_specific_prior(comp, "o25")

                    out_wr[i] = beta_binomial_shrink(wins, n_matches, *wr_prior)
                    out_cs[i] = beta_binomial_shrink(cs, n_matches, *cs_prior)
                    out_btts[i] = beta_binomial_shrink(btts, n_matches, *btts_prior)
                    out_o25[i] = beta_binomial_shrink(o25, n_matches, *o25_prior)
                    out_sf[i] = beta_binomial_shrink(sf, n_matches, 5.0, 5.0)
                    # Gamma-Poisson-ish shrinkage for goals per match
                    out_gpm[i] = (total_goals + 1.35 * 3) / (n_matches + 3)
                else:
                    # fallback to prior means
                    out_wr[i] = 0.45
                    out_cs[i] = 0.35
                    out_btts[i] = 0.50
                    out_o25[i] = 0.48
                    out_sf[i] = 0.50
                    out_gpm[i] = 1.35

                # Venue-specific win rate
                if is_home:
                    home_hist = [m for m in hist if m.get("venue") == "home"]
                    if len(home_hist) >= 3:
                        hw = sum(1 for m in home_hist if m["pts"] == 3)
                        bayes_home_wr[i] = beta_binomial_shrink(hw, len(home_hist), 4.6, 5.4)
                    else:
                        bayes_home_wr[i] = 0.46
                else:
                    away_hist = [m for m in hist if m.get("venue") == "away"]
                    if len(away_hist) >= 3:
                        aw = sum(1 for m in away_hist if m["pts"] == 3)
                        bayes_away_wr[i] = beta_binomial_shrink(aw, len(away_hist), 3.2, 6.8)
                    else:
                        bayes_away_wr[i] = 0.30

            # Bayesian probabilities from rates
            p_h = bayes_home_wr[i] * 0.6 + bayes_wr_h[i] * 0.4
            p_a = bayes_away_wr[i] * 0.6 + bayes_wr_a[i] * 0.4
            p_d = max(0.15, 1.0 - p_h - p_a)
            p_h, p_d, p_a = _norm3(p_h, p_d, p_a)
            probs[i] = [p_h, p_d, p_a]

            # Confidence based on data availability
            n_h = len(team_history.get(ht, []))
            n_a = len(team_history.get(at, []))
            conf[i] = min(1.0, (n_h + n_a) / 30.0)

            # --- update history ---
            hg_val = int(r.home_goals)
            ag_val = int(r.away_goals)
            team_history.setdefault(ht, []).append({
                "gf": hg_val, "ga": ag_val,
                "pts": _pts(hg_val, ag_val), "venue": "home",
            })
            team_history.setdefault(at, []).append({
                "gf": ag_val, "ga": hg_val,
                "pts": _pts(ag_val, hg_val), "venue": "away",
            })

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "bayes_wr_h": bayes_wr_h, "bayes_wr_a": bayes_wr_a,
                "bayes_cs_h": bayes_cs_h, "bayes_cs_a": bayes_cs_a,
                "bayes_btts_h": bayes_btts_h, "bayes_btts_a": bayes_btts_a,
                "bayes_o25_h": bayes_o25_h, "bayes_o25_a": bayes_o25_a,
                "bayes_sf_h": bayes_sf_h, "bayes_sf_a": bayes_sf_a,
                "bayes_gpm_h": bayes_gpm_h, "bayes_gpm_a": bayes_gpm_a,
                "bayes_home_wr": bayes_home_wr, "bayes_away_wr": bayes_away_wr,
                "bayes_wr_diff": bayes_wr_h - bayes_wr_a,
                "bayes_cs_diff": bayes_cs_h - bayes_cs_a,
                "bayes_gpm_diff": bayes_gpm_h - bayes_gpm_a,
                "bayes_gpm_sum": bayes_gpm_h + bayes_gpm_a,
            },
        )
