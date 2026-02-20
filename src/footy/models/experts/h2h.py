"""H2HExpert — Bayesian head-to-head with time decay and venue adjustment."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _norm3, _pts


class H2HExpert(Expert):
    """
    Bayesian head-to-head with venue-aware probability adjustment:
    - Dirichlet prior calibrated to league base rates
    - Time-decayed observations (half-life 500 days — faster squad turnover)
    - Venue-specific Dirichlet sub-model blended into main probs
    - "Bogey team" detection — systematic underperformance by the favorite
    - H2H goals scored/conceded features
    - H2H surprise rate (how often underdog wins)
    - Proper uncertainty quantification
    """
    name = "h2h"

    HALF_LIFE = 500.0       # days — faster than 730 for modern squad turnover
    PRIOR_STRENGTH = 3.0    # pseudo-count weight of prior
    VENUE_BLEND = 0.35      # weight of venue-specific model in final probs
    # base-rate priors (home-centric)
    PRIOR_H = 0.45
    PRIOR_D = 0.27
    PRIOR_A = 0.28

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        decay = math.log(2) / self.HALF_LIFE
        h2h: dict[frozenset, list] = {}         # (home, away, hg, ag, dt, elo_h, elo_a)
        venue: dict[tuple, list] = {}            # (home, away) -> [(hg, ag, dt)]
        elo: dict[str, float] = {}               # lightweight Elo for quality weighting

        probs = np.full((n, 3), [self.PRIOR_H, self.PRIOR_D, self.PRIOR_A])
        conf = np.zeros(n)
        out_n = np.zeros(n)
        out_pts_h = np.zeros(n); out_pts_a = np.zeros(n); out_gd = np.zeros(n)
        out_vn = np.zeros(n); out_vpts = np.zeros(n); out_vgd = np.zeros(n)
        # New features: H2H goals, bogey, surprise
        out_gf_h = np.zeros(n); out_ga_h = np.zeros(n)
        out_bogey = np.zeros(n)             # bogey team indicator (0-1)
        out_surprise_rate = np.zeros(n)     # how often underdog wins this H2H
        out_dominance = np.zeros(n)         # one team dominating this fixture
        out_venue_wr_h = np.zeros(n)        # home team's venue-specific win rate
        out_venue_gf = np.zeros(n)          # avg goals scored by home at this venue
        out_venue_ga = np.zeros(n)          # avg goals conceded by home at this venue
        # NEW: Elo-quality-weighted H2H
        out_quality_pts_h = np.zeros(n)     # H2H PPG adjusted by opponent Elo at match time
        out_quality_pts_a = np.zeros(n)
        # NEW: recent H2H trend (last 3 vs last 8 — momentum within this specific fixture)
        out_h2h_trend = np.zeros(n)         # positive = home improving in this fixture
        # NEW: goal volatility in this H2H (high = unpredictable fixture)
        out_goal_vol = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            dt = r.utc_date
            key = frozenset([h, a])
            past = h2h.get(key, [])

            # Bayesian update with time-decayed counts (overall H2H)
            alpha_h = self.PRIOR_STRENGTH * self.PRIOR_H
            alpha_d = self.PRIOR_STRENGTH * self.PRIOR_D
            alpha_a = self.PRIOR_STRENGTH * self.PRIOR_A
            total_w = 0.0
            pts_home_w = pts_away_w = gd_w = 0.0
            gf_w = ga_w = 0.0   # goals scored/conceded weighted
            n_h_wins = n_a_wins = n_draws = 0  # unweighted counts for surprise

            for phome, paway, hg_p, ag_p, pdt, *elo_data in past[-24:]:
                age = max(0.0, (dt - pdt).total_seconds() / 86400.0)
                w = math.exp(-decay * age)
                # determine outcome from perspective of home team in *this* match
                if phome == h and paway == a:
                    gh, ga = hg_p, ag_p
                else:
                    gh, ga = ag_p, hg_p

                if gh > ga:
                    alpha_h += w
                    n_h_wins += 1
                elif gh == ga:
                    alpha_d += w
                    n_draws += 1
                else:
                    alpha_a += w
                    n_a_wins += 1

                pts_home_w += _pts(gh, ga) * w
                pts_away_w += _pts(ga, gh) * w
                gd_w += (gh - ga) * w
                gf_w += gh * w
                ga_w += ga * w
                total_w += w

            s = alpha_h + alpha_d + alpha_a
            overall_probs = [alpha_h / s, alpha_d / s, alpha_a / s]

            n_raw = len(past[-24:])
            out_n[i] = n_raw
            out_pts_h[i] = pts_home_w / max(total_w, 1e-6)
            out_pts_a[i] = pts_away_w / max(total_w, 1e-6)
            out_gd[i] = gd_w / max(total_w, 1e-6)
            out_gf_h[i] = gf_w / max(total_w, 1e-6)
            out_ga_h[i] = ga_w / max(total_w, 1e-6)

            # --- Elo-quality-weighted H2H PPG ---
            # Weight each past result by the opponent's Elo at that time
            quality_pts_h_w = quality_pts_a_w = quality_total_w = 0.0
            for phome, paway, hg_p, ag_p, pdt, *elo_data in past[-24:]:
                age = max(0.0, (dt - pdt).total_seconds() / 86400.0)
                w = math.exp(-decay * age)
                if phome == h and paway == a:
                    gh_q, ga_q = hg_p, ag_p
                else:
                    gh_q, ga_q = ag_p, hg_p
                # Use stored Elo if available, else default
                opp_elo = elo_data[1] if len(elo_data) >= 2 else 1500.0  # away Elo for home team
                elo_weight = opp_elo / 1500.0  # >1 for strong opponents
                combined_w = w * elo_weight
                quality_pts_h_w += _pts(gh_q, ga_q) * combined_w
                quality_pts_a_w += _pts(ga_q, gh_q) * combined_w
                quality_total_w += combined_w
            out_quality_pts_h[i] = quality_pts_h_w / max(quality_total_w, 1e-6)
            out_quality_pts_a[i] = quality_pts_a_w / max(quality_total_w, 1e-6)

            # --- Recent H2H trend (momentum within this specific fixture) ---
            if n_raw >= 4:
                # Last 3 meetings vs older meetings — is home team improving?
                pts_list_h = []
                for phome_t, paway_t, hg_t, ag_t, *_ in past[-24:]:
                    if phome_t == h and paway_t == a:
                        pts_list_h.append(_pts(hg_t, ag_t))
                    else:
                        pts_list_h.append(_pts(ag_t, hg_t))
                if len(pts_list_h) >= 4:
                    recent_3 = sum(pts_list_h[-3:]) / 3.0
                    older = sum(pts_list_h[:-3]) / max(len(pts_list_h) - 3, 1)
                    out_h2h_trend[i] = recent_3 - older

            # --- Goal volatility in H2H (unpredictable fixtures) ---
            if n_raw >= 3:
                gd_list = []
                for phome_g, paway_g, hg_g, ag_g, *_ in past[-24:]:
                    if phome_g == h and paway_g == a:
                        gd_list.append(hg_g - ag_g)
                    else:
                        gd_list.append(ag_g - hg_g)
                out_goal_vol[i] = float(np.std(gd_list))

            # --- Venue-specific Dirichlet sub-model ---
            vpast = venue.get((h, a), [])[-8:]
            vn = len(vpast)
            v_alpha_h = self.PRIOR_STRENGTH * self.PRIOR_H
            v_alpha_d = self.PRIOR_STRENGTH * self.PRIOR_D
            v_alpha_a = self.PRIOR_STRENGTH * self.PRIOR_A
            vpts = vgd = vgf = vga = 0.0
            v_total_w = 0.0
            v_wins_h = 0

            for hg_v, ag_v, vdt in vpast:
                age_v = max(0.0, (dt - vdt).total_seconds() / 86400.0)
                wv = math.exp(-decay * age_v)
                if hg_v > ag_v:
                    v_alpha_h += wv
                    v_wins_h += 1
                elif hg_v == ag_v:
                    v_alpha_d += wv
                else:
                    v_alpha_a += wv
                vpts += _pts(hg_v, ag_v)
                vgd += (hg_v - ag_v)
                vgf += hg_v
                vga += ag_v
                v_total_w += wv

            vs = v_alpha_h + v_alpha_d + v_alpha_a
            venue_probs = [v_alpha_h / vs, v_alpha_d / vs, v_alpha_a / vs]

            out_vn[i] = vn
            out_vpts[i] = vpts / max(1, vn)
            out_vgd[i] = vgd / max(1, vn)
            out_venue_wr_h[i] = v_wins_h / max(1, vn)
            out_venue_gf[i] = vgf / max(1, vn)
            out_venue_ga[i] = vga / max(1, vn)

            # --- Blend overall + venue into final probability ---
            # More venue weight when we have enough venue data
            venue_w = min(self.VENUE_BLEND, vn * 0.07)  # ramp up: 0 at 0, 0.35 at 5+
            for k in range(3):
                probs[i, k] = (1 - venue_w) * overall_probs[k] + venue_w * venue_probs[k]
            # renormalize
            ps = probs[i].sum()
            if ps > 0:
                probs[i] /= ps

            # --- Bogey team detection ---
            # A "bogey" exists when the away team (or weaker team) wins
            # disproportionately often in this specific fixture
            total_decided = n_h_wins + n_a_wins
            if total_decided >= 3:
                # Bogey = away team wins more than expected (>40% of decided results)
                away_upset_rate = n_a_wins / total_decided
                # Compare to prior: if away wins much more than 38% base rate
                out_bogey[i] = max(0.0, min(1.0, (away_upset_rate - 0.38) * 3.0))
            # venue-specific bogey amplifier
            if vn >= 2:
                venue_away_rate = 1.0 - out_venue_wr_h[i]
                venue_draw_rate = sum(1 for hg_v, ag_v, _ in vpast if hg_v == ag_v) / max(1, vn)
                venue_non_hw = venue_away_rate + venue_draw_rate * 0.5
                # if home team fails to win at this venue ≥ 60% of the time
                if venue_non_hw > 0.6:
                    out_bogey[i] = max(out_bogey[i], min(1.0, (venue_non_hw - 0.4) * 1.5))

            # --- Surprise rate ---
            # How often does the "underdog" in this H2H actually win?
            if n_raw >= 3:
                # Simple: rate of non-home wins (since home team is usually favored)
                out_surprise_rate[i] = (n_a_wins + 0.5 * n_draws) / n_raw

            # --- Dominance ---
            # How one-sided is this fixture? (positives = home dominates)
            if n_raw >= 2:
                out_dominance[i] = (n_h_wins - n_a_wins) / n_raw

            # confidence = f(effective sample size)
            eff_n = total_w  # time-weighted effective sample size
            # Boost confidence when venue data also available
            venue_boost = min(0.15, vn * 0.03)
            conf[i] = min(1.0, eff_n / 8.0 + venue_boost)

            # --- update ---
            hg, ag = int(r.home_goals), int(r.away_goals)
            # Store Elo at match time for quality-adjusted H2H
            elo_h_now = elo.get(h, 1500.0)
            elo_a_now = elo.get(a, 1500.0)
            h2h.setdefault(key, []).append((h, a, hg, ag, dt, elo_h_now, elo_a_now))
            venue.setdefault((h, a), []).append((hg, ag, dt))
            # Lightweight Elo update
            e_val = 1 / (1 + 10 ** (-(elo_h_now + 40 - elo_a_now) / 400))
            s_val = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            delta = 16 * (s_val - e_val)
            elo[h] = elo_h_now + delta
            elo[a] = elo_a_now - delta

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "h2h_n": out_n, "h2h_pts_h": out_pts_h, "h2h_pts_a": out_pts_a,
                "h2h_gd": out_gd, "h2h_gf_h": out_gf_h, "h2h_ga_h": out_ga_h,
                "h2h_venue_n": out_vn, "h2h_venue_pts": out_vpts, "h2h_venue_gd": out_vgd,
                "h2h_venue_wr_h": out_venue_wr_h,
                "h2h_venue_gf": out_venue_gf, "h2h_venue_ga": out_venue_ga,
                "h2h_bogey": out_bogey,
                "h2h_surprise_rate": out_surprise_rate,
                "h2h_dominance": out_dominance,
                "h2h_quality_pts_h": out_quality_pts_h,
                "h2h_quality_pts_a": out_quality_pts_a,
                "h2h_trend": out_h2h_trend,
                "h2h_goal_vol": out_goal_vol,
            },
        )
