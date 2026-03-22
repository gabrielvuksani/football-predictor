"""Glicko2Expert — Glicko-2 rating system with uncertainty tracking."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3
from footy.models.glicko2 import (
    Glicko2State,
    DEFAULT_RATING,
    DEFAULT_RD,
    DEFAULT_VOLATILITY,
    _g,
    _e,
    _compute_v,
    _compute_delta,
    _compute_new_volatility,
    _update_rating_and_rd,
    _determine_rd,
)


class Glicko2Expert(Expert):
    """
    Glicko-2 rating system with:
    - Rating (μ): team strength estimate
    - Rating deviation (RD/φ): uncertainty in the estimate
    - Volatility (σ): magnitude of expected rating swings
    - Confidence derived from RD (low RD = high confidence)
    - Home advantage estimated from team-specific home/away performance
    - Dynamic draw probability incorporating uncertainty
    """
    name = "glicko2"

    HOME_ADV_BASE = 40.0  # Base home advantage in rating points
    TRACK_N = 20  # Window for home advantage calculation
    DEFAULT_RD_DECAY_DAYS = 1.0  # Days since match before RD increases

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        """Compute Glicko-2 features over matches sorted by utc_date ASC."""
        n = len(df)

        # Main state
        ratings: dict[str, float] = {}
        rds: dict[str, float] = {}
        volatilities: dict[str, float] = {}
        match_dates: dict[str, pd.Timestamp] = {}

        # For team-specific home advantage estimation
        home_rec: dict[str, list] = {}  # List of (gf, ga) for home matches
        away_rec: dict[str, list] = {}  # List of (gf, ga) for away matches

        # Output arrays
        out_rating_h = np.zeros(n)
        out_rating_a = np.zeros(n)
        out_rd_h = np.zeros(n)
        out_rd_a = np.zeros(n)
        out_volatility_h = np.zeros(n)
        out_volatility_a = np.zeros(n)
        out_ph = np.zeros(n)
        out_pd = np.zeros(n)
        out_pa = np.zeros(n)
        out_conf = np.zeros(n)
        out_home_adv = np.zeros(n)
        out_rating_diff = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            # Ensure teams exist
            if h not in ratings:
                ratings[h] = DEFAULT_RATING
                rds[h] = DEFAULT_RD
                volatilities[h] = DEFAULT_VOLATILITY
            if a not in ratings:
                ratings[a] = DEFAULT_RATING
                rds[a] = DEFAULT_RD
                volatilities[a] = DEFAULT_VOLATILITY

            # Get current state
            r_h = ratings[h]
            rd_h = rds[h]
            vol_h = volatilities[h]

            r_a = ratings[a]
            rd_a = rds[a]
            vol_a = volatilities[a]

            # Apply RD decay: increase uncertainty for teams that haven't played recently
            current_date = r.utc_date
            last_h = match_dates.get(h)
            last_a = match_dates.get(a)
            if last_h is not None:
                try:
                    days_since_h = max(0.0, (current_date - last_h).total_seconds() / 86400.0)
                    if days_since_h > self.DEFAULT_RD_DECAY_DAYS:
                        rd_h = _determine_rd(rd_h, days_since_h)
                        rds[h] = rd_h
                except Exception:
                    pass
            if last_a is not None:
                try:
                    days_since_a = max(0.0, (current_date - last_a).total_seconds() / 86400.0)
                    if days_since_a > self.DEFAULT_RD_DECAY_DAYS:
                        rd_a = _determine_rd(rd_a, days_since_a)
                        rds[a] = rd_a
                except Exception:
                    pass

            # Estimate team-specific home advantage
            h_home_rec = home_rec.get(h, [])
            h_away_rec = away_rec.get(h, [])
            a_away_rec = away_rec.get(a, [])

            if len(h_home_rec) >= 3 and len(h_away_rec) >= 3:
                h_home_gd = np.mean([gf - ga for gf, ga in h_home_rec[-self.TRACK_N:]])
                h_away_gd = np.mean([gf - ga for gf, ga in h_away_rec[-self.TRACK_N:]])
                adv = self.HOME_ADV_BASE + (h_home_gd - h_away_gd) * 20.0
                adv = max(0.0, min(100.0, adv))
            else:
                adv = self.HOME_ADV_BASE

            # Glicko-2 expected score
            rating_h_adj = r_h + adv
            e_h = _e(rating_h_adj, r_a, rd_a)

            # Draw probability: rises when teams closer and/or uncertainty higher
            rating_diff = abs(rating_h_adj - r_a)
            mean_rd = (rd_h + rd_a) / 2.0

            draw_base = 0.26
            draw_sensitivity = 0.05
            draw_rd_effect = 0.0002 * mean_rd
            p_draw = draw_base + draw_sensitivity * np.exp(-rating_diff / 200.0) + draw_rd_effect
            p_draw = float(np.clip(p_draw, 0.15, 0.45))

            ph = e_h * (1.0 - p_draw)
            pa = (1.0 - e_h) * (1.0 - p_draw)
            ph, pd_, pa = _norm3(ph, p_draw, pa)

            # Confidence from RD: lower RD = higher confidence
            # Confidence ranges from ~0.2 (RD=350, very uncertain) to 1.0 (RD=30, very sure)
            c_h = 1.0 - (rd_h - 30.0) / (350.0 - 30.0)
            c_a = 1.0 - (rd_a - 30.0) / (350.0 - 30.0)
            conf = (c_h + c_a) / 2.0
            conf = float(np.clip(conf, 0.0, 1.0))

            out_rating_h[i] = r_h
            out_rating_a[i] = r_a
            out_rd_h[i] = rd_h
            out_rd_a[i] = rd_a
            out_volatility_h[i] = vol_h
            out_volatility_a[i] = vol_a
            out_ph[i] = ph
            out_pd[i] = pd_
            out_pa[i] = pa
            out_conf[i] = conf
            out_home_adv[i] = adv
            out_rating_diff[i] = r_h - r_a

            # --- Update state AFTER recording pre-match values ---
            # Only update for finished matches (non-NaN goals)
            if _is_finished(r):
                hg, ag = int(r.home_goals), int(r.away_goals)

                # Determine result
                if hg > ag:
                    score_h, score_a = 1.0, 0.0
                elif hg == ag:
                    score_h, score_a = 0.5, 0.5
                else:
                    score_h, score_a = 0.0, 1.0

                # Compute v (variance) and delta (improvement) for home team
                home_results = [(r_a, rd_a, score_h)]
                v_h = _compute_v(home_results)
                delta_h = _compute_delta(v_h, home_results)
                vol_h_new = _compute_new_volatility(vol_h, delta_h, v_h)

                # Compute v and delta for away team
                away_results = [(r_h + adv, rd_h, score_a)]
                v_a = _compute_v(away_results)
                delta_a = _compute_delta(v_a, away_results)
                vol_a_new = _compute_new_volatility(vol_a, delta_a, v_a)

                # Update ratings and RDs
                r_h_new, rd_h_new = _update_rating_and_rd(r_h, rd_h, vol_h_new, home_results)
                r_a_new, rd_a_new = _update_rating_and_rd(r_a, rd_a, vol_a_new, away_results)

                # Update state
                ratings[h] = r_h_new
                rds[h] = rd_h_new
                volatilities[h] = vol_h_new
                match_dates[h] = r.utc_date

                ratings[a] = r_a_new
                rds[a] = rd_a_new
                volatilities[a] = vol_a_new
                match_dates[a] = r.utc_date

                # Track home/away records
                home_rec.setdefault(h, []).append((hg, ag))
                away_rec.setdefault(a, []).append((ag, hg))

        return ExpertResult(
            probs=np.column_stack([out_ph, out_pd, out_pa]),
            confidence=out_conf,
            features={
                "glicko2_rating_h": out_rating_h,
                "glicko2_rating_a": out_rating_a,
                "glicko2_rd_h": out_rd_h,
                "glicko2_rd_a": out_rd_a,
                "glicko2_volatility_h": out_volatility_h,
                "glicko2_volatility_a": out_volatility_a,
                "glicko2_rating_diff": out_rating_diff,
                "glicko2_home_adv": out_home_adv,
            },
        )
