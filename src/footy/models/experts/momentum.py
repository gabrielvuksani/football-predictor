"""MomentumExpert — EMA crossovers, venue-split momentum, opponent-quality weighting."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _norm3, _pts


class MomentumExpert(Expert):
    """
    Advanced momentum analysis using EMA crossovers, linear regression slopes,
    and rate-of-change features that capture *trajectory* (not just level).

    Unlike FormExpert which tracks rolling averages, MomentumExpert tracks:
    - EMA crossover signals (fast 3-game vs slow 8-game PPG)
    - **Venue-split momentum** (home-only and away-only momentum)
    - **Opponent-quality-weighted momentum** (adjusted for strength of schedule)
    - Goal scoring/conceding slope (linear regression over last N)
    - Form volatility (consistency of results)
    - Performance regression signal (distance from long-term mean)
    - Scoring burst/drought detection
    - Win/loss streak counter
    """
    name = "momentum"

    FAST_N = 3
    SLOW_N = 8
    SLOPE_N = 10
    LONG_TERM_N = 30

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        team_pts: dict[str, list[float]] = {}
        team_gf: dict[str, list[float]] = {}
        team_ga: dict[str, list[float]] = {}
        # NEW: venue-split histories
        team_home_pts: dict[str, list[float]] = {}
        team_away_pts: dict[str, list[float]] = {}
        team_home_gf: dict[str, list[float]] = {}
        team_away_gf: dict[str, list[float]] = {}
        # NEW: opponent Elo tracking for quality-weighted momentum
        team_opp_elo: dict[str, list[float]] = {}
        elo: dict[str, float] = {}

        # Outputs
        ema_cross_h = np.zeros(n)      # fast EMA > slow EMA = positive momentum
        ema_cross_a = np.zeros(n)
        gf_slope_h = np.zeros(n)       # goal scoring trend (regression slope)
        gf_slope_a = np.zeros(n)
        ga_slope_h = np.zeros(n)       # goals conceded trend
        ga_slope_a = np.zeros(n)
        pts_slope_h = np.zeros(n)      # points trend
        pts_slope_a = np.zeros(n)
        vol_h = np.zeros(n)            # form volatility (std of points)
        vol_a = np.zeros(n)
        regress_h = np.zeros(n)        # regression to mean signal
        regress_a = np.zeros(n)
        burst_h = np.zeros(n)          # scoring burst (last 3 vs last 8)
        burst_a = np.zeros(n)
        drought_h = np.zeros(n)        # scoring drought signal
        drought_a = np.zeros(n)
        def_tighten_h = np.zeros(n)    # defensive tightening signal
        def_tighten_a = np.zeros(n)
        # NEW: venue-split momentum
        venue_ema_h = np.zeros(n)      # home team's home-only EMA crossover
        venue_ema_a = np.zeros(n)      # away team's away-only EMA crossover
        venue_gf_slope_h = np.zeros(n) # home scoring trend at home
        venue_gf_slope_a = np.zeros(n) # away scoring trend away
        # NEW: opponent-quality-weighted momentum
        quality_mom_h = np.zeros(n)    # momentum weighted by opponent quality
        quality_mom_a = np.zeros(n)
        # NEW: win/loss streak counter
        streak_h = np.zeros(n)         # +N for N consecutive wins, -N for losses
        streak_a = np.zeros(n)

        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)

        def _ema(values, span):
            """Exponential moving average."""
            if not values:
                return 0.0
            alpha = 2.0 / (span + 1)
            ema_val = values[0]
            for v in values[1:]:
                ema_val = alpha * v + (1 - alpha) * ema_val
            return ema_val

        def _slope(values):
            """Linear regression slope of values."""
            if len(values) < 3:
                return 0.0
            x = np.arange(len(values), dtype=float)
            y = np.array(values, dtype=float)
            xm = x.mean()
            ym = y.mean()
            cov = np.sum((x - xm) * (y - ym))
            var = np.sum((x - xm) ** 2)
            return float(cov / max(var, 1e-8))

        def _streak_count(pts_list):
            """Count consecutive wins (+) or losses (-). 0 if last was draw."""
            if not pts_list:
                return 0.0
            s = 0
            last_pts = pts_list[-1]
            if last_pts == 1:  # draw breaks streak
                return 0.0
            for p in reversed(pts_list):
                if last_pts == 3 and p == 3:
                    s += 1
                elif last_pts == 0 and p == 0:
                    s -= 1
                else:
                    break
            return float(s)

        def _quality_weighted_momentum(pts_list, elo_list, fast_n=3, slow_n=8):
            """Momentum weighted by opponent Elo — beating strong teams = more signal."""
            if len(pts_list) < fast_n or len(elo_list) < fast_n:
                return 0.0
            ELO_MED = 1500.0
            recent_pts = pts_list[-fast_n:]
            recent_elos = elo_list[-fast_n:]
            older_pts = pts_list[-slow_n:]
            older_elos = elo_list[-slow_n:]
            # Weighted average: pts × (opp_elo / 1500)
            r_weighted = sum(p * (e / ELO_MED) for p, e in zip(recent_pts, recent_elos)) / fast_n
            o_weighted = sum(p * (e / ELO_MED) for p, e in zip(older_pts, older_elos)) / len(older_pts)
            return r_weighted - o_weighted

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            # Pre-match features
            h_pts = team_pts.get(h, [])
            a_pts = team_pts.get(a, [])
            h_gf = team_gf.get(h, [])
            a_gf = team_gf.get(a, [])
            h_ga = team_ga.get(h, [])
            a_ga = team_ga.get(a, [])

            # EMA crossover (fast - slow)
            if len(h_pts) >= self.FAST_N:
                ema_cross_h[i] = _ema(h_pts[-self.SLOW_N:], self.FAST_N) - _ema(h_pts[-self.SLOW_N:], self.SLOW_N)
            if len(a_pts) >= self.FAST_N:
                ema_cross_a[i] = _ema(a_pts[-self.SLOW_N:], self.FAST_N) - _ema(a_pts[-self.SLOW_N:], self.SLOW_N)

            # Goal scoring/conceding slopes
            gf_slope_h[i] = _slope(h_gf[-self.SLOPE_N:])
            gf_slope_a[i] = _slope(a_gf[-self.SLOPE_N:])
            ga_slope_h[i] = _slope(h_ga[-self.SLOPE_N:])
            ga_slope_a[i] = _slope(a_ga[-self.SLOPE_N:])
            pts_slope_h[i] = _slope(h_pts[-self.SLOPE_N:])
            pts_slope_a[i] = _slope(a_pts[-self.SLOPE_N:])

            # Form volatility
            if len(h_pts) >= 5:
                vol_h[i] = float(np.std(h_pts[-self.SLOPE_N:]))
            if len(a_pts) >= 5:
                vol_a[i] = float(np.std(a_pts[-self.SLOPE_N:]))

            # Regression to mean signal (recent - long term)
            if len(h_pts) >= self.SLOW_N:
                recent = np.mean(h_pts[-self.FAST_N:]) if len(h_pts) >= self.FAST_N else np.mean(h_pts)
                longterm = np.mean(h_pts[-self.LONG_TERM_N:])
                regress_h[i] = recent - longterm
            if len(a_pts) >= self.SLOW_N:
                recent = np.mean(a_pts[-self.FAST_N:]) if len(a_pts) >= self.FAST_N else np.mean(a_pts)
                longterm = np.mean(a_pts[-self.LONG_TERM_N:])
                regress_a[i] = recent - longterm

            # Scoring burst: recent GF avg / older GF avg
            if len(h_gf) >= self.SLOW_N:
                recent_gf = np.mean(h_gf[-self.FAST_N:])
                older_gf = np.mean(h_gf[-self.SLOW_N:])
                burst_h[i] = recent_gf - older_gf
            if len(a_gf) >= self.SLOW_N:
                recent_gf = np.mean(a_gf[-self.FAST_N:])
                older_gf = np.mean(a_gf[-self.SLOW_N:])
                burst_a[i] = recent_gf - older_gf

            # Scoring drought: consecutive games scoring 0
            if h_gf:
                d = 0
                for g in reversed(h_gf):
                    if g == 0:
                        d += 1
                    else:
                        break
                drought_h[i] = float(d)
            if a_gf:
                d = 0
                for g in reversed(a_gf):
                    if g == 0:
                        d += 1
                    else:
                        break
                drought_a[i] = float(d)

            # Defensive tightening: recent GA decrease
            if len(h_ga) >= self.SLOW_N:
                def_tighten_h[i] = np.mean(h_ga[-self.SLOW_N:-self.FAST_N]) - np.mean(h_ga[-self.FAST_N:])
            if len(a_ga) >= self.SLOW_N:
                def_tighten_a[i] = np.mean(a_ga[-self.SLOW_N:-self.FAST_N]) - np.mean(a_ga[-self.FAST_N:])

            # NEW: venue-split momentum
            h_home_pts = team_home_pts.get(h, [])
            a_away_pts = team_away_pts.get(a, [])
            h_home_gf = team_home_gf.get(h, [])
            a_away_gf = team_away_gf.get(a, [])
            if len(h_home_pts) >= self.FAST_N:
                venue_ema_h[i] = _ema(h_home_pts[-self.SLOW_N:], self.FAST_N) - _ema(h_home_pts[-self.SLOW_N:], self.SLOW_N)
            if len(a_away_pts) >= self.FAST_N:
                venue_ema_a[i] = _ema(a_away_pts[-self.SLOW_N:], self.FAST_N) - _ema(a_away_pts[-self.SLOW_N:], self.SLOW_N)
            venue_gf_slope_h[i] = _slope(h_home_gf[-self.SLOPE_N:])
            venue_gf_slope_a[i] = _slope(a_away_gf[-self.SLOPE_N:])

            # NEW: opponent-quality-weighted momentum
            h_opp_elos = team_opp_elo.get(h, [])
            a_opp_elos = team_opp_elo.get(a, [])
            quality_mom_h[i] = _quality_weighted_momentum(h_pts, h_opp_elos, self.FAST_N, self.SLOW_N)
            quality_mom_a[i] = _quality_weighted_momentum(a_pts, a_opp_elos, self.FAST_N, self.SLOW_N)

            # NEW: win/loss streak counter
            streak_h[i] = _streak_count(h_pts)
            streak_a[i] = _streak_count(a_pts)

            # Momentum-based probability — blend overall + venue momentum
            mom_diff = ema_cross_h[i] - ema_cross_a[i]
            slope_diff = pts_slope_h[i] - pts_slope_a[i]
            venue_mom_diff = venue_ema_h[i] - venue_ema_a[i]
            combined = 0.45 * mom_diff + 0.3 * slope_diff + 0.25 * venue_mom_diff
            p_h = 1 / (1 + math.exp(-1.5 * combined)) * 0.7 + 0.15
            p_a = (1 - p_h + 0.15) * 0.7
            p_d = 1.0 - p_h - p_a
            if p_d < 0.15:
                p_d = 0.25
            probs[i] = _norm3(max(0.05, p_h), max(0.10, p_d), max(0.05, p_a))

            n_h = len(h_pts)
            n_a = len(a_pts)
            conf[i] = min(1.0, (n_h + n_a) / 16.0)

            # --- Update state ---
            hg, ag = int(r.home_goals), int(r.away_goals)
            h_pts_val = float(_pts(hg, ag))
            a_pts_val = float(_pts(ag, hg))
            team_pts.setdefault(h, []).append(h_pts_val)
            team_pts.setdefault(a, []).append(a_pts_val)
            team_gf.setdefault(h, []).append(float(hg))
            team_gf.setdefault(a, []).append(float(ag))
            team_ga.setdefault(h, []).append(float(ag))
            team_ga.setdefault(a, []).append(float(hg))
            # venue-split updates
            team_home_pts.setdefault(h, []).append(h_pts_val)
            team_away_pts.setdefault(a, []).append(a_pts_val)
            team_home_gf.setdefault(h, []).append(float(hg))
            team_away_gf.setdefault(a, []).append(float(ag))
            # opponent Elo tracking
            opp_elo_a = elo.get(a, 1500.0)
            opp_elo_h = elo.get(h, 1500.0)
            team_opp_elo.setdefault(h, []).append(opp_elo_a)
            team_opp_elo.setdefault(a, []).append(opp_elo_h)
            # lightweight Elo update
            rh = elo.get(h, 1500.0); ra = elo.get(a, 1500.0)
            e_val = 1 / (1 + 10 ** (-(rh + 40 - ra) / 400))
            s_val = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            delta = 16 * (s_val - e_val)
            elo[h] = rh + delta; elo[a] = ra - delta

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "mom_ema_cross_h": ema_cross_h, "mom_ema_cross_a": ema_cross_a,
                "mom_ema_cross_diff": ema_cross_h - ema_cross_a,
                "mom_gf_slope_h": gf_slope_h, "mom_gf_slope_a": gf_slope_a,
                "mom_ga_slope_h": ga_slope_h, "mom_ga_slope_a": ga_slope_a,
                "mom_pts_slope_h": pts_slope_h, "mom_pts_slope_a": pts_slope_a,
                "mom_pts_slope_diff": pts_slope_h - pts_slope_a,
                "mom_vol_h": vol_h, "mom_vol_a": vol_a,
                "mom_vol_diff": vol_h - vol_a,
                "mom_regress_h": regress_h, "mom_regress_a": regress_a,
                "mom_burst_h": burst_h, "mom_burst_a": burst_a,
                "mom_drought_h": drought_h, "mom_drought_a": drought_a,
                "mom_def_tighten_h": def_tighten_h, "mom_def_tighten_a": def_tighten_a,
                # NEW: venue-split momentum
                "mom_venue_ema_h": venue_ema_h, "mom_venue_ema_a": venue_ema_a,
                "mom_venue_gf_slope_h": venue_gf_slope_h, "mom_venue_gf_slope_a": venue_gf_slope_a,
                # NEW: opponent-quality-weighted momentum
                "mom_quality_h": quality_mom_h, "mom_quality_a": quality_mom_a,
                "mom_quality_diff": quality_mom_h - quality_mom_a,
                # NEW: win/loss streak counter
                "mom_streak_h": streak_h, "mom_streak_a": streak_a,
            },
        )
