"""SquadRotationExpert -- fixture congestion, rest days, fatigue impact.

NOTE on feature overlap with ContextExpert:
    rot_home_days_rest ≈ ctx_rest_h (same underlying data)
    rot_home_matches_7d ≈ ctx_cong7_h
    rot_rest_advantage ≈ ctx_rest_diff

    The overlapping features are kept for backward compatibility with trained
    models, but the ensemble's L2 regularization handles multicollinearity.
    SquadRotation's UNIQUE value is:
    - rot_*_congestion: composite congestion score (different formula)
    - rot_*_fatigue_impact: historical rested-vs-congested PPG differential
    These features capture team-specific fatigue sensitivity that Context doesn't.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3, _pts


class SquadRotationExpert(Expert):
    """
    Predicts the likelihood and impact of squad rotation based on fixture
    congestion, cup commitments, and league position.

    Key signals:
    - Days since last match (rest)
    - Fixture density over 7 / 14 / 30 day windows
    - Congestion score (0-1) combining density metrics
    - Fatigue impact on performance (congested teams underperform)
    - Rest advantage differential between home and away
    - Historical performance when well-rested vs congested
    """

    name = "squad_rotation"

    # Thresholds
    IDEAL_REST_DAYS = 6.0          # ~1 match per week is ideal
    MIN_REST_DAYS = 2.0            # less than this = extreme congestion
    CONGESTION_7D_HIGH = 3         # 3 games in 7 days = very congested
    CONGESTION_30D_HIGH = 10       # 10 games in 30 days = very congested

    # Probability adjustment caps
    MAX_REST_BOOST_PER_DAY = 0.04  # ~4% per extra day of rest advantage
    MAX_TOTAL_BOOST = 0.12         # cap total fatigue-based adjustment

    # Confidence parameters
    MIN_GAMES_FOR_CONF = 5         # need at least 5 games per team
    FULL_CONF_GAMES = 20           # full confidence at 20 games per team

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)

        # Per-team state tracking
        # List of datetime objects for each team's match dates
        team_dates: dict[str, list[datetime]] = {}
        # Track each team's league points and games for league position proxy
        team_league_pts: dict[str, float] = {}
        team_league_games: dict[str, int] = {}
        # Track performance when rested vs congested
        team_rested_results: dict[str, list[float]] = {}   # pts when rest >= 5 days
        team_congested_results: dict[str, list[float]] = {}  # pts when rest < 4 days

        # Output arrays
        home_days_rest = np.zeros(n)
        away_days_rest = np.zeros(n)
        home_matches_7d = np.zeros(n)
        away_matches_7d = np.zeros(n)
        home_matches_14d = np.zeros(n)
        away_matches_14d = np.zeros(n)
        home_matches_30d = np.zeros(n)
        away_matches_30d = np.zeros(n)
        home_congestion = np.zeros(n)
        away_congestion = np.zeros(n)
        congestion_diff = np.zeros(n)
        home_fatigue_impact = np.zeros(n)
        away_fatigue_impact = np.zeros(n)
        rest_advantage = np.zeros(n)

        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            match_date = _parse_date(r.utc_date)
            comp = getattr(r, "competition", "")

            # ---------------------------------------------------------------
            # Compute pre-match features from current state
            # ---------------------------------------------------------------
            h_dates = team_dates.get(h, [])
            a_dates = team_dates.get(a, [])

            # Days since last match
            h_rest = _days_since_last(match_date, h_dates)
            a_rest = _days_since_last(match_date, a_dates)
            home_days_rest[i] = h_rest
            away_days_rest[i] = a_rest

            # Matches in rolling windows
            h_m7 = _matches_in_window(match_date, h_dates, 7)
            a_m7 = _matches_in_window(match_date, a_dates, 7)
            h_m14 = _matches_in_window(match_date, h_dates, 14)
            a_m14 = _matches_in_window(match_date, a_dates, 14)
            h_m30 = _matches_in_window(match_date, h_dates, 30)
            a_m30 = _matches_in_window(match_date, a_dates, 30)

            home_matches_7d[i] = h_m7
            away_matches_7d[i] = a_m7
            home_matches_14d[i] = h_m14
            away_matches_14d[i] = a_m14
            home_matches_30d[i] = h_m30
            away_matches_30d[i] = a_m30

            # Congestion score (0-1)
            h_cong = _congestion_score(h_rest, h_m7, h_m14, h_m30)
            a_cong = _congestion_score(a_rest, a_m7, a_m14, a_m30)
            home_congestion[i] = h_cong
            away_congestion[i] = a_cong
            congestion_diff[i] = h_cong - a_cong

            # Fatigue impact: how much congestion is expected to hurt
            # performance.  Combines congestion score with historical
            # rested-vs-congested performance differential.
            h_fatigue = _fatigue_impact(
                h_cong,
                team_rested_results.get(h, []),
                team_congested_results.get(h, []),
            )
            a_fatigue = _fatigue_impact(
                a_cong,
                team_rested_results.get(a, []),
                team_congested_results.get(a, []),
            )
            home_fatigue_impact[i] = h_fatigue
            away_fatigue_impact[i] = a_fatigue

            # Rest advantage (positive = home has more rest)
            rest_adv = h_rest - a_rest
            rest_advantage[i] = rest_adv

            # ---------------------------------------------------------------
            # Build probabilities from fatigue differential
            # ---------------------------------------------------------------
            # Fatigue differential: negative means that side is more fatigued
            # (a more negative fatigue_impact means worse expected performance)
            fatigue_diff = a_fatigue - h_fatigue  # positive = home in better shape

            # Also factor in raw rest difference (capped contribution)
            rest_signal = np.clip(rest_adv, -5.0, 5.0) * self.MAX_REST_BOOST_PER_DAY

            # Combined adjustment
            raw_adj = 0.6 * fatigue_diff + 0.4 * rest_signal
            adj = np.clip(raw_adj, -self.MAX_TOTAL_BOOST, self.MAX_TOTAL_BOOST)

            p_h = 1.0 / 3.0 + adj
            p_a = 1.0 / 3.0 - adj
            p_d = 1.0 / 3.0  # draw unaffected by fatigue differential

            probs[i] = _norm3(max(0.05, p_h), max(0.15, p_d), max(0.05, p_a))

            # ---------------------------------------------------------------
            # Confidence: ramps with number of games tracked per team
            # ---------------------------------------------------------------
            n_h = len(h_dates)
            n_a = len(a_dates)
            min_games = min(n_h, n_a)
            if min_games < self.MIN_GAMES_FOR_CONF:
                conf[i] = 0.0
            else:
                conf[i] = min(
                    1.0,
                    (min_games - self.MIN_GAMES_FOR_CONF)
                    / (self.FULL_CONF_GAMES - self.MIN_GAMES_FOR_CONF),
                )

            # ---------------------------------------------------------------
            # Update state (only for finished matches)
            # ---------------------------------------------------------------
            if not _is_finished(r):
                continue

            hg, ag = int(r.home_goals), int(r.away_goals)
            h_pts_val = float(_pts(hg, ag))
            a_pts_val = float(_pts(ag, hg))

            # Record match date
            team_dates.setdefault(h, []).append(match_date)
            team_dates.setdefault(a, []).append(match_date)

            # Update league position proxy (only for league matches)
            if _is_league(comp):
                team_league_pts[h] = team_league_pts.get(h, 0.0) + h_pts_val
                team_league_pts[a] = team_league_pts.get(a, 0.0) + a_pts_val
                team_league_games[h] = team_league_games.get(h, 0) + 1
                team_league_games[a] = team_league_games.get(a, 0) + 1

            # Track performance by rest status
            if h_rest >= 5.0:
                team_rested_results.setdefault(h, []).append(h_pts_val)
            elif h_rest > 0.0:
                team_congested_results.setdefault(h, []).append(h_pts_val)

            if a_rest >= 5.0:
                team_rested_results.setdefault(a, []).append(a_pts_val)
            elif a_rest > 0.0:
                team_congested_results.setdefault(a, []).append(a_pts_val)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "rot_home_days_rest": home_days_rest,
                "rot_away_days_rest": away_days_rest,
                "rot_home_matches_7d": home_matches_7d,
                "rot_away_matches_7d": away_matches_7d,
                "rot_home_matches_30d": home_matches_30d,
                "rot_away_matches_30d": away_matches_30d,
                "rot_home_congestion": home_congestion,
                "rot_away_congestion": away_congestion,
                "rot_congestion_diff": congestion_diff,
                "rot_home_fatigue_impact": home_fatigue_impact,
                "rot_away_fatigue_impact": away_fatigue_impact,
                "rot_rest_advantage": rest_advantage,
            },
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_date(utc_date) -> datetime:
    """Convert utc_date (string or datetime-like) to a datetime object."""
    if isinstance(utc_date, datetime):
        return utc_date
    if isinstance(utc_date, pd.Timestamp):
        return utc_date.to_pydatetime()
    try:
        # Try ISO format first
        return datetime.fromisoformat(str(utc_date).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass
    try:
        return pd.to_datetime(utc_date).to_pydatetime()
    except Exception:
        return datetime(2000, 1, 1)


def _days_since_last(match_date: datetime, dates: list[datetime]) -> float:
    """Days since the team's most recent match.  Returns 30.0 if no history."""
    if not dates:
        return 30.0  # assume well-rested if no data
    last = dates[-1]
    delta = (match_date - last).total_seconds() / 86400.0
    # Clamp to reasonable range; negative means data ordering issue
    return max(0.0, min(delta, 60.0))


def _matches_in_window(
    match_date: datetime, dates: list[datetime], window_days: int
) -> float:
    """Count how many matches a team played in the last *window_days* days."""
    if not dates:
        return 0.0
    cutoff = match_date - timedelta(days=window_days)
    count = 0
    # Walk backwards for efficiency (dates are chronological)
    for d in reversed(dates):
        if d < cutoff:
            break
        count += 1
    return float(count)


def _congestion_score(
    days_rest: float,
    matches_7d: float,
    matches_14d: float,
    matches_30d: float,
) -> float:
    """
    Combine rest and density metrics into a single 0-1 congestion score.

    0.0 = completely fresh, 1.0 = extremely congested.
    """
    # Rest component: 0 at 7+ days, 1 at 2 or fewer days
    rest_score = np.clip(1.0 - (days_rest - 2.0) / 5.0, 0.0, 1.0)

    # 7-day density: 0 at 0-1 games, 1 at 3+ games
    density_7 = np.clip((matches_7d - 1.0) / 2.0, 0.0, 1.0)

    # 14-day density: 0 at 0-2 games, 1 at 5+ games
    density_14 = np.clip((matches_14d - 2.0) / 3.0, 0.0, 1.0)

    # 30-day density: 0 at 0-4 games, 1 at 10+ games
    density_30 = np.clip((matches_30d - 4.0) / 6.0, 0.0, 1.0)

    # Weighted combination
    score = 0.40 * rest_score + 0.30 * density_7 + 0.15 * density_14 + 0.15 * density_30
    return float(np.clip(score, 0.0, 1.0))


def _fatigue_impact(
    congestion: float,
    rested_results: list[float],
    congested_results: list[float],
) -> float:
    """
    Estimate performance impact from fatigue.

    Returns a value in roughly [-0.15, 0.0] where more negative means
    worse expected performance due to fatigue.

    Logic:
    - Base impact scales with congestion score (high congestion = negative)
    - Modulated by historical evidence: if team historically drops points
      when congested, the impact is amplified.
    - If no rotation data, use a generic fatigue curve.
    """
    if congestion <= 0.05:
        # Fresh team -- no fatigue penalty
        return 0.0

    # Base fatigue curve: exponential penalty as congestion rises
    # At congestion=0.5 -> ~-0.04, at congestion=1.0 -> ~-0.12
    base_impact = -0.12 * (1.0 - math.exp(-2.5 * congestion))

    # Historical modulation
    if len(rested_results) >= 3 and len(congested_results) >= 3:
        avg_rested = sum(rested_results[-15:]) / min(len(rested_results), 15)
        avg_congested = sum(congested_results[-15:]) / min(len(congested_results), 15)
        # ppg difference: positive means team does worse when congested
        ppg_drop = (avg_rested - avg_congested) / 3.0  # normalise to [0, 1]
        ppg_drop = float(np.clip(ppg_drop, -0.3, 0.3))
        # Amplify or dampen base impact
        base_impact *= (1.0 + ppg_drop)

    return float(np.clip(base_impact, -0.15, 0.0))


def _is_league(competition) -> bool:
    """Heuristic to detect league (non-cup) competitions."""
    if not competition:
        return True  # assume league if unknown
    comp_lower = str(competition).lower()
    cup_keywords = ("cup", "copa", "coupe", "pokal", "trophy", "shield",
                    "champions league", "europa league", "conference league",
                    "supercup", "community")
    return not any(kw in comp_lower for kw in cup_keywords)
