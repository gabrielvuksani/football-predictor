"""
Consolidated Elo rating core — single source of truth for all Elo logic.

Four callers previously had their own inline Elo implementations with divergent
constants.  This module provides configurable functions that each caller can
parameterise while keeping the maths in one place.

Typical usage::

    from footy.models.elo_core import elo_expected, elo_predict, elo_update

    # In-memory dict variant (used by v3, pipeline backtests, meta training)
    probs = elo_predict(ratings, home, away)
    elo_update(ratings, home, away, hg, ag)

    # Or with custom constants
    probs = elo_predict(ratings, home, away, home_adv=65.0, draw_base=0.26)
"""
from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------

def elo_expected(r_home: float, r_away: float) -> float:
    """Standard Elo expected-score for the home side."""
    return 1.0 / (1.0 + 10.0 ** (-(r_home - r_away) / 400.0))


def elo_draw_prob(
    r_home_adj: float,
    r_away: float,
    draw_base: float = 0.26,
    draw_sensitivity: float = 0.06,
    draw_min: float = 0.18,
    draw_max: float = 0.34,
) -> float:
    """Dynamic draw probability — rises when teams are closer in rating."""
    diff = abs(r_home_adj - r_away)
    p = draw_base + draw_sensitivity * math.exp(-diff / 200.0)
    return max(draw_min, min(draw_max, p))


def elo_predict(
    ratings: dict[str, float],
    home: str,
    away: str,
    *,
    default_rating: float = 1500.0,
    home_adv: float = 60.0,
    draw_base: float = 0.26,
    dynamic_draw: bool = False,
) -> tuple[float, float, float]:
    """
    Predict match probabilities from an in-memory ratings dict.

    When *dynamic_draw* is False the draw probability is fixed at *draw_base*.
    When True, it uses ``elo_draw_prob`` which adjusts for rating closeness.

    Returns (p_home, p_draw, p_away) summing to 1.0.
    """
    rh = ratings.get(home, default_rating) + home_adv
    ra = ratings.get(away, default_rating)

    e_home = elo_expected(rh, ra)

    if dynamic_draw:
        p_draw = elo_draw_prob(rh, ra, draw_base=draw_base)
    else:
        p_draw = draw_base

    p_home = e_home * (1.0 - p_draw)
    p_away = (1.0 - e_home) * (1.0 - p_draw)
    s = p_home + p_draw + p_away
    return (p_home / s, p_draw / s, p_away / s)


def elo_predict_with_diff(
    ratings: dict[str, float],
    home: str,
    away: str,
    *,
    default_rating: float = 1500.0,
    home_adv: float = 60.0,
    draw_base: float = 0.26,
) -> tuple[float, float, float, float]:
    """Like ``elo_predict`` but also returns the raw elo_diff (no home-adv)."""
    probs = elo_predict(
        ratings, home, away,
        default_rating=default_rating,
        home_adv=home_adv,
        draw_base=draw_base,
    )
    elo_diff = float(ratings.get(home, default_rating) - ratings.get(away, default_rating))
    return (*probs, elo_diff)


def elo_update(
    ratings: dict[str, float],
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    *,
    default_rating: float = 1500.0,
    home_adv: float = 60.0,
    k: float = 20.0,
) -> None:
    """Update an in-memory ratings dict after a match result (fixed K).

    Args:
        ratings: Dictionary mapping team names to Elo ratings
        home: Home team name
        away: Away team name
        home_goals: Goals scored by home team
        away_goals: Goals scored by away team
        default_rating: Default rating for teams not in dict
        home_adv: Home advantage adjustment
        k: K-factor for rating changes

    Raises:
        ValueError: If inputs are invalid
    """
    # Input validation
    if not isinstance(ratings, dict):
        raise ValueError("ratings must be a dictionary")

    if not home or not away:
        raise ValueError("Team names must be non-empty strings")

    if home == away:
        raise ValueError("Home and away teams must be different")

    if home_goals < 0 or away_goals < 0:
        raise ValueError("Goals must be non-negative integers")

    if not isinstance(home_goals, int) or not isinstance(away_goals, int):
        raise ValueError("Goals must be integers")

    if default_rating < 800 or default_rating > 3000:
        raise ValueError(f"default_rating {default_rating} outside reasonable range [800, 3000]")

    if k <= 0 or k > 100:
        raise ValueError(f"k-factor {k} must be positive and reasonable (0 < k <= 100)")

    rh0 = ratings.get(home, default_rating)
    ra0 = ratings.get(away, default_rating)

    # Validate existing ratings
    if not (800 <= rh0 <= 3000) or not (800 <= ra0 <= 3000):
        raise ValueError(f"Existing ratings outside reasonable range: home={rh0}, away={ra0}")

    exp_home = elo_expected(rh0 + home_adv, ra0)

    if home_goals > away_goals:
        s_home = 1.0
    elif home_goals == away_goals:
        s_home = 0.5
    else:
        s_home = 0.0

    delta = k * (s_home - exp_home)
    new_home_rating = rh0 + delta
    new_away_rating = ra0 - delta

    # Validate new ratings are finite
    if not (800 <= new_home_rating <= 3000) or not (800 <= new_away_rating <= 3000):
        raise ValueError(f"Rating update would produce out-of-range values: home={new_home_rating}, away={new_away_rating}")

    ratings[home] = new_home_rating
    ratings[away] = new_away_rating


def dynamic_k(
    match_count: int,
    goal_diff: int,
    k_base: float = 20.0,
    convergence_matches: int = 60,
) -> float:
    """
    Dynamic K-factor used by ``elo.py`` and ``council.py``:
    - Higher for teams with few matches (fast convergence)
    - Scales with goal difference (decisive results are more informative)

    Args:
        match_count: Number of matches played by the team
        goal_diff: Difference in goals (home_goals - away_goals)
        k_base: Base K-factor
        convergence_matches: Number of matches needed for full convergence

    Returns:
        Adjusted K-factor

    Raises:
        ValueError: If inputs are invalid
    """
    # Input validation
    if match_count < 0:
        raise ValueError("match_count must be non-negative")

    if not isinstance(goal_diff, int):
        raise ValueError("goal_diff must be an integer")

    if k_base <= 0 or k_base > 100:
        raise ValueError(f"k_base {k_base} must be positive and <= 100")

    if convergence_matches < 1:
        raise ValueError("convergence_matches must be at least 1")

    convergence = 1.0 + max(0.0, 1.0 - match_count / convergence_matches)

    gd = abs(goal_diff)
    if gd <= 1:
        gd_mult = 1.0
    elif gd == 2:
        gd_mult = 1.15
    elif gd == 3:
        gd_mult = 1.30
    else:
        gd_mult = min(1.50, 1.30 + 0.05 * (gd - 3))

    result = k_base * convergence * gd_mult
    assert result > 0, "K-factor must be positive"
    return result
