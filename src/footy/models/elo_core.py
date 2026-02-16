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
    """Update an in-memory ratings dict after a match result (fixed K)."""
    rh0 = ratings.get(home, default_rating)
    ra0 = ratings.get(away, default_rating)
    exp_home = elo_expected(rh0 + home_adv, ra0)

    if home_goals > away_goals:
        s_home = 1.0
    elif home_goals == away_goals:
        s_home = 0.5
    else:
        s_home = 0.0

    delta = k * (s_home - exp_home)
    ratings[home] = rh0 + delta
    ratings[away] = ra0 - delta


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
    """
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

    return k_base * convergence * gd_mult
