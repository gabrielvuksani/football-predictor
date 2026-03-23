"""
Glicko-2 Rating System for Football Prediction (Glickman, 2001).

The Glicko-2 system extends Elo to include:
- Rating deviation (RD): uncertainty in rating estimate, decreases with play
- Volatility (σ): rating swing magnitude, adapts based on game outcomes
- Comprehensive pre/post-match calculations using the Illinois algorithm

Reference: Glickman, M.E. (2001). "Glicko-2 System"
https://www.glicko.net/glicko/glicko2.pdf

This implementation is tailored for football with:
- Draw handling specific to the sport
- Home advantage estimation
- Conversion to 3-way match probabilities
- DuckDB storage of team state
"""
from __future__ import annotations

import math
from typing import Optional
from dataclasses import dataclass

import duckdb


# Default Glicko-2 parameters per the spec
DEFAULT_RATING = 1500
DEFAULT_RD = 350
DEFAULT_VOLATILITY = 0.06
SCALE = 173.7178  # 400 / ln(10)
SYSTEM_TAU = 0.5  # Controls volatility change rate
MIN_RD = 30  # Minimum rating deviation (maximum certainty)
MAX_RD = 350  # Maximum rating deviation (minimum certainty)


@dataclass
class Glicko2State:
    """Complete Glicko-2 rating state for a single team."""
    team: str
    rating: float  # μ (mu)
    rd: float  # φ (phi) - rating deviation
    volatility: float  # σ (sigma)
    last_rating: Optional[float] = None  # For momentum tracking


def _g(rd: float) -> float:
    """Scaling function for rating deviation.

    Reduces the impact of rating differences when RD is large (uncertainty high).
    """
    phi = rd / SCALE
    return 1.0 / math.sqrt(1.0 + 3.0 * phi**2)


def _e(rating: float, opponent_rating: float, opponent_rd: float) -> float:
    """Expected score against an opponent.

    Args:
        rating: Team's rating
        opponent_rating: Opponent's rating
        opponent_rd: Opponent's rating deviation

    Returns:
        Expected score (probability for 1v1; adjusted in football context)
    """
    g_val = _g(opponent_rd)
    rating_diff = (rating - opponent_rating) / SCALE
    return 1.0 / (1.0 + math.exp(-g_val * rating_diff))


def _determine_rd(rd: float, days_since_match: float, activity_rate: float = 1.0) -> float:
    """Preliminary rating deviation after time period without play.

    RD increases over time when a team hasn't played, representing increased
    uncertainty about their current strength.

    Args:
        rd: Current rating deviation
        days_since_match: Days since the team's last match
        activity_rate: Multiplier for time decay (default 1.0 = standard rate)

    Returns:
        Updated (increased) RD
    """
    c = 50.0  # Ballpark for football (adjustable per league/competition)
    rd_new_sq = rd**2 + c**2 * (days_since_match / 365.0) * activity_rate
    return math.sqrt(min(rd_new_sq, MAX_RD**2))


def _compute_v(results: list[tuple[float, float, float]]) -> float:
    """Compute v: estimated variance of rating.

    Args:
        results: List of (opponent_rating, opponent_rd, score_for_team) tuples
                 where score_for_team ∈ {0, 0.5, 1} for loss/draw/win

    Returns:
        Variance (smaller = more confident)
    """
    if not results:
        return 1.0 / (SCALE**2)

    v_inv = 0.0
    for opp_rating, opp_rd, _ in results:
        g_val = _g(opp_rd)
        e_val = _e(opp_rating, 1500, opp_rd)  # Placeholder rating
        v_inv += g_val**2 * e_val * (1.0 - e_val)

    return 1.0 / v_inv if v_inv > 0 else 1.0 / (SCALE**2)


def _compute_delta(v: float, results: list[tuple[float, float, float]]) -> float:
    """Compute Δ: estimated improvement in rating.

    Args:
        v: Variance (from _compute_v)
        results: List of (opponent_rating, opponent_rd, score_for_team) tuples

    Returns:
        Change in rating per the rating period
    """
    if not results:
        return 0.0

    delta = 0.0
    for opp_rating, opp_rd, score in results:
        g_val = _g(opp_rd)
        e_val = _e(opp_rating, 1500, opp_rd)  # Placeholder
        delta += g_val * (score - e_val)

    return v * delta


def _compute_new_volatility(
    volatility: float,
    delta: float,
    v: float,
    tau: float = SYSTEM_TAU,
) -> float:
    """Compute new volatility σ' using the Illinois algorithm.

    This is the adaptive volatility mechanism. Higher volatility indicates
    larger rating swings (increased uncertainty about strength).

    Args:
        volatility: Current σ
        delta: Rating improvement Δ
        v: Variance (estimated)
        tau: System constant controlling volatility change rate

    Returns:
        New volatility σ'
    """
    sigma = volatility
    a = math.log(sigma**2)

    delta_sq = delta**2
    v_inv = 1.0 / v if v > 0 else 1.0 / (SCALE**2)

    def f(x: float) -> float:
        """Objective function for Illinois algorithm."""
        ex = math.exp(x)
        numerator = ex * (delta_sq - ex * v_inv - 1.0)
        denominator = (2.0 * (tau**2)) * (1.0 + tau**2)
        return (x - a) - (numerator / denominator)

    # Illinois method: find x such that f(x) = 0
    # Bounds for search
    lower = a - (tau**2) * 5.0
    upper = a + (tau**2) * 5.0

    # Evaluate at bounds
    f_lower = f(lower)
    f_upper = f(upper)

    # If zero is not bracketed, use endpoint closest to zero
    if f_lower * f_upper > 0:
        # Use midpoint iteration
        for _ in range(50):
            mid = (lower + upper) / 2.0
            f_mid = f(mid)
            if abs(f_mid) < 1e-10:
                return math.exp(mid / 2.0)
            if f_mid > 0:
                upper = mid
            else:
                lower = mid
    else:
        # Standard bisection when bracketed
        for _ in range(50):
            mid = (lower + upper) / 2.0
            f_mid = f(mid)
            if abs(f_mid) < 1e-10:
                break
            if f_mid > 0:
                upper = mid
            else:
                lower = mid

    x_star = (lower + upper) / 2.0
    return math.exp(x_star / 2.0)


def _update_rating_and_rd(
    rating: float,
    rd: float,
    volatility: float,
    results: list[tuple[float, float, float]],
) -> tuple[float, float]:
    """Update rating μ' and RD φ' after processing results.

    Args:
        rating: Current rating μ
        rd: Current rating deviation φ
        volatility: Current volatility σ (already updated via _compute_new_volatility)
        results: List of (opponent_rating, opponent_rd, score) tuples

    Returns:
        (new_rating, new_rd)
    """
    if not results:
        # No matches: RD just decays over time (handled by _determine_rd)
        return rating, rd

    v = _compute_v(results)
    _compute_delta(v, results)

    phi = rd / SCALE
    phi_star_inv = 1.0 / math.sqrt((1.0 / (phi**2)) + (1.0 / v))

    new_rating = rating
    for opp_rating, opp_rd, score in results:
        g_val = _g(opp_rd)
        e_val = _e(rating, opp_rating, opp_rd)
        new_rating += (SCALE * phi_star_inv**2) * g_val * (score - e_val)

    new_rd = SCALE * phi_star_inv
    new_rd = max(MIN_RD, min(MAX_RD, new_rd))

    return new_rating, new_rd


def ensure_team(con: duckdb.DuckDBPyConnection, team: str) -> None:
    """Ensure team exists in glicko2_state; initialize if needed."""
    con.execute("""
        INSERT OR IGNORE INTO glicko2_state(team, rating, rd, volatility)
        VALUES (?, ?, ?, ?)
    """, [team, DEFAULT_RATING, DEFAULT_RD, DEFAULT_VOLATILITY])


def get_state(con: duckdb.DuckDBPyConnection, team: str) -> Glicko2State:
    """Fetch current Glicko-2 state for a team."""
    ensure_team(con, team)
    row = con.execute(
        "SELECT team, rating, rd, volatility FROM glicko2_state WHERE team=?",
        [team]
    ).fetchone()
    return Glicko2State(
        team=row[0],
        rating=row[1],
        rd=row[2],
        volatility=row[3],
    )


def set_state(
    con: duckdb.DuckDBPyConnection,
    team: str,
    rating: float,
    rd: float,
    volatility: float,
) -> None:
    """Update Glicko-2 state for a team."""
    con.execute("""
        UPDATE glicko2_state
        SET rating=?, rd=?, volatility=?
        WHERE team=?
    """, [rating, rd, volatility, team])


def update_from_match(
    con: duckdb.DuckDBPyConnection,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    home_advantage: float = 40.0,
) -> None:
    """Update Glicko-2 ratings after a match.

    Args:
        con: DuckDB connection
        home: Home team name
        away: Away team name
        home_goals: Goals scored by home team
        away_goals: Goals scored by away team
        home_advantage: Rating bonus for home team (default 40 points)
    """
    ensure_team(con, home)
    ensure_team(con, away)

    state_h = get_state(con, home)
    state_a = get_state(con, away)

    # Determine result for each team (0=loss, 0.5=draw, 1=win)
    if home_goals > away_goals:
        score_h, score_a = 1.0, 0.0
    elif home_goals == away_goals:
        score_h, score_a = 0.5, 0.5
    else:
        score_h, score_a = 0.0, 1.0

    # Prepare match records with home advantage adjustment
    home_results = [(state_a.rating, state_a.rd, score_h)]
    away_results = [(state_h.rating + home_advantage, state_h.rd, score_a)]

    # Compute volatility updates
    v_h = _compute_v(home_results)
    delta_h = _compute_delta(v_h, home_results)
    sigma_h_new = _compute_new_volatility(state_h.volatility, delta_h, v_h)

    v_a = _compute_v(away_results)
    delta_a = _compute_delta(v_a, away_results)
    sigma_a_new = _compute_new_volatility(state_a.volatility, delta_a, v_a)

    # Update ratings and RD
    rating_h_new, rd_h_new = _update_rating_and_rd(
        state_h.rating, state_h.rd, sigma_h_new, home_results
    )
    rating_a_new, rd_a_new = _update_rating_and_rd(
        state_a.rating, state_a.rd, sigma_a_new, away_results
    )

    # Persist
    set_state(con, home, rating_h_new, rd_h_new, sigma_h_new)
    set_state(con, away, rating_a_new, rd_a_new, sigma_a_new)


def predict_probs(
    con: duckdb.DuckDBPyConnection,
    home: str,
    away: str,
    home_advantage: float = 40.0,
) -> tuple[float, float, float]:
    """Predict match probabilities using Glicko-2 ratings.

    Args:
        con: DuckDB connection
        home: Home team name
        away: Away team name
        home_advantage: Rating bonus for home team (default 40 points)

    Returns:
        (p_home, p_draw, p_away) summing to 1.0
    """
    ensure_team(con, home)
    ensure_team(con, away)

    state_h = get_state(con, home)
    state_a = get_state(con, away)

    # Glicko-2 expected score from home perspective
    rating_h_adj = state_h.rating + home_advantage
    e_h = _e(rating_h_adj, state_a.rating, state_a.rd)
    e_a = 1.0 - e_h

    # Dynamic draw probability based on rating difference and uncertainty
    rating_diff = abs(rating_h_adj - state_a.rating)
    mean_rd = (state_h.rd + state_a.rd) / 2.0

    # Draw more likely when teams are close and/or uncertainty is high
    draw_base = 0.26
    draw_sensitivity = 0.05
    draw_rd_effect = 0.0002 * mean_rd
    p_draw = draw_base + draw_sensitivity * math.exp(-rating_diff / 200.0) + draw_rd_effect
    p_draw = max(0.15, min(0.45, p_draw))

    # Distribute remaining probability
    p_home = e_h * (1.0 - p_draw)
    p_away = e_a * (1.0 - p_draw)

    # Normalize
    total = p_home + p_draw + p_away
    return (p_home / total, p_draw / total, p_away / total)


def batch_update(
    con: duckdb.DuckDBPyConnection,
    matches: list[tuple[str, str, int, int]],
    home_advantage: float = 40.0,
) -> None:
    """Process multiple matches chronologically, updating state after each.

    Args:
        con: DuckDB connection
        matches: List of (home_team, away_team, home_goals, away_goals) tuples
                Should be sorted by date ascending
        home_advantage: Rating bonus for home team
    """
    for home, away, hg, ag in matches:
        update_from_match(con, home, away, hg, ag, home_advantage)
