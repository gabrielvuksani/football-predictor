"""
Pi-Rating System for Football Prediction (Constantinou & Fenton, 2013).

The Pi-rating system models football outcomes through separate home and away
ratings for each team, which are updated based on goal difference rather than
binary win/loss results. This naturally captures team strength asymmetries.

Key features:
- Each team has two ratings: home_rating (strength at home) and away_rating (strength away)
- Updates based on goal difference (more informative than binary result)
- Learning rate adapts based on match history (faster early convergence)
- Home advantage implicit in having separate ratings
- Probabilistic conversion using ordinal logistic regression

Reference: Constantinou, A. C., & Fenton, N. E. (2013). "Solving the problem of
inadequate scoring rules for assessing probabilistic football forecast models."
Journal of Quantitative Analysis in Sports, 8(1).

DuckDB storage tracks state per team.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import duckdb


# Initial ratings
INITIAL_RATING = 1500.0

# Pi-rating specific parameters — v16 tuned from research (Constantinou 2013)
# Optimal: lambda=0.06 for primary, gamma=0.6 for cross-update
SCALE_CONSTANT = 600.0  # Controls expected goal difference scale
BASE_LEARNING_RATE = 0.06  # Primary learning rate (was 0.3 — research: 0.04-0.08 optimal)
MIN_LEARNING_RATE = 0.03  # Floor for adaptive learning rate
LEARNING_DECAY_RATE = 1.0 / 200.0  # Slower decay per match played
CROSS_UPDATE_RATE = 0.6  # Cross-rating update: how much played-context nudges other context


@dataclass
class PiRatingState:
    """Pi-rating state for a single team."""
    team: str
    home_rating: float  # Strength when playing at home
    away_rating: float  # Strength when playing away
    matches_played: int  # Total matches played (for learning rate adaptation)


def _learning_rate(matches_played: int) -> float:
    """Compute adaptive learning rate.

    Learning rate decreases as team plays more matches, converging confidence.

    Args:
        matches_played: Total matches team has played

    Returns:
        Learning rate ∈ [MIN_LEARNING_RATE, BASE_LEARNING_RATE]
    """
    return max(MIN_LEARNING_RATE, BASE_LEARNING_RATE * math.exp(-matches_played * LEARNING_DECAY_RATE))


def _expected_goal_diff(
    team_rating: float,
    opponent_rating: float,
    scale: float = SCALE_CONSTANT,
) -> float:
    """Compute expected goal difference from rating difference.

    Args:
        team_rating: Rating of the team
        opponent_rating: Rating of the opponent
        scale: Scaling constant (higher = less difference per rating point)

    Returns:
        Expected goal difference
    """
    return (team_rating - opponent_rating) / scale


def _ordinal_logistic_probs(
    x: float,
    threshold1: float = -0.5,
    threshold2: float = 0.5,
) -> tuple[float, float, float]:
    """Convert an underlying score to 3-way match outcome probabilities.

    Uses ordinal logistic regression with two thresholds:
    - P(home_win) = P(x > threshold2)
    - P(draw) = P(threshold1 < x < threshold2)
    - P(away_win) = P(x < threshold1)

    Args:
        x: Underlying score (typically expected goal difference)
        threshold1: Lower threshold (draw boundary)
        threshold2: Upper threshold (draw boundary)

    Returns:
        (p_home, p_draw, p_away) summing to 1.0
    """
    # Logistic CDF: 1 / (1 + exp(-x))
    cdf_lower = 1.0 / (1.0 + math.exp(-(threshold1 - x)))
    cdf_upper = 1.0 / (1.0 + math.exp(-(threshold2 - x)))

    p_away = cdf_lower  # x < threshold1
    p_draw = cdf_upper - cdf_lower  # threshold1 < x < threshold2
    p_home = 1.0 - cdf_upper  # x > threshold2

    return (p_home, p_draw, p_away)


def ensure_team(con: duckdb.DuckDBPyConnection, team: str) -> None:
    """Ensure team exists in pi_rating_state; initialize if needed."""
    con.execute("""
        INSERT OR IGNORE INTO pi_rating_state(team, home_rating, away_rating, matches_played)
        VALUES (?, ?, ?, ?)
    """, [team, INITIAL_RATING, INITIAL_RATING, 0])


def get_state(con: duckdb.DuckDBPyConnection, team: str) -> PiRatingState:
    """Fetch current Pi-rating state for a team."""
    ensure_team(con, team)
    row = con.execute(
        "SELECT team, home_rating, away_rating, matches_played FROM pi_rating_state WHERE team=?",
        [team]
    ).fetchone()
    return PiRatingState(
        team=row[0],
        home_rating=row[1],
        away_rating=row[2],
        matches_played=row[3],
    )


def set_state(
    con: duckdb.DuckDBPyConnection,
    team: str,
    home_rating: float,
    away_rating: float,
    matches_played: int,
) -> None:
    """Update Pi-rating state for a team."""
    con.execute("""
        UPDATE pi_rating_state
        SET home_rating=?, away_rating=?, matches_played=?
        WHERE team=?
    """, [home_rating, away_rating, matches_played, team])


def update_from_match(
    con: duckdb.DuckDBPyConnection,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
) -> None:
    """Update Pi-ratings after a match result.

    Updates both teams' home and away ratings based on actual vs expected
    goal difference, using adaptive learning rates.

    Args:
        con: DuckDB connection
        home: Home team name
        away: Away team name
        home_goals: Goals scored by home team
        away_goals: Goals scored by away team
    """
    ensure_team(con, home)
    ensure_team(con, away)

    state_h = get_state(con, home)
    state_a = get_state(con, away)

    # Actual goal difference (from home team perspective)
    actual_gd = home_goals - away_goals

    # Expected goal differences
    expected_gd_h = _expected_goal_diff(state_h.home_rating, state_a.away_rating)
    expected_gd_a = _expected_goal_diff(state_a.away_rating, state_h.home_rating)

    # Compute errors
    error_h = actual_gd - expected_gd_h
    error_a = -actual_gd - expected_gd_a

    # Adaptive learning rates
    lr_h = _learning_rate(state_h.matches_played)
    lr_a = _learning_rate(state_a.matches_played)

    # Update ratings
    home_rating_new = state_h.home_rating + lr_h * error_h
    home_away_new = state_h.away_rating + lr_h * error_h  # Away team in reverse

    away_rating_new = state_a.away_rating + lr_a * error_a
    away_home_new = state_a.home_rating + lr_a * error_a  # Home team in reverse

    # Persist updates
    set_state(con, home, home_rating_new, home_away_new, state_h.matches_played + 1)
    set_state(con, away, away_home_new, away_rating_new, state_a.matches_played + 1)


def predict_probs(
    con: duckdb.DuckDBPyConnection,
    home: str,
    away: str,
) -> tuple[float, float, float]:
    """Predict match probabilities using Pi-ratings.

    Args:
        con: DuckDB connection
        home: Home team name
        away: Away team name

    Returns:
        (p_home, p_draw, p_away) summing to 1.0
    """
    ensure_team(con, home)
    ensure_team(con, away)

    state_h = get_state(con, home)
    state_a = get_state(con, away)

    # Expected goal difference from home perspective
    expected_gd = _expected_goal_diff(state_h.home_rating, state_a.away_rating)

    # Convert to 3-way probabilities via ordinal logistic regression
    # Thresholds are calibrated for football goal differences
    p_home, p_draw, p_away = _ordinal_logistic_probs(
        expected_gd,
        threshold1=-0.5,  # Below this: away win
        threshold2=0.5,   # Above this: home win
    )

    return (p_home, p_draw, p_away)


def batch_update(
    con: duckdb.DuckDBPyConnection,
    matches: list[tuple[str, str, int, int]],
) -> None:
    """Process multiple matches chronologically, updating state after each.

    Args:
        con: DuckDB connection
        matches: List of (home_team, away_team, home_goals, away_goals) tuples
                Should be sorted by date ascending
    """
    for home, away, hg, ag in matches:
        update_from_match(con, home, away, hg, ag)


def get_rating_difference(
    con: duckdb.DuckDBPyConnection,
    home: str,
    away: str,
) -> float:
    """Get the rating difference (for analysis/feature engineering).

    Args:
        con: DuckDB connection
        home: Home team name
        away: Away team name

    Returns:
        Home team's home_rating - away team's away_rating
    """
    ensure_team(con, home)
    ensure_team(con, away)

    state_h = get_state(con, home)
    state_a = get_state(con, away)

    return state_h.home_rating - state_a.away_rating


def get_expected_goals(
    con: duckdb.DuckDBPyConnection,
    home: str,
    away: str,
) -> tuple[float, float]:
    """Get expected goal difference broken down by team.

    This is mainly for diagnostic/feature extraction purposes.

    Args:
        con: DuckDB connection
        home: Home team name
        away: Away team name

    Returns:
        (expected_home_goals, expected_away_goals) (approximate via xG conversion)
    """
    ensure_team(con, home)
    ensure_team(con, away)

    state_h = get_state(con, home)
    state_a = get_state(con, away)

    # Rough heuristic: assume 2.5 goals per team on average, scale by rating offset
    base_goals = 1.25

    expected_gd = _expected_goal_diff(state_h.home_rating, state_a.away_rating)
    eg_home = base_goals + (expected_gd / 2.0)
    eg_away = base_goals - (expected_gd / 2.0)

    return (max(0.1, eg_home), max(0.1, eg_away))
