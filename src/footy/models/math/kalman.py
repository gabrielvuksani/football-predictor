"""Kalman filtering for dynamic team strength tracking.

Implements state-space models for tracking team attack and defence strength
as they evolve through a season, with automatic Bayesian updates based
on match outcomes vs. expected performance.

References:
    Rue & Salvesen (2000) "Prediction and retrospective analysis of soccer"
    Koopman & Lit (2015) "A dynamic bivariate Poisson model for association football"
    Harvey (1989) "Forecasting, Structural Time Series and the Kalman Filter"
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class KalmanStrengthState:
    """State vector for Kalman-filter team strength tracker.

    Tracks:
        attack: Team's attacking strength (expected goals scored)
        defence: Team's defensive strength (expected goals against)
        variance: Uncertainty in state estimates
    """
    attack: float = 0.0
    defence: float = 0.0
    variance: float = 1.0


def kalman_attack_defence_update(
    prev: KalmanStrengthState,
    observed_for: float,
    observed_against: float,
    expected_for: float,
    expected_against: float,
    process_var: float = 0.05,
    measurement_var: float = 0.20,
) -> KalmanStrengthState:
    """Single-step Kalman-style update for dynamic team strength.

    The Kalman filter performs optimal Bayesian updating when:
    1. State evolves linearly (here: constant strength with noise)
    2. Measurements are linear functions of state
    3. Noise is Gaussian

    This is used to update team strength estimates after each match,
    automatically balancing prior strength with new evidence.

    Args:
        prev: Previous state (before this match)
        observed_for: Actual goals scored by team
        observed_against: Actual goals conceded by team
        expected_for: Expected goals against this opponent (prior)
        expected_against: Expected goals conceded vs. opponent (prior)
        process_var: How much strength can change per match (≈ 0.05)
        measurement_var: Noise in match observations (≈ 0.20)

    Returns:
        Updated state after this match

    References:
        Rue & Salvesen (2000), Koopman & Lit (2015)
    """
    # Prediction step: account for natural evolution (process noise)
    prior_var = prev.variance + process_var

    # Kalman gain: how much to trust new data vs. prior
    # High measurement_var → trust prior more
    # Low measurement_var → trust new data more
    kalman_gain = prior_var / (prior_var + measurement_var)

    # Innovation (prediction error): how far off we were
    attack_residual = float(observed_for) - float(expected_for)
    defence_residual = float(expected_against) - float(observed_against)

    # Update: move state towards observation proportionally to Kalman gain
    new_attack = prev.attack + kalman_gain * attack_residual
    new_defence = prev.defence + kalman_gain * defence_residual

    # Updated variance: reduced by observation
    new_variance = (1.0 - kalman_gain) * prior_var

    return KalmanStrengthState(
        attack=new_attack,
        defence=new_defence,
        variance=new_variance,
    )


def kalman_batch_update(
    goals_for: list[float],
    goals_against: list[float],
    expected_for: list[float],
    expected_against: list[float],
    process_var: float = 0.05,
    measurement_var: float = 0.20,
) -> list[KalmanStrengthState]:
    """Run Kalman filter over a sequence of matches for one team.

    Processes an entire match history, updating strength estimates
    after each game. Useful for tracking strength evolution.

    Args:
        goals_for: List of goals scored in each match
        goals_against: List of goals conceded in each match
        expected_for: Expected goals (xG) in each match
        expected_against: Expected goals against in each match
        process_var: Process noise (strength evolution)
        measurement_var: Measurement noise (match randomness)

    Returns:
        List of states: [initial_state, state_after_match_1, ..., state_after_match_N]
        Length is N+1, where N is number of matches

    Raises:
        ValueError: If input lists have different lengths
    """
    n = len(goals_for)
    if not (len(goals_against) == n and len(expected_for) == n and len(expected_against) == n):
        raise ValueError(
            f"All input lists must have equal length, got {n}, "
            f"{len(goals_against)}, {len(expected_for)}, {len(expected_against)}"
        )

    states = [KalmanStrengthState()]
    for gf, ga, ef, ea in zip(goals_for, goals_against, expected_for, expected_against):
        new_state = kalman_attack_defence_update(
            states[-1], gf, ga, ef, ea, process_var, measurement_var,
        )
        states.append(new_state)

    return states
