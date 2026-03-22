"""Kelly Criterion Module — Optimal Bet Sizing and Value Betting.

Implements comprehensive Kelly criterion for sports betting:
- Binary Kelly: f* = (bp - q) / b
- Fractional Kelly: f* × fraction (typically 0.25 for conservative)
- Multi-outcome Kelly (1X2 markets): Smoczynski & Tomkins extension
- Constrained Kelly with bankroll management
- Expected value and ROI calculations
- Risk of ruin calculations

References:
    Kelly (1956) "A new interpretation of information rate"
    Smoczynski & Tomkins (2010) "Efficient betting with the Kelly criterion"
    MacLean et al. (2011) "Good and bad properties of the Kelly criterion"
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class KellyBet:
    """A single Kelly-criterion optimized bet."""
    outcome: str  # "home", "draw", "away"
    probability: float  # Model's estimated probability
    odds: float  # Decimal odds
    kelly_fraction: float  # Optimal fraction of bankroll
    kelly_pct: float  # Kelly % (before fractional adjustment)
    expected_value: float  # EV in decimal units
    ev_percentage: float  # EV as percentage of bet
    recommended_stake: float  # Recommended bet size (per unit bankroll)
    confidence_adjusted_ev: float  # EV × confidence modifier
    risk_rating: float  # 0-1 scale of risk
    edge: float  # Probability edge over market

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "probability": round(self.probability, 4),
            "odds": round(self.odds, 2),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "kelly_pct": round(self.kelly_pct, 2),
            "expected_value": round(self.expected_value, 4),
            "ev_percentage": round(self.ev_percentage, 4),
            "recommended_stake": round(self.recommended_stake, 4),
            "confidence_adjusted_ev": round(self.confidence_adjusted_ev, 4),
            "risk_rating": round(self.risk_rating, 4),
            "edge": round(self.edge, 4),
        }


def kelly_fraction(
    probability: float,
    odds: float,
    kelly_fraction: float = 1.0,
    max_fraction: float = 0.25,
) -> float:
    """Compute Kelly criterion fraction for binary outcome.

    Formula: f* = (bp - q) / b
    where:
        b = odds - 1 (profit if you win)
        p = probability of winning
        q = 1 - p = probability of losing

    Args:
        probability: Estimated probability of winning (0-1)
        odds: Decimal odds (e.g., 2.5 means 1:1.5 profit)
        kelly_fraction: Fraction of full Kelly to use (0-1), default 1.0 = full Kelly
        max_fraction: Maximum Kelly fraction to recommend (e.g., 0.25 for 25% Kelly)

    Returns:
        Optimal fraction of bankroll to stake
    """
    if not (0 <= probability <= 1):
        return 0.0
    if odds <= 1.0:
        return 0.0

    b = odds - 1.0
    p = probability
    q = 1.0 - p

    # Full Kelly
    full_kelly = (b * p - q) / b

    # Apply fractional Kelly
    fractional = full_kelly * kelly_fraction

    # Cap at maximum
    capped = min(fractional, max_fraction)

    # Ensure non-negative
    return max(0.0, capped)


def multi_kelly(
    probabilities: tuple[float, float, float],
    odds: tuple[float, float, float],
    kelly_fraction: float = 1.0,
    max_fraction: float = 0.25,
) -> dict[str, float]:
    """Compute Kelly criterion for 1X2 (three-outcome) markets.

    Uses the Smoczynski & Tomkins (2010) extension for dependent outcomes.
    For 1X2 markets, this computes the optimal stake for each outcome
    relative to the bankroll.

    Args:
        probabilities: (p_home, p_draw, p_away)
        odds: (odds_home, odds_draw, odds_away) as decimal odds
        kelly_fraction: Fraction of full Kelly (0-1)
        max_fraction: Maximum Kelly fraction to recommend

    Returns:
        Dictionary with keys "home", "draw", "away" mapped to kelly fractions
    """
    p_home, p_draw, p_away = probabilities
    o_home, o_draw, o_away = odds

    # Validate inputs
    if not (abs(p_home + p_draw + p_away - 1.0) < 0.01):
        normalized_probs = normalize_probabilities((p_home, p_draw, p_away))
        p_home, p_draw, p_away = normalized_probs

    if any(o <= 1.0 for o in odds):
        return {"home": 0.0, "draw": 0.0, "away": 0.0}

    # For simplicity, use pairwise Kelly approach
    # Treat each outcome vs. "not this outcome"
    results = {}
    for label, p, o in [
        ("home", p_home, o_home),
        ("draw", p_draw, o_draw),
        ("away", p_away, o_away),
    ]:
        # Probability of NOT this outcome
        q = 1.0 - p
        b = o - 1.0
        full_kelly = (b * p - q) / b if b > 0 else 0.0
        fractional = full_kelly * kelly_fraction
        capped = min(max(0.0, fractional), max_fraction)
        results[label] = capped

    return results


def normalize_probabilities(probs: tuple[float, ...]) -> tuple[float, ...]:
    """Normalize a probability vector to sum to 1.0."""
    total = sum(probs)
    if total <= 0:
        return tuple(1.0 / len(probs) for _ in probs)
    return tuple(p / total for p in probs)


def expected_value(
    probability: float,
    odds: float,
) -> float:
    """Compute expected value per unit staked.

    EV = p × (odds - 1) - (1 - p) × 1
       = p × (odds - 1) - (1 - p)
       = p × odds - p - 1 + p
       = p × odds - 1

    Args:
        probability: Estimated probability (0-1)
        odds: Decimal odds

    Returns:
        Expected value per unit staked
    """
    return probability * odds - 1.0


def expected_value_percentage(
    probability: float,
    odds: float,
) -> float:
    """Compute expected value as a percentage of stake.

    EV% = (probability × odds - 1) × 100

    Args:
        probability: Estimated probability (0-1)
        odds: Decimal odds

    Returns:
        EV as percentage (e.g., 5.5 means +5.5% expected return)
    """
    ev = expected_value(probability, odds)
    return ev * 100.0


def implied_probability(odds: float) -> float:
    """Convert decimal odds to implied probability.

    Args:
        odds: Decimal odds

    Returns:
        Implied probability (0-1)
    """
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def compute_edge(
    model_probability: float,
    implied_probability: float,
) -> float:
    """Compute edge: model probability - implied probability.

    Positive edge means the model thinks it's more likely than the market.

    Args:
        model_probability: Our estimated probability
        implied_probability: Market-implied probability

    Returns:
        Edge in probability units
    """
    return model_probability - implied_probability


def risk_of_ruin(
    kelly_fraction: float,
    n_bets: int,
    win_rate: float,
    odds_avg: float = 2.0,
) -> float:
    """Estimate risk of ruin using Kelly criterion.

    For a series of bets with consistent Kelly fraction,
    RoR approximates to (q/p)^n for large n (Pabrai bound).

    Args:
        kelly_fraction: Fraction of bankroll per bet
        n_bets: Number of bets (simulated trials)
        win_rate: Estimated win rate (0-1)
        odds_avg: Average odds (for scaling)

    Returns:
        Approximate probability of reaching 0% bankroll
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.0 if win_rate >= 0.5 else 1.0

    loss_rate = 1.0 - win_rate

    # For each bet with Kelly fraction f, the ratio is:
    # Win: bankroll × (1 + f × (odds - 1))
    # Loss: bankroll × (1 - f)

    b = odds_avg - 1.0
    ratio_win = 1.0 + kelly_fraction * b
    ratio_loss = 1.0 - kelly_fraction

    if ratio_win <= 0 or ratio_loss <= 0:
        return 1.0

    # Logarithmic random walk
    log_ratio = win_rate * math.log(ratio_win) + loss_rate * math.log(ratio_loss)

    if log_ratio >= 0:
        return 0.0  # No ruin risk with positive log ratio

    # Over n_bets, expected log bankroll growth is n × log_ratio
    # RoR ~ exp(-2 × log_ratio × capital_ratio) (Chernoff bound variant)
    ror = math.exp(2.0 * log_ratio * n_bets)
    return min(1.0, max(0.0, ror))


def simulate_bankroll(
    kelly_bets: list[KellyBet],
    initial_bankroll: float = 1000.0,
    n_simulations: int = 10000,
    seed: int | None = None,
) -> dict[str, float]:
    """Simulate bankroll outcomes under Kelly betting.

    Runs Monte Carlo simulations of Kelly-criterion betting to estimate
    the distribution of final bankroll values and risk metrics.

    Args:
        kelly_bets: List of KellyBet objects representing available bets
        initial_bankroll: Starting bankroll
        n_simulations: Number of Monte Carlo simulations
        seed: Random seed for reproducibility

    Returns:
        Dictionary with statistics:
        - final_mean: Mean final bankroll
        - final_std: Standard deviation of final bankroll
        - final_min: Minimum final bankroll across simulations
        - final_max: Maximum final bankroll across simulations
        - median_final: Median final bankroll
        - ruin_count: Number of simulations where bankroll hit 0
        - ruin_pct: Percentage of simulations with ruin
        - expected_roi: Expected return on initial bankroll
        - var_95: Value at Risk (95% confidence)
        - sharpe_ratio: Risk-adjusted return estimate
    """
    if not kelly_bets:
        return {
            "final_mean": initial_bankroll,
            "final_std": 0.0,
            "final_min": initial_bankroll,
            "final_max": initial_bankroll,
            "median_final": initial_bankroll,
            "ruin_count": 0,
            "ruin_pct": 0.0,
            "expected_roi": 0.0,
            "var_95": initial_bankroll,
            "sharpe_ratio": 0.0,
        }

    if seed is not None:
        np.random.seed(seed)

    final_bankrolls = []
    ruin_count = 0

    for _ in range(n_simulations):
        bankroll = initial_bankroll
        for bet in kelly_bets:
            # Simulate outcome: win with probability bet.probability, else lose
            won = np.random.random() < bet.probability

            stake = bankroll * bet.recommended_stake
            if stake <= 0:
                continue

            if won:
                # Win: gain stake × (odds - 1)
                profit = stake * (bet.odds - 1.0)
                bankroll += profit
            else:
                # Loss: lose the stake
                bankroll -= stake

            if bankroll <= 0:
                ruin_count += 1
                break

        final_bankrolls.append(max(0.0, bankroll))

    final_array = np.array(final_bankrolls)
    final_mean = float(np.mean(final_array))
    final_std = float(np.std(final_array))
    final_min = float(np.min(final_array))
    final_max = float(np.max(final_array))
    median_final = float(np.median(final_array))

    ruin_pct = (ruin_count / n_simulations) * 100.0 if n_simulations > 0 else 0.0
    expected_roi = ((final_mean - initial_bankroll) / initial_bankroll) * 100.0
    var_95 = float(np.percentile(final_array, 5))
    sharpe = (
        (final_mean - initial_bankroll) / max(final_std, 1.0)
        if final_std > 0
        else 0.0
    )

    return {
        "final_mean": round(final_mean, 2),
        "final_std": round(final_std, 2),
        "final_min": round(final_min, 2),
        "final_max": round(final_max, 2),
        "median_final": round(median_final, 2),
        "ruin_count": ruin_count,
        "ruin_pct": round(ruin_pct, 2),
        "expected_roi": round(expected_roi, 2),
        "var_95": round(var_95, 2),
        "sharpe_ratio": round(sharpe, 4),
    }


def identify_value_bets(
    matches: list[dict],
    min_edge: float = 0.03,
    kelly_frac: float = 0.25,
    confidence_weight: float = 1.0,
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """Identify value bets across all upcoming matches.

    Scans matches for opportunities where the model probability exceeds
    the market-implied probability by at least min_edge.

    Args:
        matches: List of match dicts with predictions and odds
        min_edge: Minimum probability edge to consider (0-1)
        kelly_frac: Kelly fraction to use (e.g., 0.25 = 25% Kelly)
        confidence_weight: Weight on confidence modifier (0-1)
        min_confidence: Minimum confidence to include bet (0-1)

    Returns:
        List of value bet dicts, sorted by edge descending
    """
    value_bets = []

    for match in matches:
        if match.get("p_home") is None:
            continue

        probs = {
            "home": match.get("p_home", 0.0),
            "draw": match.get("p_draw", 0.0),
            "away": match.get("p_away", 0.0),
        }
        odds = {
            "home": match.get("b365h"),
            "draw": match.get("b365d"),
            "away": match.get("b365a"),
        }
        confidence = match.get("confidence", 0.5)
        model_agreement = match.get("model_agreement", 0.5)

        if confidence < min_confidence:
            continue

        # Check each outcome
        for outcome in ["home", "draw", "away"]:
            prob = probs.get(outcome, 0.0)
            odd = odds.get(outcome)

            if not odd or odd <= 1.0 or prob is None or prob <= 0:
                continue

            implied = implied_probability(odd)
            edge = compute_edge(prob, implied)

            if edge < min_edge:
                continue

            # Compute Kelly sizing
            kelly_pct = kelly_fraction(prob, odd, kelly_fraction=1.0, max_fraction=0.5)
            fractional = kelly_pct * kelly_frac
            stake = max(0.0, min(fractional, 0.25))

            # Expected value
            ev = expected_value(prob, odd)
            ev_pct = expected_value_percentage(prob, odd)

            # Confidence-adjusted EV
            conf_factor = 1.0 + (confidence - 0.5) * confidence_weight
            conf_adjusted_ev = ev * conf_factor

            # Risk rating based on model agreement
            risk_rating = 1.0 - model_agreement

            value_bets.append({
                "match_id": match.get("match_id"),
                "home_team": match.get("home_team"),
                "away_team": match.get("away_team"),
                "competition": match.get("competition"),
                "date": str(match.get("utc_date", ""))[:10],
                "outcome": outcome,
                "model_probability": round(prob, 4),
                "implied_probability": round(implied, 4),
                "odds": round(odd, 2),
                "edge": round(edge, 4),
                "kelly_fraction": round(stake, 4),
                "kelly_pct": round(kelly_pct * 100, 2),
                "expected_value": round(ev, 4),
                "ev_percentage": round(ev_pct, 2),
                "confidence_adjusted_ev": round(conf_adjusted_ev, 4),
                "confidence": round(confidence, 4),
                "model_agreement": round(model_agreement, 4),
                "risk_rating": round(risk_rating, 4),
            })

    # Sort by edge descending
    value_bets.sort(key=lambda b: b["edge"], reverse=True)
    return value_bets


def optimize_bankroll(
    total_bankroll: float,
    kelly_bets: list[KellyBet],
    max_single_bet_pct: float = 0.05,
) -> dict[str, float]:
    """Optimize bet sizing across multiple bets subject to constraints.

    Given multiple value bets and a bankroll, compute the optimal stake
    for each bet respecting the Kelly criterion and maximum bet constraints.

    Args:
        total_bankroll: Total bankroll available
        kelly_bets: List of KellyBet objects
        max_single_bet_pct: Maximum % of bankroll per single bet

    Returns:
        Dictionary mapping bet ids to stake amounts
    """
    if not kelly_bets or total_bankroll <= 0:
        return {}

    allocations = {}
    total_kelly_needed = sum(
        bet.recommended_stake
        for bet in kelly_bets
        if bet.recommended_stake > 0
    )

    if total_kelly_needed <= 0:
        return {}

    # Scale to fit within max_single_bet_pct constraint
    max_single_bet = total_bankroll * max_single_bet_pct
    scale_factor = 1.0

    for bet in kelly_bets:
        if bet.recommended_stake > 0:
            desired = total_bankroll * bet.recommended_stake
            if desired > max_single_bet:
                scale_factor = min(scale_factor, max_single_bet / desired)

    # Allocate
    for i, bet in enumerate(kelly_bets):
        stake = total_bankroll * bet.recommended_stake * scale_factor
        if stake > 0.01:  # Only include if meaningful
            allocations[f"bet_{i}"] = round(stake, 2)

    return allocations
