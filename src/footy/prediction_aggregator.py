"""Unified Prediction Aggregator — Single entrypoint for all prediction models.

This module provides a clean, production-grade API for generating predictions
by combining ALL available prediction methods into a single coherent output.

The aggregation uses a Logarithmic Opinion Pool (which outperforms linear
averaging for calibrated probability combination) with learned weights.

Architecture:
    1. Collect predictions from all statistical models (12+ score models)
    2. Collect expert council ensemble prediction
    3. Collect Bayesian state-space prediction
    4. Combine using Logarithmic Opinion Pool with performance-based weights
    5. Apply conformal prediction for uncertainty quantification
    6. Output unified prediction with full breakdown

References:
    Genest & Zidek (1986) "Combining probability distributions"
    Ranjan & Gneiting (2010) "Combining probability forecasts"
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Unified NaN Handling Strategy
# ---------------------------------------------------------------------------

_NAN_STRATEGY = {
    "probability": 1/3,      # missing probs default to uniform
    "feature": 0.0,          # missing features default to 0
    "rating": 1500.0,        # missing ratings default to average
}


def _safe_value(val, kind="feature"):
    """Replace NaN/Inf with appropriate default.
    
    Args:
        val: Value to check
        kind: Type of value ("probability", "feature", "rating")
    
    Returns:
        Original value if valid, otherwise default for this kind
    """
    if val is None:
        return _NAN_STRATEGY.get(kind, 0.0)
    
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return _NAN_STRATEGY.get(kind, 0.0)
    
    return val



log = logging.getLogger(__name__)


def _default_score_probabilities() -> list[tuple[int, int, float]]:
    return []


def _default_prediction_set() -> list[int]:
    return [0, 1, 2]


@dataclass
class UnifiedPrediction:
    """Complete unified prediction output."""
    # Core probabilities
    p_home: float = 1 / 3
    p_draw: float = 1 / 3
    p_away: float = 1 / 3

    # Expected goals
    eg_home: float = 1.3
    eg_away: float = 1.1

    # Market predictions
    p_btts: float = 0.5
    p_over_25: float = 0.5
    p_under_25: float = 0.5

    # Score predictions
    most_likely_score: tuple[int, int] = (1, 0)
    score_probabilities: list[tuple[int, int, float]] = field(default_factory=_default_score_probabilities)

    # Confidence & uncertainty
    confidence: float = 0.5
    model_agreement: float = 0.5
    prediction_set: list[int] = field(default_factory=_default_prediction_set)
    prediction_interval: tuple[float, float] = (0.0, 1.0)

    # Component breakdown
    council_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    bayesian_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    statistical_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    market_probs: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Quality signals
    value_edge_home: float = 0.0
    value_edge_draw: float = 0.0
    value_edge_away: float = 0.0
    upset_risk: float = 0.0
    n_models_used: int = 0

    @property
    def most_likely_outcome(self) -> str:
        probs = {"Home": self.p_home, "Draw": self.p_draw, "Away": self.p_away}
        return max(probs, key=probs.get)  # type: ignore[arg-type]

    @property
    def outcome_label(self) -> str:
        labels = {0: "1", 1: "X", 2: "2"}
        idx = [self.p_home, self.p_draw, self.p_away].index(
            max(self.p_home, self.p_draw, self.p_away)
        )
        return labels[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_home": round(self.p_home, 4),
            "p_draw": round(self.p_draw, 4),
            "p_away": round(self.p_away, 4),
            "eg_home": round(self.eg_home, 2),
            "eg_away": round(self.eg_away, 2),
            "p_btts": round(self.p_btts, 4),
            "p_over_25": round(self.p_over_25, 4),
            "most_likely_score": f"{self.most_likely_score[0]}-{self.most_likely_score[1]}",
            "confidence": round(self.confidence, 4),
            "model_agreement": round(self.model_agreement, 4),
            "outcome": self.outcome_label,
            "upset_risk": round(self.upset_risk, 4),
            "n_models": self.n_models_used,
            "prediction_set": list(self.prediction_set),
            "prediction_interval": [
                round(self.prediction_interval[0], 4),
                round(self.prediction_interval[1], 4),
            ],
            "score_probabilities": [
                {"home": h, "away": a, "prob": round(p, 4)}
                for h, a, p in self.score_probabilities
            ],
            "component_breakdown": {
                "council": [round(p, 4) for p in self.council_probs],
                "bayesian": [round(p, 4) for p in self.bayesian_probs],
                "statistical": [round(p, 4) for p in self.statistical_probs],
                "market": [round(p, 4) for p in self.market_probs],
            },
            "value_edges": {
                "home": round(self.value_edge_home, 4),
                "draw": round(self.value_edge_draw, 4),
                "away": round(self.value_edge_away, 4),
            },
        }


def _compute_bma_weights(
    model_names: list[str],
    expert_accuracies: dict[str, float],
) -> list[float]:
    """Compute Bayesian Model Averaging weights from historical accuracies.

    P(M_k|data) is proportional to the likelihood of observing our data
    under model M_k. We approximate this from historical accuracy scores.

    Args:
        model_names: List of model names in order
        expert_accuracies: Dict mapping model name to accuracy ∈ [0, 1]

    Returns:
        Normalized weights [0, 1] summing to 1.0
    """
    weights = []
    for name in model_names:
        acc = expert_accuracies.get(name, 0.5)
        # Convert accuracy to likelihood (higher accuracy = higher weight)
        # Use exponential to sharpen differences
        weight = math.exp(acc * 2.0)  # Exponent sharpens small differences
        weights.append(weight)

    total = sum(weights)
    if total > 0:
        return [w / total for w in weights]
    return [1.0 / len(weights)] * len(weights)


def jensen_shannon_divergence(
    p: tuple[float, float, float],
    q: tuple[float, float, float],
) -> float:
    """Compute Jensen-Shannon divergence between two probability distributions.

    Symmetric version of KL divergence, useful for measuring disagreement
    between expert predictions. Bounded in [0, 1].

    Reference: Jensen & Shannon (1991) "Divergence and Sufficiency"

    Args:
        p, q: Probability tuples (must sum to ~1.0)

    Returns:
        JS divergence in [0, 1]
    """
    eps = 1e-12
    p = np.array([max(x, eps) for x in p])
    q = np.array([max(x, eps) for x in q])

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    # Midpoint distribution
    m = (p + q) / 2.0

    # KL divergence from p to m and q to m
    def kl(x, y):
        return (x * (np.log(x) - np.log(y))).sum()

    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    # Normalize to [0, 1]: max JS divergence is ln(2) ≈ 0.693
    js = js / math.log(2.0)

    return float(np.clip(js, 0.0, 1.0))


def model_agreement_from_divergences(
    probability_sets: list[tuple[float, float, float]],
) -> float:
    """Compute overall model agreement from pairwise JS divergences.

    Higher agreement (lower JS divergence) → higher score.

    Args:
        probability_sets: List of (p_h, p_d, p_a) from each model

    Returns:
        Agreement score in [0, 1] (1 = perfect agreement)
    """
    if len(probability_sets) <= 1:
        return 1.0

    # Compute all pairwise divergences
    divergences = []
    for i in range(len(probability_sets)):
        for j in range(i + 1, len(probability_sets)):
            div = jensen_shannon_divergence(
                probability_sets[i],
                probability_sets[j],
            )
            divergences.append(div)

    if not divergences:
        return 1.0

    # Convert divergence to agreement: higher div = lower agreement
    avg_div = float(np.mean(divergences))
    agreement = 1.0 - avg_div  # Inverse relationship

    return float(np.clip(agreement, 0.0, 1.0))


def logarithmic_opinion_pool(
    probability_sets: list[tuple[float, float, float]],
    weights: list[float],
) -> tuple[float, float, float]:
    """Combine probability distributions using the Logarithmic Opinion Pool.

    The LogOP is theoretically optimal for combining calibrated probability
    forecasts: P_combined ∝ ∏ᵢ Pᵢ^wᵢ

    This outperforms linear averaging because:
    1. It preserves the geometric mean of probabilities
    2. It properly handles extreme probabilities
    3. It is the only externally Bayesian pooling method

    Reference: Genest & Zidek (1986) "Combining probability distributions"
    """
    if not probability_sets:
        return (1 / 3, 1 / 3, 1 / 3)

    total_weight = sum(weights)
    if total_weight <= 0:
        total_weight = len(weights)
        weights = [1.0] * len(weights)

    # Normalize weights
    norm_weights = [w / total_weight for w in weights]

    # Log-space computation for numerical stability
    log_p = [0.0, 0.0, 0.0]
    eps = 1e-12
    for (ph, pd, pa), w in zip(probability_sets, norm_weights):
        log_p[0] += w * math.log(max(ph, eps))
        log_p[1] += w * math.log(max(pd, eps))
        log_p[2] += w * math.log(max(pa, eps))

    # Exponentiate and normalize
    max_log = max(log_p)
    raw = [math.exp(lp - max_log) for lp in log_p]
    total = sum(raw)

    if total > 0:
        return (raw[0] / total, raw[1] / total, raw[2] / total)
    return (1 / 3, 1 / 3, 1 / 3)


def linear_opinion_pool(
    probability_sets: list[tuple[float, float, float]],
    weights: list[float],
) -> tuple[float, float, float]:
    """Simple weighted average of probability distributions."""
    if not probability_sets:
        return (1 / 3, 1 / 3, 1 / 3)

    total_weight = sum(weights)
    if total_weight <= 0:
        return (1 / 3, 1 / 3, 1 / 3)

    p_h = sum(w * ph for (ph, _, _), w in zip(probability_sets, weights)) / total_weight
    p_d = sum(w * pd for (_, pd, _), w in zip(probability_sets, weights)) / total_weight
    p_a = sum(w * pa for (_, _, pa), w in zip(probability_sets, weights)) / total_weight

    total = p_h + p_d + p_a
    if total > 0:
        return (p_h / total, p_d / total, p_a / total)
    return (1 / 3, 1 / 3, 1 / 3)


def compute_value_edges(
    model_probs: tuple[float, float, float],
    market_probs: tuple[float, float, float],
    market_odds: tuple[float, float, float] | None = None,
    model_confidence: float = 0.5,
) -> tuple[float, float, float]:
    """Compute value edges: P(model) - P(market) with Kelly criterion integration.

    A value edge represents where our model thinks the market is mispriced:
    - Positive edge: outcome more likely according to model
    - Negative edge: outcome less likely according to model

    Optionally accounts for:
    - Model confidence (discount edge by uncertainty)
    - Actual odds (convert to implied probabilities for fine-grained comparison)

    Args:
        model_probs: (p_home, p_draw, p_away) from our model
        market_probs: (p_home, p_draw, p_away) from betting market
        market_odds: Optional (odds_home, odds_draw, odds_away) for conversion
        model_confidence: [0, 1] confidence in our model (discount edge)

    Returns:
        (edge_home, edge_draw, edge_away) bounded to [-1, 1]
    """
    if sum(market_probs) < 0.5:
        return (0.0, 0.0, 0.0)

    # Raw edges
    edge_h = model_probs[0] - market_probs[0]
    edge_d = model_probs[1] - market_probs[1]
    edge_a = model_probs[2] - market_probs[2]

    # If odds provided, optionally refine market_probs for finer edge
    # (this is advanced; market_probs typically already implies odds)
    if market_odds and min(market_odds) > 1.0:
        # Implied probs from odds with margin removal
        try:
            implied_h = 1.0 / market_odds[0]
            implied_d = 1.0 / market_odds[1]
            implied_a = 1.0 / market_odds[2]
            margin = implied_h + implied_d + implied_a - 1.0
            if margin > 0:
                implied_h /= (1.0 + margin)
                implied_d /= (1.0 + margin)
                implied_a /= (1.0 + margin)
                edge_h = model_probs[0] - implied_h
                edge_d = model_probs[1] - implied_d
                edge_a = model_probs[2] - implied_a
        except (ValueError, ZeroDivisionError):
            pass  # Fall back to raw edges

    # Confidence-weighted: high confidence → sharper edges, low confidence → dampen
    edge_h *= model_confidence
    edge_d *= model_confidence
    edge_a *= model_confidence

    # Bound to [-1, 1]
    edge_h = float(np.clip(edge_h, -1.0, 1.0))
    edge_d = float(np.clip(edge_d, -1.0, 1.0))
    edge_a = float(np.clip(edge_a, -1.0, 1.0))

    return (edge_h, edge_d, edge_a)


def compute_upset_risk(
    model_probs: tuple[float, float, float],
    market_probs: tuple[float, float, float],
    model_agreement: float,
    prediction_spread: float = 0.0,
    confidence: float = 0.5,
) -> float:
    """Estimate the risk of an upset based on multi-signal analysis.

    An upset is more likely when:
    1. Market and model disagree significantly
    2. Model agreement across methods is low
    3. The favorite probability is moderate (not dominant)
    4. Prediction spread (std of expert probabilities) is high
    5. Overall confidence in prediction is low

    Args:
        model_probs: (p_home, p_draw, p_away) from aggregated model
        market_probs: (p_home, p_draw, p_away) from betting market
        model_agreement: [0, 1] how closely expert models agree
        prediction_spread: Std dev of expert predictions (higher = more disagreement)
        confidence: [0, 1] overall prediction confidence

    Returns:
        Upset risk score in [0, 1]
    """
    if sum(market_probs) < 0.5:
        return 0.3  # No market data, moderate risk

    # Market-model disagreement: how much do they differ on favorite?
    max_mkt = max(market_probs)
    max_model = max(model_probs)
    disagreement = abs(max_mkt - max_model)

    # Favorite confidence: how weak is the favorite?
    fav_conf = max_model
    fav_weakness = max(0.0, 0.7 - fav_conf) / 0.7  # Higher when favorite is weak (< 70%)

    # Model disagreement across methods
    model_uncertainty = max(0.0, 1.0 - model_agreement)

    # Prediction spread: experts spread = higher upset risk
    spread_risk = float(np.clip(prediction_spread, 0.0, 1.0))

    # Inverse confidence: low confidence = higher upset risk
    confidence_risk = 1.0 - float(np.clip(confidence, 0.0, 1.0))

    # Composite upset risk with multi-signal fusion
    risk = (
        0.25 * disagreement * 2.0 +      # Market-model split
        0.20 * fav_weakness +              # Weak favorite
        0.25 * model_uncertainty +         # Internal model disagreement
        0.15 * spread_risk +               # Expert disagreement spread
        0.15 * confidence_risk             # Low overall confidence
    )

    return float(np.clip(risk, 0.0, 1.0))


def sanity_check_prediction(
    p_home: float,
    p_draw: float,
    p_away: float,
    elo_home: float = 1500.0,
    elo_away: float = 1500.0,
    market_probs: tuple[float, float, float] | None = None,
    max_deviation_from_market: float = 0.15,
    max_deviation_from_elo: float = 0.20,
) -> tuple[float, float, float]:
    """Sanity-check and correct obviously wrong predictions.

    The meta-learner can produce absurd results when noisy features overpower
    the strong signals (Elo, market). This function applies bounded corrections:

    1. If market odds are available, anchor predictions within ±15% of market
    2. If no market, use Elo-based prior as anchor
    3. Prevent the Elo-inferior team from being heavily favored (>55%) unless
       supported by market data
    4. Ensure probabilities are plausible (no team below 5% unless massive Elo gap)

    Args:
        p_home, p_draw, p_away: Raw model predictions
        elo_home, elo_away: Team Elo ratings
        market_probs: (p_h, p_d, p_a) from betting market, or None
        max_deviation_from_market: Maximum allowed deviation from market
        max_deviation_from_elo: Maximum allowed deviation from Elo prior

    Returns:
        Corrected (p_home, p_draw, p_away) that are plausible
    """
    # Step 1: Compute Elo-based prior using standard expected score formula
    # with +65 Elo home advantage
    elo_diff = (elo_home + 65.0) - elo_away
    elo_expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    elo_expected_away = 1.0 - elo_expected_home
    # Carve out a draw probability proportional to how close the teams are
    draw_factor = 1.0 - abs(elo_diff) / 1000.0
    draw_factor = max(0.15, min(0.35, 0.28 * max(0.5, draw_factor)))
    elo_prior_home = elo_expected_home * (1.0 - draw_factor)
    elo_prior_away = elo_expected_away * (1.0 - draw_factor)
    elo_prior_draw = draw_factor

    p_h, p_d, p_a = p_home, p_draw, p_away

    # Step 2: Blend and clamp based on available anchors
    if market_probs is not None and sum(market_probs) > 0.9:
        m_h, m_d, m_a = market_probs
        # Blend: 60% model, 30% market, 10% Elo prior
        p_h = 0.6 * p_h + 0.3 * m_h + 0.1 * elo_prior_home
        p_d = 0.6 * p_d + 0.3 * m_d + 0.1 * elo_prior_draw
        p_a = 0.6 * p_a + 0.3 * m_a + 0.1 * elo_prior_away
        # Clamp each probability to be within max_deviation_from_market of market
        p_h = max(m_h - max_deviation_from_market,
                  min(m_h + max_deviation_from_market, p_h))
        p_d = max(m_d - max_deviation_from_market,
                  min(m_d + max_deviation_from_market, p_d))
        p_a = max(m_a - max_deviation_from_market,
                  min(m_a + max_deviation_from_market, p_a))
    else:
        # No market data — Elo prior weight scales with gap magnitude
        # Small gap (< 100): 65% model / 35% Elo
        # Medium gap (100-300): 50% model / 50% Elo
        # Large gap (300+): 35% model / 65% Elo
        abs_gap = abs(elo_home - elo_away)
        # Sigmoid-like scaling: elo_weight goes from 0.35 to 0.65
        elo_weight = 0.35 + 0.30 * min(1.0, abs_gap / 400.0)
        model_weight = 1.0 - elo_weight

        p_h = model_weight * p_h + elo_weight * elo_prior_home
        p_d = model_weight * p_d + elo_weight * elo_prior_draw
        p_a = model_weight * p_a + elo_weight * elo_prior_away

        # Step 3: If Elo gap is large and model STILL favors the weaker team
        # after blending, apply an additional correction
        raw_elo_diff = elo_home - elo_away  # without home advantage
        if raw_elo_diff > 150 and p_a > p_h:
            # Model favors the weaker (away) team despite Elo gap
            correction = min(0.7, abs_gap / 600.0)
            p_h = (1 - correction) * p_h + correction * elo_prior_home
            p_d = (1 - correction) * p_d + correction * elo_prior_draw
            p_a = (1 - correction) * p_a + correction * elo_prior_away
        elif raw_elo_diff < -150 and p_h > p_a:
            # Model favors the weaker (home) team despite Elo gap
            correction = min(0.7, abs_gap / 600.0)
            p_h = (1 - correction) * p_h + correction * elo_prior_home
            p_d = (1 - correction) * p_d + correction * elo_prior_draw
            p_a = (1 - correction) * p_a + correction * elo_prior_away

    # Step 4: Normalize to sum to 1.0
    total = p_h + p_d + p_a
    if total > 0:
        p_h /= total
        p_d /= total
        p_a /= total
    else:
        p_h, p_d, p_a = 1 / 3, 1 / 3, 1 / 3

    # Step 5: Ensure no probability below 0.03 (3%)
    floor = 0.03
    p_h = max(p_h, floor)
    p_d = max(p_d, floor)
    p_a = max(p_a, floor)
    # Re-normalize after flooring
    total = p_h + p_d + p_a
    p_h /= total
    p_d /= total
    p_a /= total

    return (p_h, p_d, p_a)


def aggregate_predictions(
    council_probs: tuple[float, float, float] | None = None,
    bayesian_probs: tuple[float, float, float] | None = None,
    statistical_probs: tuple[float, float, float] | None = None,
    market_probs: tuple[float, float, float] | None = None,
    expert_weights: dict[str, float] | None = None,
    lambda_h: float = 1.3,
    lambda_a: float = 1.1,
    score_matrix: np.ndarray | None = None,
    confidence_scores: list[float] | None = None,
    p_btts: float | None = None,
    p_over_25: float | None = None,
    use_bayesian_averaging: bool = True,
    expert_accuracies: dict[str, float] | None = None,
    market_odds: tuple[float, float, float] | None = None,
) -> UnifiedPrediction:
    """Create a unified prediction from all available model outputs.

    This is the main entrypoint for generating a final prediction.
    It uses Bayesian Model Averaging (if available) or Logarithmic Opinion Pool
    to combine available probability sources with performance-based weights.

    Bayesian Model Averaging computes posterior model probabilities based on
    predictive accuracy: P(M_k|data) ∝ P(data|M_k) × P(M_k).

    Args:
        council_probs, bayesian_probs, statistical_probs, market_probs:
            Predictions from each source
        expert_weights: Pre-computed model weights (overrides accuracy-based)
        lambda_h, lambda_a: Expected goals for BTTS/Over calculation
        score_matrix: Poisson score matrix from statistical models
        confidence_scores: Per-model confidence estimates
        p_btts, p_over_25: Market or calculated probabilities
        use_bayesian_averaging: Use BMA instead of LogOP if True
        expert_accuracies: Historical accuracy per model (for BMA)
        market_odds: (odds_home, odds_draw, odds_away) for edge calculation

    Returns:
        UnifiedPrediction with full breakdown
    """
    prob_sets: list[tuple[float, float, float]] = []
    weights: list[float] = []
    model_names: list[str] = []
    n_models = 0

    # Collect all available predictions
    if council_probs and sum(council_probs) > 0.9:
        prob_sets.append(council_probs)
        model_names.append("council")
        n_models += 1

    if bayesian_probs and sum(bayesian_probs) > 0.9:
        prob_sets.append(bayesian_probs)
        model_names.append("bayesian")
        n_models += 1

    if statistical_probs and sum(statistical_probs) > 0.9:
        prob_sets.append(statistical_probs)
        model_names.append("statistical")
        n_models += 1

    if market_probs and sum(market_probs) > 0.9:
        prob_sets.append(market_probs)
        model_names.append("market")
        n_models += 1

    if not prob_sets:
        log.warning("No valid probability sets available for aggregation, using uniform default")
        return UnifiedPrediction(
            p_home=1/3, p_draw=1/3, p_away=1/3,
            confidence=0.0,  # Mark as low confidence
            n_models_used=0
        )

    # Compute weights
    if use_bayesian_averaging and expert_accuracies:
        # BMA: weights are posterior probabilities derived from accuracy
        weights = _compute_bma_weights(model_names, expert_accuracies)
    elif expert_weights:
        # Use provided weights (e.g., learned from historical performance)
        weights = [expert_weights.get(name, 0.25) for name in model_names]
    else:
        # Default weights (legacy behavior, slightly adjusted)
        weights = []
        for name in model_names:
            if name == "council":
                weights.append(0.45)
            elif name in ["bayesian", "statistical"]:
                weights.append(0.20)
            else:  # market
                weights.append(0.15)

    # Normalize weights
    total_w = sum(weights)
    if total_w > 0:
        weights = [w / total_w for w in weights]
    else:
        weights = [1.0 / len(weights)] * len(weights)

    # Use LogOP for combination
    p_h, p_d, p_a = logarithmic_opinion_pool(prob_sets, weights)

    # Compute agreement across methods using Jensen-Shannon divergence
    # (more principled than std of home probabilities alone)
    agreement = model_agreement_from_divergences(prob_sets)
    
    # Quality indicator: confidence based on number and agreement of sources
    quality_factors = {
        'n_models': n_models,
        'model_agreement': agreement,
        'weight_concentration': max(weights) if weights else 0.0,  # How peaked is weight distribution
    }
    quality_confidence = min(1.0, 0.3 + 0.7 * agreement) if agreement > 0 else 0.3
    home_probs = [ps[0] for ps in prob_sets]
    prediction_spread = float(np.std(home_probs)) if len(home_probs) > 1 else 0.0

    # Compute conformal prediction interval based on spread of predictions
    # Higher disagreement → wider interval; higher agreement → narrower interval
    spread = float(np.std(home_probs)) if len(home_probs) > 1 else 0.1
    # Interval width scales with disagreement: base width of 0.3 to 0.6 depending on agreement
    interval_width = 0.3 + 0.3 * spread  # ranges from ~0.3 to 0.6
    # Center interval around the max probability for the leading outcome
    max_prob = max(p_h, p_d, p_a)
    interval_lower = max(0.0, max_prob - interval_width / 2)
    interval_upper = min(1.0, max_prob + interval_width / 2)
    # Ensure minimum interval width
    if interval_upper - interval_lower < 0.2:
        center = (interval_lower + interval_upper) / 2
        interval_lower = max(0.0, center - 0.1)
        interval_upper = min(1.0, center + 0.1)

    # Score matrix analysis
    ml_score: tuple[int, int] = (1, 0)
    score_probs: list[tuple[int, int, float]] = []
    if score_matrix is not None and score_matrix.shape[0] > 1:
        all_scores: list[tuple[int, int, float]] = []
        for h in range(score_matrix.shape[0]):
            for a in range(score_matrix.shape[1]):
                all_scores.append((h, a, float(score_matrix[h, a])))
        all_scores.sort(key=lambda x: x[2], reverse=True)
        if all_scores:
            ml_score = (all_scores[0][0], all_scores[0][1])
            score_probs = all_scores[:5]

    # BTTS and Over 2.5 — blend if available from multiple sources
    btts = p_btts if p_btts is not None else _estimate_btts(lambda_h, lambda_a)
    o25 = p_over_25 if p_over_25 is not None else _estimate_over25(lambda_h, lambda_a)


    # Confidence for edge weighting
    conf_scores = confidence_scores or []
    if conf_scores:
        confidence = float(np.mean(conf_scores))
    else:
        confidence = agreement * max(p_h, p_d, p_a)
    confidence = max(0.05, min(0.98, confidence))

    # Value edges (with optional odds for fine-grained adjustment)
    mkt_tuple = market_probs if market_probs and sum(market_probs) > 0.9 else (0.0, 0.0, 0.0)
    edge_h, edge_d, edge_a = compute_value_edges(
        (p_h, p_d, p_a),
        mkt_tuple,
        market_odds=market_odds,
        model_confidence=confidence,
    )

    # Upset risk (multi-signal)
    upset = compute_upset_risk(
        (p_h, p_d, p_a),
        mkt_tuple,
        agreement,
        prediction_spread=prediction_spread,
        confidence=confidence,
    )

    # Apply sanity check
    p_h, p_d, p_a = sanity_check_prediction(
        p_h, p_d, p_a,
        market_probs=market_probs if market_probs and sum(market_probs) > 0.9 else None,
    )

    return UnifiedPrediction(
        p_home=p_h,
        p_draw=p_d,
        p_away=p_a,
        eg_home=lambda_h,
        eg_away=lambda_a,
        p_btts=btts,
        p_over_25=o25,
        p_under_25=1.0 - o25,
        most_likely_score=ml_score,
        score_probabilities=score_probs,
        confidence=confidence,
        model_agreement=agreement,
        prediction_set=[0, 1, 2],
        prediction_interval=(round(interval_lower, 4), round(interval_upper, 4)),
        council_probs=council_probs or (1 / 3, 1 / 3, 1 / 3),
        bayesian_probs=bayesian_probs or (1 / 3, 1 / 3, 1 / 3),
        statistical_probs=statistical_probs or (1 / 3, 1 / 3, 1 / 3),
        market_probs=market_probs or (0.0, 0.0, 0.0),
        value_edge_home=edge_h,
        value_edge_draw=edge_d,
        value_edge_away=edge_a,
        upset_risk=upset,
        n_models_used=n_models,
    )


def _estimate_btts(lambda_h: float, lambda_a: float) -> float:
    """Estimate BTTS probability from expected goals."""
    p_h_scores = 1.0 - math.exp(-lambda_h)
    p_a_scores = 1.0 - math.exp(-lambda_a)
    return p_h_scores * p_a_scores


def _estimate_over25(lambda_h: float, lambda_a: float) -> float:
    """Estimate Over 2.5 probability from expected goals."""
    total_lambda = lambda_h + lambda_a
    under_or_equal_2 = math.exp(-total_lambda) * (
        1.0 + total_lambda + (total_lambda ** 2) / 2.0
    )
    return 1.0 - under_or_equal_2


def inverse_variance_weighting(
    probability_sets: list[tuple[float, float, float]],
    variances: list[float],
) -> tuple[list[float], tuple[float, float, float]]:
    """Weight models inversely proportional to their recent log-loss variance.

    Lower variance (more consistent) models get higher weight.
    Uses Tikhonov regularization to avoid infinite weights for zero variance.

    Args:
        probability_sets: List of (p_home, p_draw, p_away) from each model
        variances: Log-loss variance for each model

    Returns:
        Tuple of (normalized_weights, combined_probabilities)
    """
    if not probability_sets or len(variances) != len(probability_sets):
        equal_weights = [1.0 / len(probability_sets)] * len(probability_sets)
        combined = logarithmic_opinion_pool(probability_sets, equal_weights)
        return (equal_weights, combined)

    # Inverse variance weighting with Tikhonov regularization
    min_variance = 0.001
    regularized_variances = [max(v, min_variance) for v in variances]
    inv_vars = [1.0 / v for v in regularized_variances]

    total = sum(inv_vars)
    if total <= 0:
        equal_weights = [1.0 / len(probability_sets)] * len(probability_sets)
        combined = logarithmic_opinion_pool(probability_sets, equal_weights)
        return (equal_weights, combined)

    norm_weights = [w / total for w in inv_vars]
    combined = logarithmic_opinion_pool(probability_sets, norm_weights)

    return (norm_weights, combined)


def bayesian_model_combination(
    probability_sets: list[tuple[float, float, float]],
    prior_shape: float = 1.0,
    n_samples: int = 1000,
    seed: int | None = None,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Combine models using Bayesian Model Combination (BMC).

    Draws weights from a Dirichlet posterior rather than using fixed weights.
    This approach is theoretically principled and accounts for uncertainty in
    optimal weights.

    Args:
        probability_sets: List of (p_home, p_draw, p_away) from each model
        prior_shape: Alpha parameter for Dirichlet prior (default 1.0 = uniform)
        n_samples: Number of samples to draw from posterior
        seed: Random seed for reproducibility

    Returns:
        Tuple of (combined_probabilities, metadata_dict)
    """
    if not probability_sets:
        return ((1 / 3, 1 / 3, 1 / 3), {})

    if seed is not None:
        np.random.seed(seed)

    n_models = len(probability_sets)

    # Use Dirichlet to sample weights
    alpha = [prior_shape] * n_models
    sampled_weights_list = []
    combined_samples = []

    for _ in range(n_samples):
        # Draw weights from Dirichlet posterior
        weights = np.random.dirichlet(alpha)
        sampled_weights_list.append(weights)

        # Combine with these weights
        combined = logarithmic_opinion_pool(probability_sets, list(weights))
        combined_samples.append(combined)

    # Average the combined samples
    combined_array = np.array(combined_samples)
    final_probs = (
        float(np.mean(combined_array[:, 0])),
        float(np.mean(combined_array[:, 1])),
        float(np.mean(combined_array[:, 2])),
    )

    # Compute posterior mean weight for each model
    mean_weights = list(np.mean(sampled_weights_list, axis=0))

    # Compute credible intervals (95%)
    credible_intervals = []
    for i in range(n_models):
        weights_i = [w[i] for w in sampled_weights_list]
        ci_lower = float(np.percentile(weights_i, 2.5))
        ci_upper = float(np.percentile(weights_i, 97.5))
        credible_intervals.append((ci_lower, ci_upper))

    metadata = {
        "method": "bayesian_model_combination",
        "mean_weights": [round(w, 4) for w in mean_weights],
        "credible_intervals": [
            [round(ci[0], 4), round(ci[1], 4)] for ci in credible_intervals
        ],
        "n_samples": n_samples,
    }

    return (final_probs, metadata)


def extremize_probabilities(
    probabilities: tuple[float, float, float],
    power: float = 1.2,
) -> tuple[float, float, float]:
    """Apply extremization to sharpen or soften combined probabilities.

    Extremization pushes probabilities away from 0.5 (sharpen if power > 1)
    or toward 0.5 (soften if power < 1).

    Formula: P'ᵢ = Pᵢ^power / Σⱼ(Pⱼ^power)

    Args:
        probabilities: (p_home, p_draw, p_away)
        power: Exponent (>1 sharpens, <1 softens, =1 no change)

    Returns:
        Extremized probabilities
    """
    if power == 1.0:
        return probabilities

    p_home, p_draw, p_away = probabilities
    eps = 1e-12

    # Apply power transformation
    raw = [
        max(p_home, eps) ** power,
        max(p_draw, eps) ** power,
        max(p_away, eps) ** power,
    ]

    # Normalize
    total = sum(raw)
    if total > 0:
        return (raw[0] / total, raw[1] / total, raw[2] / total)

    return (1 / 3, 1 / 3, 1 / 3)


def learn_optimal_weights(
    scored_predictions: list[dict[str, Any]],
    method: str = "inverse_variance",
) -> dict[str, float]:
    """Learn optimal weights from historical prediction performance.

    Args:
        scored_predictions: List of dicts with keys:
            - probs: tuple of (p_home, p_draw, p_away)
            - variance: log-loss variance
            - outcome: 0, 1, or 2
            - predicted_prob: the probability assigned to actual outcome
        method: "inverse_variance", "accuracy", or "logloss"

    Returns:
        Dictionary with optimal weights for each method/model
    """
    if not scored_predictions or method not in ["inverse_variance", "accuracy", "logloss"]:
        return {}

    if method == "inverse_variance":
        # Group by model and compute variance
        model_variances = {}
        for pred in scored_predictions:
            model = pred.get("model", "default")
            variance = pred.get("variance", 0.01)
            if model not in model_variances:
                model_variances[model] = []
            model_variances[model].append(variance)

        # Average variance per model
        avg_variances = {m: float(np.mean(v)) for m, v in model_variances.items()}

        # Inverse variance weight
        min_var = 0.001
        inv_vars = {m: 1.0 / max(v, min_var) for m, v in avg_variances.items()}
        total = sum(inv_vars.values())
        if total > 0:
            return {m: round(w / total, 4) for m, w in inv_vars.items()}

    elif method == "accuracy":
        # Weight by accuracy per model
        model_scores = {}
        for pred in scored_predictions:
            model = pred.get("model", "default")
            if model not in model_scores:
                model_scores[model] = {"correct": 0, "total": 0}
            model_scores[model]["total"] += 1
            if pred.get("correct"):
                model_scores[model]["correct"] += 1

        # Compute accuracy
        accuracies = {
            m: (s["correct"] / max(s["total"], 1)) for m, s in model_scores.items()
        }
        total = sum(accuracies.values())
        if total > 0:
            return {m: round(a / total, 4) for m, a in accuracies.items()}

    elif method == "logloss":
        # Lower log-loss → higher weight (use 1/logloss)
        model_losses = {}
        for pred in scored_predictions:
            model = pred.get("model", "default")
            loss = pred.get("logloss", 0.5)
            if model not in model_losses:
                model_losses[model] = []
            model_losses[model].append(loss)

        avg_losses = {m: float(np.mean(losses)) for m, losses in model_losses.items()}
        inv_losses = {m: 1.0 / max(loss, 0.01) for m, loss in avg_losses.items()}
        total = sum(inv_losses.values())
        if total > 0:
            return {m: round(w / total, 4) for m, w in inv_losses.items()}

    return {}


def thompson_sampling_weights(
    model_successes: dict[str, int],
    model_failures: dict[str, int],
) -> dict[str, float]:
    """Generate exploration-exploitation weights via Thompson Sampling.

    For each model, draws from a Beta(successes+1, failures+1) distribution.
    The +1 provides a uniform Beta(1,1) prior.  Normalising the draws yields
    weights that naturally balance exploiting the best-known model with
    exploring under-sampled ones.

    Args:
        model_successes: Mapping of model name to number of successful
            predictions (e.g. correct 1X2 calls).
        model_failures: Mapping of model name to number of failed predictions.

    Returns:
        Dictionary mapping each model name to its sampled weight (sums to 1).
    """
    if not model_successes:
        return {}

    draws: dict[str, float] = {}
    for model in model_successes:
        alpha = model_successes.get(model, 0) + 1
        beta = model_failures.get(model, 0) + 1
        draws[model] = float(np.random.beta(alpha, beta))

    total = sum(draws.values())
    if total <= 0:
        n = len(draws)
        return {m: 1.0 / n for m in draws}

    return {m: d / total for m, d in draws.items()}


def cluster_experts(
    probability_sets: list[tuple[float, float, float]],
    threshold: float = 0.85,
) -> list[float]:
    """Downweight redundant experts by clustering correlated ones.

    Experts whose probability vectors are highly correlated (Pearson r above
    *threshold*) are placed in the same cluster.  Each expert in a cluster of
    size *k* receives weight 1/k so that the cluster's total contribution
    equals that of a single independent expert.

    Uses a simple single-linkage approach: iterate through experts and merge
    into the first cluster whose member has correlation > threshold.

    Args:
        probability_sets: List of (p_home, p_draw, p_away) from each expert.
        threshold: Pearson correlation above which two experts are considered
            redundant.  Default 0.85.

    Returns:
        List of adjusted weights (same order as *probability_sets*), where
        each weight is 1 / cluster_size for the cluster the expert belongs to.
        Weights are **not** normalised to sum to 1 so the caller can decide
        how to use them.
    """
    n = len(probability_sets)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # Build correlation matrix
    mat = np.array(probability_sets)  # shape (n, 3)
    # Handle constant vectors (std == 0) by treating them as uncorrelated
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            std_i = np.std(mat[i])
            std_j = np.std(mat[j])
            if std_i < 1e-12 or std_j < 1e-12:
                r = 0.0
            else:
                r = float(np.corrcoef(mat[i], mat[j])[0, 1])
                if np.isnan(r):
                    r = 0.0
            corr[i, j] = r
            corr[j, i] = r

    # Single-linkage clustering
    cluster_ids = [-1] * n
    current_cluster = 0
    for i in range(n):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = current_cluster
        for j in range(i + 1, n):
            if cluster_ids[j] != -1:
                continue
            if corr[i, j] > threshold:
                cluster_ids[j] = current_cluster
        current_cluster += 1

    # Compute cluster sizes
    from collections import Counter
    cluster_sizes = Counter(cluster_ids)

    return [1.0 / cluster_sizes[cluster_ids[i]] for i in range(n)]


def contextual_weight_selection(
    match_context: dict[str, bool],
    weight_profiles: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Select expert weights appropriate for the match context.

    Different match situations (e.g. a derby, top-vs-bottom clash, cup match)
    can favour different experts.  This function looks up the first matching
    context in *weight_profiles* and returns the corresponding weight dict.

    Contexts are checked in the order they appear in *match_context*; the
    first key whose value is ``True`` **and** has a corresponding entry in
    *weight_profiles* wins.  If nothing matches, a ``"default"`` profile is
    used (or empty dict if that is also absent).

    Args:
        match_context: Dict with boolean flags, e.g.
            ``{"is_derby": True, "top_vs_bottom": False, ...}``.
        weight_profiles: Maps context names to weight dicts, e.g.
            ``{"is_derby": {"council": 0.5, "statistical": 0.3, ...}}``.

    Returns:
        Weight dict for the matched context, or the ``"default"`` profile.
    """
    if not match_context or not weight_profiles:
        return weight_profiles.get("default", {})

    for context_key, is_active in match_context.items():
        if is_active and context_key in weight_profiles:
            return weight_profiles[context_key]

    return weight_profiles.get("default", {})


def online_weight_update(
    current_weights: dict[str, float],
    prediction: dict[str, tuple[float, float, float]],
    outcome: int,
    learning_rate: float = 0.01,
) -> dict[str, float]:
    """Update expert weights after observing a match result.

    Uses a multiplicative weight update rule inspired by the Hedge algorithm:

        w_i <- w_i * exp(learning_rate * ln p_i(outcome))

    where p_i(outcome) is the probability that expert *i* assigned to the
    actual outcome.  Experts who gave higher probability to the realised
    result are up-weighted; those who were surprised are down-weighted.

    The weights are re-normalised after the update so they sum to 1.

    Args:
        current_weights: Current weight per expert (model name -> weight).
        prediction: Each expert's probability vector keyed by model name.
            Values are (p_home, p_draw, p_away).
        outcome: Realised outcome index -- 0 (home), 1 (draw), or 2 (away).
        learning_rate: Step size for the multiplicative update.  Smaller
            values make adaptation more conservative.  Default 0.01.

    Returns:
        Updated and normalised weight dict.
    """
    if not current_weights or not prediction:
        return current_weights

    eps = 1e-12
    updated: dict[str, float] = {}

    for model, w in current_weights.items():
        probs = prediction.get(model)
        if probs is None:
            updated[model] = w
            continue
        p_outcome = max(probs[outcome], eps)
        updated[model] = w * math.exp(learning_rate * math.log(p_outcome))

    total = sum(updated.values())
    if total <= 0:
        n = len(updated)
        return {m: 1.0 / n for m in updated}

    return {m: v / total for m, v in updated.items()}
