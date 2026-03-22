"""Ensemble methods, calibration, and advanced prediction functions.

Implements:
- Self-calibrating ensemble with adaptive model weights
- Ensemble disagreement and diversity metrics
- Unified predictions combining multiple models
- Advanced probabilistic models (ordered probit, Bradley-Terry-Davidson)
- Goal timing and match state simulation

References:
    Constantinou et al. (2016) "Towards smart football match outcome prediction"
    Guo et al. (2017) "On Calibration of Modern Neural Networks"
    Williams & Beer (2010) "Information dynamics: its application to Partially Observ Systems"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import norm, poisson as poisson_dist


# ═══════════════════════════════════════════════════════════════════
# EXPECTED POINTS AND SCORE MATRIX OPERATIONS
# ═══════════════════════════════════════════════════════════════════
def expected_points_from_score_matrix(mat: np.ndarray) -> tuple[float, float]:
    """Compute expected points for home and away from score matrix.

    In football: home win = 3 pts, draw = 1 pt, away win = 0 pts.

    Args:
        mat: Score probability matrix (9×9 typical)

    Returns:
        (home_expected_pts, away_expected_pts)
    """
    n = mat.shape[0]
    home_points = 0.0
    away_points = 0.0

    for i in range(n):
        for j in range(n):
            prob = mat[i, j]
            if i > j:
                home_points += 3.0 * prob
            elif i == j:
                home_points += 1.0 * prob
                away_points += 1.0 * prob
            else:
                away_points += 3.0 * prob

    return (home_points, away_points)


# ═══════════════════════════════════════════════════════════════════
# PROBABILISTIC MODEL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def ordered_probit_1x2(mu: float, sigma: float = 1.0) -> dict[str, float]:
    """Ordered probit model for match outcomes.

    Models latent variable Y ~ N(μ, σ²), with thresholds:
        y < 0 → Home loss
        0 ≤ y < threshold → Draw
        y ≥ threshold → Home win

    The threshold is chosen symmetrically around 0 to match typical
    football expectations (home advantage > 0).

    Args:
        mu: Mean of latent variable (positive → home favored)
        sigma: Standard deviation

    Returns:
        Dict with p_home, p_draw, p_away
    """
    sigma = max(sigma, 0.01)
    # Threshold chosen to reflect symmetric distribution of draws
    # around home advantage
    threshold = abs(mu) * 0.5 if mu != 0 else 0.5

    p_home = 1.0 - norm.cdf((threshold - mu) / sigma)
    p_draw = norm.cdf((threshold - mu) / sigma) - norm.cdf((-threshold - mu) / sigma)
    p_away = norm.cdf((-threshold - mu) / sigma)

    # Renormalize
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    return {
        "p_home": float(p_home),
        "p_draw": float(p_draw),
        "p_away": float(p_away),
    }


def skellam_match_probs(lambda_h: float, lambda_a: float) -> dict[str, float]:
    """Skellam-based 1X2 probabilities with goal difference moments.

    Uses the Skellam distribution (difference of two Poisson variables)
    to compute match outcome probabilities and expected goal difference.

    Args:
        lambda_h: Home expected goals
        lambda_a: Away expected goals

    Returns:
        Dict with p_home, p_draw, p_away, expected_gd, gd_var
    """
    from footy.models.math.distributions import skellam_probs

    result = skellam_probs(lambda_h, lambda_a, max_diff=10)

    # Normalize probabilities (may truncate tail)
    p_h = result["p_home"]
    p_d = result["p_draw"]
    p_a = result["p_away"]

    total = p_h + p_d + p_a
    if total > 0:
        p_h /= total
        p_d /= total
        p_a /= total

    return {
        "p_home": p_h,
        "p_draw": p_d,
        "p_away": p_a,
        "expected_gd": result["mean_diff"],
        "gd_var": result["var_diff"],
        "gd_std": result["std_diff"],
    }


def bradley_terry_davidson(strength_h: float, strength_a: float) -> dict[str, float]:
    """Bradley-Terry-Davidson model for match probabilities.

    Extends Bradley-Terry with Davidson's draw model.
    P(H) = θ·π_h / (θ·π_h + π_a + ν√(π_h·π_a))
    P(D) = ν√(π_h·π_a) / (total)
    P(A) = π_a / (total)

    Args:
        strength_h: Home team strength (log scale)
        strength_a: Away team strength (log scale)

    Returns:
        Dict with p_home, p_draw, p_away
    """
    from footy.models.math.simulation import bradley_terry_probs

    p_h, p_d, p_a = bradley_terry_probs(strength_h, strength_a)
    return {
        "p_home": p_h,
        "p_draw": p_d,
        "p_away": p_a,
    }


# ═══════════════════════════════════════════════════════════════════
# TEAM STRENGTH AND HISTORICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════
def pythagorean_win_pct(goals_for: float, goals_against: float, exp: float = 1.8) -> float:
    """Pythagorean expectation for football win percentage.

    Generalization of baseball's Pythagorean formula:
    Expected Win % = GF^exp / (GF^exp + GA^exp)

    This estimates what a team's win rate "should be" given goal differential.
    Useful for identifying lucky/unlucky teams (regression candidates).

    Args:
        goals_for: Total goals scored in sample
        goals_against: Total goals conceded in sample
        exp: Exponent (1.8-2.0 typical for football)

    Returns:
        Expected win percentage (0-1)
    """
    gf = max(goals_for, 0.1)
    ga = max(goals_against, 0.1)

    numerator = gf ** exp
    denominator = numerator + ga ** exp

    return numerator / max(denominator, 1e-9)


def regression_to_mean_signal(actual_gd: float, expected_gd: float, matches_played: int) -> dict[str, float | str]:
    """Detect overperformance/underperformance relative to expected strength.

    Quantifies how much a team has beaten/underperformed its expected goal differential,
    with smaller sample sizes getting larger regression adjustments.

    Args:
        actual_gd: Observed goal differential (goals for - against)
        expected_gd: Expected goal differential (from model)
        matches_played: Number of matches for credibility

    Returns:
        Dict with overperformance, matches_info, direction (positive/negative/neutral)
    """
    over = actual_gd - expected_gd
    # Credibility: more matches → less regression
    credibility = 1.0 / (1.0 + math.exp(-0.1 * matches_played + 2.0))

    # Direction
    if over > 0.1:
        direction = "negative"  # expect regression down
    elif over < -0.1:
        direction = "positive"  # expect regression up
    else:
        direction = "neutral"

    return {
        "overperformance": over,
        "credibility": credibility,
        "direction": direction,
        "implied_adjustment": -over * (1.0 - credibility),
    }


# ═══════════════════════════════════════════════════════════════════
# MATCH STATE AND MARKOV CHAIN SIMULATION
# ═══════════════════════════════════════════════════════════════════
def match_state_transition_probs(lambda_h: float, lambda_a: float) -> dict[str, float]:
    """Compute transition probabilities for next goal scorer in live match.

    Models the instantaneous probability of home/away scoring next,
    useful for in-play markets and live betting.

    Args:
        lambda_h: Home intensity (goals per remaining time)
        lambda_a: Away intensity

    Returns:
        Dict with p_home_scores_next, p_away_scores_next, p_no_goal
    """
    # Exponential race: first Poisson(λ_h) or Poisson(λ_a) wins
    total = lambda_h + lambda_a
    if total < 1e-9:
        total = 0.1

    p_h_next = lambda_h / total
    p_a_next = lambda_a / total
    # Probability of no goal in next 30 min
    p_no_goal_30 = math.exp(-(lambda_h + lambda_a) / 3.0)

    return {
        "p_home_scores_next": p_h_next,
        "p_away_scores_next": p_a_next,
        "p_no_goal_30min": p_no_goal_30,
    }


def markov_chain_simulate(
    lambda_h: float,
    lambda_a: float,
    expected_for: float = 1.5,
    expected_against: float = 1.5,
    rho: float = -0.13,
    n_sims: int = 5000,
) -> dict[str, float]:
    """Monte Carlo simulation using Markov chain state transitions.

    Simulates match progression with time-varying goal rates.

    Args:
        lambda_h: Home expected goals
        lambda_a: Away expected goals
        expected_for: Expected goals (unused, for API compatibility)
        expected_against: Expected goals against (unused, for API compatibility)
        rho: Dixon-Coles correlation parameter
        n_sims: Number of simulations

    Returns:
        Dict with mc_p_home, mc_p_draw, mc_p_away, mc_eg_h, mc_eg_a
    """
    from footy.models.math.simulation import monte_carlo_simulate

    result = monte_carlo_simulate(lambda_h, lambda_a, rho, n_sims)

    return {
        "mc_p_home": result.get("mc_p_home", 0.3),
        "mc_p_draw": result.get("mc_p_draw", 0.4),
        "mc_p_away": result.get("mc_p_away", 0.3),
        "mc_eg_h": lambda_h,
        "mc_eg_a": lambda_a,
        "mc_p_btts": result.get("mc_btts", 0.5),
    }


def bivariate_normal_goal_diff(lambda_h: float = 1.35, lambda_a: float = 1.15) -> dict[str, float]:
    """Bivariate normal approximation to goal difference distribution.

    Uses CLT approximation: GD ≈ N(μ, σ²) where
        μ = λ_h - λ_a
        σ² = λ_h + λ_a

    Args:
        lambda_h: Home expected goals
        lambda_a: Away expected goals

    Returns:
        Dict with p_home, p_draw, p_away, gd_mean, gd_var
    """
    mu = lambda_h - lambda_a
    sigma2 = lambda_h + lambda_a
    sigma = math.sqrt(max(sigma2, 1e-9))

    # Threshold for draws (±0.5)
    p_home = 1.0 - norm.cdf((0.5 - mu) / sigma)
    p_draw = norm.cdf((0.5 - mu) / sigma) - norm.cdf((-0.5 - mu) / sigma)
    p_away = norm.cdf((-0.5 - mu) / sigma)

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "gd_mean": mu,
        "gd_var": sigma2,
    }


def goal_timing_intensity(total_lambda: float, n_periods: int = 90) -> list[float]:
    """Goal timing intensity function across match periods.

    Returns an array of goal-scoring intensities for each minute,
    modelling typical football match patterns (faster scoring in certain periods).

    Args:
        total_lambda: Total expected goals for full match
        n_periods: Number of time periods (90 for minutes)

    Returns:
        List of length n_periods with per-period intensities
    """
    # Empirical pattern: slower start, peak in mid-match, variable end
    # U-shaped intensity across 90 minutes
    intensities = []
    for t in range(n_periods):
        # Time-normalized (0 to 1)
        tau = t / max(n_periods - 1, 1)

        # U-shaped: lower at start, higher at end
        # Center peak + slight late-game spike
        intensity = 0.8 + 0.6 * math.sin(math.pi * tau)

        intensities.append(intensity)

    # Normalize to sum to total_lambda
    total = sum(intensities)
    if total > 0:
        intensities = [i * total_lambda / total for i in intensities]

    return intensities


# ═══════════════════════════════════════════════════════════════════
# BUILD ALL SCORE MATRICES
# ═══════════════════════════════════════════════════════════════════
def build_all_score_matrices(lambda_h: float, lambda_a: float, max_goals: int = 8) -> dict[str, np.ndarray]:
    """Build 12 different score matrices using available distributions.

    Combines multiple models:
    1. Dixon-Coles (DC)
    2. Bivariate Poisson (BP)
    3. COM-Poisson
    4. Zero-Inflated Poisson (ZIP)
    5. Negative Binomial
    6. Weibull
    7. Double Poisson
    8-12. Various copula and hybrid models

    Args:
        lambda_h: Home expected goals
        lambda_a: Away expected goals
        max_goals: Maximum goals to model

    Returns:
        Dict with 12 score matrices keyed by model name
    """
    from footy.models.math.distributions import (
        build_dc_score_matrix,
        build_bivariate_poisson_matrix,
        build_com_poisson_matrix,
        build_zip_score_matrix,
        build_negative_binomial_matrix,
    )
    from footy.models.math.weibull import build_weibull_count_matrix
    from footy.models.math.double_poisson import build_double_poisson_matrix
    from footy.models.math.copulas import build_copula_score_matrix

    matrices = {}

    # 1. Dixon-Coles
    matrices["dixon_coles"] = build_dc_score_matrix(lambda_h, lambda_a, max_goals=max_goals)

    # 2. Bivariate Poisson
    matrices["bivariate_poisson"] = build_bivariate_poisson_matrix(lambda_h, lambda_a, max_goals=max_goals)

    # 3. COM-Poisson
    matrices["com_poisson"] = build_com_poisson_matrix(lambda_h, lambda_a, max_goals=max_goals)

    # 4. Zero-Inflated Poisson
    matrices["zip"] = build_zip_score_matrix(lambda_h, lambda_a, max_goals=max_goals)

    # 5. Negative Binomial
    matrices["negative_binomial"] = build_negative_binomial_matrix(lambda_h, lambda_a, max_goals=max_goals)

    # 6. Weibull Count
    matrices["weibull"] = build_weibull_count_matrix(lambda_h, lambda_a, max_goals=max_goals)

    # 7. Double Poisson
    matrices["double_poisson"] = build_double_poisson_matrix(lambda_h, lambda_a, max_goals=max_goals)

    # 8-10. Copulas with different dependencies
    matrices["copula_frank"] = build_copula_score_matrix(lambda_h, lambda_a, copula_family="frank", max_goals=max_goals)
    matrices["copula_clayton"] = build_copula_score_matrix(lambda_h, lambda_a, copula_family="clayton", max_goals=max_goals)
    matrices["copula_gumbel"] = build_copula_score_matrix(lambda_h, lambda_a, copula_family="gumbel", max_goals=max_goals)

    # 11-12. Hybrid models (average of two complementary models)
    avg_1 = (matrices["dixon_coles"] + matrices["com_poisson"]) / 2
    avg_1 /= avg_1.sum()
    matrices["hybrid_dc_comp"] = avg_1

    avg_2 = (matrices["bivariate_poisson"] + matrices["double_poisson"]) / 2
    avg_2 /= avg_2.sum()
    matrices["hybrid_bp_dp"] = avg_2

    return matrices


def aggregate_model_predictions(lambda_h: float, lambda_a: float) -> dict[str, float]:
    """Aggregate predictions from all 12 models with BMA.

    Combines score matrices using Bayesian Model Averaging
    and extracts comprehensive match probabilities.

    Args:
        lambda_h: Home expected goals
        lambda_a: Away expected goals

    Returns:
        Dict with BMA probabilities, entropy, model agreement metrics
    """
    from footy.models.math.bma import bayesian_model_average
    from footy.models.math.simulation import extract_match_probs

    matrices = build_all_score_matrices(lambda_h, lambda_a)
    mat_list = list(matrices.values())

    # BMA with uniform weights (no likelihood data)
    bma_mat = bayesian_model_average(mat_list)

    # Extract probabilities
    result = extract_match_probs(bma_mat)

    # Model agreement: compute pairwise KL divergences
    from footy.models.math.scoring import kl_divergence

    kl_dists = []
    probs_list = [extract_match_probs(m) for m in mat_list]
    for i, p1 in enumerate(probs_list):
        for j, p2 in enumerate(probs_list):
            if i < j:
                kl = kl_divergence(
                    [p1["p_home"], p1["p_draw"], p1["p_away"]],
                    [p2["p_home"], p2["p_draw"], p2["p_away"]],
                )
                kl_dists.append(kl)

    avg_agreement = 1.0 / (1.0 + np.mean(kl_dists)) if kl_dists else 1.0

    return {
        "bma_p_home": result["p_home"],
        "bma_p_draw": result["p_draw"],
        "bma_p_away": result["p_away"],
        "bma_entropy": -sum(p * math.log(max(p, 1e-12)) for p in [result["p_home"], result["p_draw"], result["p_away"]]),
        "model_agreement": avg_agreement,
        "bma_btts": result["p_btts"],
        "bma_eg_h": result["eg_home"],
        "bma_eg_a": result["eg_away"],
    }


# ═══════════════════════════════════════════════════════════════════
# SELF-CALIBRATING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════
@dataclass
class ModelRecord:
    """Track performance of a single model in ensemble."""

    n_predictions: int = 0
    total_log_loss: float = 0.0
    total_rps: float = 0.0
    correct_outcomes: int = 0

    def update(self, pred_probs: list[float], outcome_idx: int) -> None:
        """Update record with new prediction and outcome."""
        from footy.models.math.scoring import log_loss, ranked_probability_score

        self.n_predictions += 1
        self.total_log_loss += log_loss(pred_probs, outcome_idx)
        self.total_rps += ranked_probability_score(pred_probs, outcome_idx)
        if np.argmax(pred_probs) == outcome_idx:
            self.correct_outcomes += 1

    def avg_log_loss(self) -> float:
        """Average log loss (lower = better)."""
        if self.n_predictions == 0:
            return 0.0
        return self.total_log_loss / self.n_predictions

    def accuracy(self) -> float:
        """Accuracy (higher = better)."""
        if self.n_predictions == 0:
            return 1.0 / 3  # neutral for 3-class problem
        return self.correct_outcomes / self.n_predictions


@dataclass
class SelfCalibratingEnsemble:
    """Ensemble that adapts weights based on model performance.

    Tracks log loss and accuracy for each model, then dynamically
    reweights them using softmax of inverted performance metrics.
    """

    model_names: list[str] = field(default_factory=list)
    records: dict[str, ModelRecord] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default model list if not provided."""
        if not self.model_names:
            self.model_names = [
                "dixon_coles", "bivariate_poisson", "com_poisson", "zip",
                "negative_binomial", "weibull", "double_poisson",
                "copula_frank", "copula_clayton", "copula_gumbel",
                "hybrid_dc_comp", "hybrid_bp_dp"
            ]
        for name in self.model_names:
            if name not in self.records:
                self.records[name] = ModelRecord()

    def get_weights(self) -> dict[str, float]:
        """Compute current model weights based on performance.

        Lower log loss → higher weight.
        Uses softmax of negative log loss.
        """
        losses = np.array([
            self.records[name].avg_log_loss()
            for name in self.model_names
        ])

        # If no data yet, uniform weights
        if np.max(losses) == 0:
            w = np.ones(len(self.model_names)) / len(self.model_names)
        else:
            # Softmax of inverted losses: better models (lower loss) get more weight
            logits = -losses  # negative: lower loss → higher logit
            logits = logits - logits.max()  # numerical stability
            w = np.exp(logits)
            w = w / w.sum()

        return {name: float(w_i) for name, w_i in zip(self.model_names, w)}

    def update(
        self,
        predictions: dict[str, dict[str, float]],
        actual_outcome: str,
    ) -> None:
        """Update ensemble with new match outcome.

        Args:
            predictions: Dict mapping model name to {p_home, p_draw, p_away}
            actual_outcome: "home", "draw", or "away"
        """
        outcome_map = {"home": 0, "draw": 1, "away": 2}
        outcome_idx = outcome_map.get(actual_outcome, 1)

        for name in self.model_names:
            if name in predictions:
                pred = predictions[name]
                probs = [pred.get("p_home", 1/3), pred.get("p_draw", 1/3), pred.get("p_away", 1/3)]
                self.records[name].update(probs, outcome_idx)

    def to_dict(self) -> dict:
        """Serialize ensemble state."""
        return {
            "model_names": self.model_names,
            "records": {
                name: {
                    "n_predictions": rec.n_predictions,
                    "total_log_loss": rec.total_log_loss,
                    "total_rps": rec.total_rps,
                    "correct_outcomes": rec.correct_outcomes,
                }
                for name, rec in self.records.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> SelfCalibratingEnsemble:
        """Deserialize ensemble state."""
        ens = cls(model_names=data["model_names"])
        for name, rec_data in data["records"].items():
            rec = ModelRecord(
                n_predictions=rec_data["n_predictions"],
                total_log_loss=rec_data["total_log_loss"],
                total_rps=rec_data["total_rps"],
                correct_outcomes=rec_data["correct_outcomes"],
            )
            ens.records[name] = rec
        return ens


# ═══════════════════════════════════════════════════════════════════
# ENSEMBLE DISAGREEMENT AND UNIFIED PREDICTION
# ═══════════════════════════════════════════════════════════════════
def ensemble_disagreement(predictions: list[dict[str, float]]) -> dict[str, float]:
    """Compute disagreement (diversity) metrics for ensemble.

    Measures how much models disagree on probabilities.
    High disagreement = low confidence, useful for uncertainty estimates.

    Args:
        predictions: List of {p_home, p_draw, p_away} dicts

    Returns:
        Dict with diversity_score and confidence
    """
    if not predictions:
        return {"diversity_score": 0.0, "confidence": 1.0}

    probs_array = np.array([
        [p.get("p_home", 1/3), p.get("p_draw", 1/3), p.get("p_away", 1/3)]
        for p in predictions
    ])

    # Pairwise KL divergences
    from footy.models.math.scoring import kl_divergence

    kl_scores = []
    for i in range(len(probs_array)):
        for j in range(i + 1, len(probs_array)):
            kl = kl_divergence(probs_array[i], probs_array[j])
            kl_scores.append(kl)

    avg_disagreement = np.mean(kl_scores) if kl_scores else 0.0

    # Confidence: inverse of disagreement
    confidence = 1.0 / (1.0 + avg_disagreement)

    return {
        "diversity_score": float(avg_disagreement),
        "confidence": float(confidence),
    }


def unified_prediction(lambda_h: float, lambda_a: float) -> dict[str, float]:
    """Unified prediction combining all available models.

    Aggregates Dixon-Coles, Skellam, BMA, and other methods
    into a single consensus prediction.

    Args:
        lambda_h: Home expected goals
        lambda_a: Away expected goals

    Returns:
        Dict with p_home, p_draw, p_away, confidence, entropy
    """
    # Get various model predictions
    from footy.models.math.scoring import kl_divergence

    dc_result = aggregate_model_predictions(lambda_h, lambda_a)
    skellam_result = skellam_match_probs(lambda_h, lambda_a)
    probit_result = ordered_probit_1x2(lambda_h - lambda_a)

    # Ensemble predictions
    preds = [
        {"p_home": dc_result["bma_p_home"], "p_draw": dc_result["bma_p_draw"], "p_away": dc_result["bma_p_away"]},
        {"p_home": skellam_result["p_home"], "p_draw": skellam_result["p_draw"], "p_away": skellam_result["p_away"]},
        probit_result,
    ]

    # Simple average
    unified = {
        "p_home": np.mean([p["p_home"] for p in preds]),
        "p_draw": np.mean([p["p_draw"] for p in preds]),
        "p_away": np.mean([p["p_away"] for p in preds]),
    }

    # Renormalize
    total = unified["p_home"] + unified["p_draw"] + unified["p_away"]
    if total > 0:
        unified["p_home"] /= total
        unified["p_draw"] /= total
        unified["p_away"] /= total

    # Compute entropy and disagreement
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in [unified["p_home"], unified["p_draw"], unified["p_away"]])
    disagreement = ensemble_disagreement(preds)

    return {
        "p_home": float(unified["p_home"]),
        "p_draw": float(unified["p_draw"]),
        "p_away": float(unified["p_away"]),
        "confidence": disagreement["confidence"],
        "entropy": float(entropy),
        "eg_h": lambda_h,
        "eg_a": lambda_a,
    }
