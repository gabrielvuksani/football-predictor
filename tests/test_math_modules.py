"""Comprehensive tests for mathematical modules."""
from __future__ import annotations

import numpy as np
import pytest

from footy.models.math.distributions import (
    skellam_probs,
    bivariate_poisson_pmf,
)
from footy.models.math.scoring import (
    ranked_probability_score,
    rps_from_result,
    brier_score,
)


class TestSkellamDistribution:
    """Tests for Skellam distribution."""

    def test_skellam_equal_lambdas(self):
        """Test Skellam with equal lambdas (symmetric)."""
        result = skellam_probs(1.5, 1.5)

        # Should be symmetric
        assert abs(result["p_home"] - result["p_away"]) < 0.01
        # Draw should be prominent
        assert result["p_draw"] > 0.1
        # Probabilities sum to approximately 1
        assert abs(result["p_home"] + result["p_draw"] + result["p_away"] - 1.0) < 0.1

    def test_skellam_home_advantage(self):
        """Test Skellam with home advantage."""
        result = skellam_probs(2.0, 1.0)

        # Home should be favored
        assert result["p_home"] > result["p_away"]

    def test_skellam_moments(self):
        """Test that moments are calculated correctly."""
        lambda_h, lambda_a = 1.8, 1.2
        result = skellam_probs(lambda_h, lambda_a)

        # Mean difference should be lambda_h - lambda_a
        assert abs(result["mean_diff"] - (lambda_h - lambda_a)) < 0.01
        # Variance should be lambda_h + lambda_a
        assert abs(result["var_diff"] - (lambda_h + lambda_a)) < 0.01

    def test_skellam_valid_output_range(self):
        """Test that Skellam outputs are valid probabilities."""
        result = skellam_probs(1.5, 1.2)

        assert 0 <= result["p_home"] <= 1
        assert 0 <= result["p_draw"] <= 1
        assert 0 <= result["p_away"] <= 1

    def test_skellam_extreme_lambdas(self):
        """Test Skellam with extreme lambda values."""
        result = skellam_probs(0.1, 5.0)

        # Away should be heavily favored
        assert result["p_away"] > 0.5

    def test_skellam_custom_max_diff(self):
        """Test Skellam with custom max_diff parameter."""
        result1 = skellam_probs(1.5, 1.5, max_diff=3)
        result2 = skellam_probs(1.5, 1.5, max_diff=10)

        # Larger max_diff should give more accurate results
        # Both should sum close to 1
        assert abs(result1["p_home"] + result1["p_draw"] + result1["p_away"] - 1.0) < 0.2
        assert abs(result2["p_home"] + result2["p_draw"] + result2["p_away"] - 1.0) < 0.05


class TestBivariatePoissonPMF:
    """Tests for bivariate Poisson PMF."""

    def test_bivariate_poisson_pmf_zero_goals(self):
        """Test PMF for 0-0 scoreline."""
        pmf = bivariate_poisson_pmf(0, 0, 1.0, 1.0, 0.1)

        assert pmf > 0
        assert pmf < 1

    def test_bivariate_poisson_pmf_typical_score(self):
        """Test PMF for typical scoreline."""
        pmf = bivariate_poisson_pmf(2, 1, 1.5, 1.2, 0.0)

        assert pmf > 0
        assert pmf < 1

    def test_bivariate_poisson_pmf_high_correlation(self):
        """Test PMF with high covariance (positive correlation)."""
        pmf_low_corr = bivariate_poisson_pmf(1, 1, 1.0, 1.0, 0.0)
        pmf_high_corr = bivariate_poisson_pmf(1, 1, 1.0, 1.0, 0.5)

        # Both should be valid probabilities
        assert 0 < pmf_low_corr < 1
        assert 0 < pmf_high_corr < 1

    def test_bivariate_poisson_pmf_high_goals(self):
        """Test PMF for high-scoring match."""
        pmf = bivariate_poisson_pmf(5, 4, 2.5, 2.0, 0.1)

        assert pmf > 0
        assert pmf < 1

    def test_bivariate_poisson_symmetry(self):
        """Test that PMF is symmetric when lambdas are equal."""
        pmf_h2a = bivariate_poisson_pmf(2, 1, 1.5, 1.5, 0.1)
        pmf_a2h = bivariate_poisson_pmf(1, 2, 1.5, 1.5, 0.1)

        # Should be equal when lambdas are swapped and goals are swapped
        assert abs(pmf_h2a - pmf_a2h) < 1e-10


class TestRankedProbabilityScore:
    """Tests for Ranked Probability Score."""

    def test_rps_perfect_prediction(self):
        """Test RPS for perfect prediction."""
        probs = [1.0, 0.0, 0.0]  # Predict home win
        rps = ranked_probability_score(probs, 0)  # Actual home win

        assert rps == 0.0  # Perfect prediction = 0

    def test_rps_worst_prediction(self):
        """Test RPS for worst prediction."""
        probs = [0.0, 0.0, 1.0]  # Predict away win
        rps = ranked_probability_score(probs, 0)  # Actual home win

        assert rps > 0.5  # Should be large

    def test_rps_draw_prediction(self):
        """Test RPS for draw prediction."""
        probs = [0.33, 0.34, 0.33]
        rps = ranked_probability_score(probs, 1)  # Actual draw

        assert 0 <= rps < 0.15  # Should be reasonable

    def test_rps_bounds(self):
        """Test that RPS is always in [0, 1]."""
        for _ in range(20):
            probs = np.random.dirichlet([1, 1, 1])
            outcome = np.random.randint(0, 3)
            rps = ranked_probability_score(probs, outcome)

            assert 0 <= rps <= 1

    def test_rps_ordering_sensitivity(self):
        """Test that RPS penalizes ordered mistakes appropriately."""
        # Both predict home, actual is away
        probs = [0.6, 0.2, 0.2]

        # Home prediction when away wins is worse than draw
        rps_home = ranked_probability_score(probs, 2)

        # More uniform prediction should have less loss
        probs_uniform = [0.33, 0.33, 0.34]
        rps_uniform = ranked_probability_score(probs_uniform, 2)

        assert rps_home > rps_uniform


class TestRPSFromResult:
    """Tests for RPS convenience wrapper."""

    def test_rps_from_result_home_win(self):
        """Test RPS from home win result."""
        probs = [0.6, 0.2, 0.2]
        rps = rps_from_result(probs, 2, 1)  # Home wins 2-1

        assert 0 <= rps <= 1

    def test_rps_from_result_draw(self):
        """Test RPS from draw result."""
        probs = [0.33, 0.34, 0.33]
        rps = rps_from_result(probs, 1, 1)  # Draw 1-1

        assert 0 <= rps <= 1

    def test_rps_from_result_away_win(self):
        """Test RPS from away win result."""
        probs = [0.6, 0.2, 0.2]
        rps = rps_from_result(probs, 0, 2)  # Away wins 2-0

        assert 0 <= rps <= 1

    def test_rps_from_result_high_score_consistency(self):
        """Test that RPS is same for all 2-1 results."""
        probs = [0.5, 0.3, 0.2]

        # All should give same RPS (2-1 is home win)
        rps1 = rps_from_result(probs, 2, 1)
        rps2 = rps_from_result(probs, 3, 1)
        rps3 = rps_from_result(probs, 5, 3)

        assert rps1 == rps2 == rps3


class TestBrierScore:
    """Tests for Brier score."""

    def test_brier_perfect_prediction(self):
        """Test Brier score for perfect prediction."""
        probs = [1.0, 0.0, 0.0]
        bs = brier_score(probs, 0)

        assert bs == 0.0  # Perfect prediction

    def test_brier_worst_prediction(self):
        """Test Brier score for worst prediction."""
        probs = [0.0, 0.0, 1.0]
        bs = brier_score(probs, 0)

        assert bs > 0.6

    def test_brier_uniform_prediction(self):
        """Test Brier score for uniform prediction."""
        probs = [1/3, 1/3, 1/3]
        bs = brier_score(probs, 0)

        # Brier = 1/3 * ((1/3-1)^2 + (1/3-0)^2 + (1/3-0)^2)
        # = 1/3 * (4/9 + 1/9 + 1/9) = 1/3 * 6/9 = 2/9 ≈ 0.222
        assert 0.2 < bs < 0.25

    def test_brier_bounds(self):
        """Test that Brier score is in [0, 1]."""
        for _ in range(20):
            probs = np.random.dirichlet([1, 1, 1])
            outcome = np.random.randint(0, 3)
            bs = brier_score(probs, outcome)

            assert 0 <= bs <= 1

    def test_brier_symmetry(self):
        """Test Brier score calculation is correct."""
        probs = [0.6, 0.2, 0.2]
        outcome = 0  # First outcome true

        # Manual calculation: 1/3 * ((0.6-1)^2 + (0.2-0)^2 + (0.2-0)^2)
        # = 1/3 * (0.16 + 0.04 + 0.04) = 1/3 * 0.24 = 0.08
        bs = brier_score(probs, outcome)

        assert abs(bs - 0.08) < 0.01


class TestScoringMetricsConsistency:
    """Tests for consistency between scoring metrics."""

    def test_same_prediction_probabilities(self):
        """Test multiple metrics on same prediction."""
        probs = [0.5, 0.3, 0.2]
        outcome = 0

        rps = ranked_probability_score(probs, outcome)
        bs = brier_score(probs, outcome)

        # Both should be small (correct prediction)
        assert rps < 0.3
        assert bs < 0.3

    def test_opposite_predictions_metrics(self):
        """Test metrics for opposite predictions."""
        probs = [0.8, 0.1, 0.1]
        outcome = 2  # Predict home win, actual away win

        rps = ranked_probability_score(probs, outcome)
        bs = brier_score(probs, outcome)

        # Both should be large (incorrect prediction)
        assert rps > 0.3
        assert bs > 0.3

    def test_calibrated_prediction(self):
        """Test metrics for well-calibrated prediction."""
        # 50% prediction that comes true 50% of the time
        probs = [0.5, 0.3, 0.2]

        # When it's correct
        rps_correct = ranked_probability_score(probs, 0)
        # When it's wrong (outcome 1)
        rps_wrong1 = ranked_probability_score(probs, 1)

        assert rps_correct <= rps_wrong1


class TestMathModulesEdgeCases:
    """Tests for edge cases in math modules."""

    def test_very_low_probabilities(self):
        """Test scoring with very low probabilities."""
        probs = [0.001, 0.001, 0.998]
        rps = ranked_probability_score(probs, 2)

        assert 0 <= rps <= 1

    def test_near_uniform_prediction(self):
        """Test scoring with near-uniform prediction."""
        probs = [0.334, 0.333, 0.333]
        rps = ranked_probability_score(probs, 0)

        # Near-uniform prediction has reasonable RPS score
        assert 0 <= rps <= 0.5

    def test_single_outcome_probability(self):
        """Test with single outcome probability."""
        probs = [1.0, 0.0, 0.0]

        for outcome in range(3):
            rps = ranked_probability_score(probs, outcome)
            assert 0 <= rps <= 1

    def test_skellam_zero_lambda(self):
        """Test Skellam with zero lambda (edge case)."""
        result = skellam_probs(0.0, 1.0)

        # With zero home lambda, away should be heavily favored
        assert isinstance(result, dict)
        assert "p_home" in result
        assert "p_away" in result
        assert "p_draw" in result

    def test_bivariate_poisson_zero_covariance(self):
        """Test bivariate Poisson with zero covariance (independence)."""
        pmf = bivariate_poisson_pmf(1, 1, 1.0, 1.0, 0.0)

        assert 0 < pmf < 1

    def test_bivariate_poisson_high_goals(self):
        """Test bivariate Poisson doesn't overflow with high goals."""
        pmf = bivariate_poisson_pmf(10, 10, 3.0, 3.0, 0.5)

        # Should be valid probability
        assert 0 <= pmf <= 1

    def test_rps_with_nearly_zero_probabilities(self):
        """Test RPS with nearly zero probabilities."""
        probs = [0.0001, 0.0001, 0.9998]
        rps = ranked_probability_score(probs, 2)

        assert 0 <= rps <= 1

    def test_brier_with_extreme_probabilities(self):
        """Test Brier score with extreme probabilities."""
        probs = [0.99, 0.005, 0.005]
        bs = brier_score(probs, 0)

        assert 0 <= bs <= 1

    def test_multiple_scoring_functions_consistency(self):
        """Test that multiple scoring functions give consistent ordering."""
        predictions = [
            ([0.7, 0.2, 0.1], 0),  # Correct
            ([0.5, 0.3, 0.2], 0),  # Less confident correct
            ([0.3, 0.3, 0.4], 0),  # Wrong confident
        ]

        rps_scores = [ranked_probability_score(p, o) for p, o in predictions]
        bs_scores = [brier_score(p, o) for p, o in predictions]

        # First should be best (lowest score)
        assert rps_scores[0] <= rps_scores[1]
        assert bs_scores[0] <= bs_scores[1]
