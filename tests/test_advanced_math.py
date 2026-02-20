"""Tests for the advanced_math module — v10 mathematical foundations."""
import math
import pytest
import numpy as np


class TestBetaBinomialShrinkage:
    """Test empirical Bayes Beta-Binomial shrinkage."""

    def test_basic_shrinkage(self):
        from footy.models.advanced_math import beta_binomial_shrink
        # 5 wins in 10 games, uniform prior
        result = beta_binomial_shrink(5, 10, 2.0, 2.0)
        assert 0.4 < result < 0.6  # shrunk towards prior mean 0.5

    def test_perfect_record_shrunk(self):
        from footy.models.advanced_math import beta_binomial_shrink
        # Perfect record should be shrunk towards prior
        result = beta_binomial_shrink(5, 5, 2.0, 2.0)
        assert result < 1.0  # not 100%
        assert result > 0.7  # but still high

    def test_zero_record_shrunk(self):
        from footy.models.advanced_math import beta_binomial_shrink
        # No wins should still be above 0
        result = beta_binomial_shrink(0, 5, 2.0, 2.0)
        assert result > 0.0
        assert result < 0.3

    def test_large_sample_converges(self):
        from footy.models.advanced_math import beta_binomial_shrink
        # With many observations, shrinkage diminishes
        result = beta_binomial_shrink(70, 100, 2.0, 2.0)
        assert abs(result - 0.70) < 0.05  # close to observed rate

    def test_league_specific_priors(self):
        from footy.models.advanced_math import league_specific_prior
        # All known leagues should return valid priors
        for league in ["PL", "PD", "SA", "BL1", "FL1"]:
            alpha, beta = league_specific_prior(league, "home_win")
            assert alpha > 0
            assert beta > 0
            assert alpha + beta == pytest.approx(10.0)

    def test_unknown_league_returns_default(self):
        from footy.models.advanced_math import league_specific_prior
        alpha, beta = league_specific_prior("UNKNOWN", "home_win")
        assert alpha > 0
        assert beta > 0


class TestDixonColesTau:
    """Test the Dixon-Coles τ correction factor."""

    def test_tau_00(self):
        from footy.models.advanced_math import dixon_coles_tau
        # 0-0: τ = 1 - λ·μ·ρ, with ρ < 0 → τ > 1 (increases 0-0 probability)
        tau = dixon_coles_tau(0, 0, 1.5, 1.2, rho=-0.13)
        assert tau > 1.0

    def test_tau_10(self):
        from footy.models.advanced_math import dixon_coles_tau
        # 1-0: τ = 1 + λ·ρ, with ρ < 0 → τ < 1 (decreases 1-0 probability)
        tau = dixon_coles_tau(1, 0, 1.5, 1.2, rho=-0.13)
        assert tau < 1.0

    def test_tau_01(self):
        from footy.models.advanced_math import dixon_coles_tau
        tau = dixon_coles_tau(0, 1, 1.5, 1.2, rho=-0.13)
        assert tau < 1.0

    def test_tau_11(self):
        from footy.models.advanced_math import dixon_coles_tau
        # 1-1: τ = 1 - ρ, with ρ < 0 → τ > 1
        tau = dixon_coles_tau(1, 1, 1.5, 1.2, rho=-0.13)
        assert tau > 1.0

    def test_tau_high_scores_unchanged(self):
        from footy.models.advanced_math import dixon_coles_tau
        # Higher scores: τ = 1 (no adjustment)
        for hg, ag in [(2, 0), (2, 1), (3, 2), (0, 3)]:
            assert dixon_coles_tau(hg, ag, 1.5, 1.2, rho=-0.13) == 1.0

    def test_tau_non_negative(self):
        from footy.models.advanced_math import dixon_coles_tau
        # τ should never be negative even with extreme ρ
        for rho in [-0.5, -1.0, 0.5]:
            for hg, ag in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                assert dixon_coles_tau(hg, ag, 2.0, 2.0, rho) >= 0.0

    def test_build_dc_score_matrix(self):
        from footy.models.advanced_math import build_dc_score_matrix
        mx = build_dc_score_matrix(1.5, 1.2, rho=-0.13, max_goals=6)
        assert mx.shape == (7, 7)
        assert abs(mx.sum() - 1.0) < 1e-8  # normalized
        assert np.all(mx >= 0)  # non-negative probabilities

    def test_dc_matrix_higher_00_than_poisson(self):
        from footy.models.advanced_math import build_dc_score_matrix
        from scipy.stats import poisson as poisson_dist
        # DC should have higher 0-0 probability than independent Poisson (ρ < 0)
        dc_mx = build_dc_score_matrix(1.3, 1.1, rho=-0.13, max_goals=6)
        h_dist = poisson_dist.pmf(range(7), 1.3)
        a_dist = poisson_dist.pmf(range(7), 1.1)
        pois_00 = h_dist[0] * a_dist[0]
        assert dc_mx[0, 0] > pois_00


class TestSkellam:
    """Test Skellam distribution features."""

    def test_skellam_probs_sum_to_one(self):
        from footy.models.advanced_math import skellam_probs
        result = skellam_probs(1.5, 1.2)
        total = result["p_home"] + result["p_draw"] + result["p_away"]
        assert abs(total - 1.0) < 0.01

    def test_skellam_mean_diff(self):
        from footy.models.advanced_math import skellam_probs
        result = skellam_probs(2.0, 1.0)
        assert abs(result["mean_diff"] - 1.0) < 0.01

    def test_skellam_variance(self):
        from footy.models.advanced_math import skellam_probs
        result = skellam_probs(2.0, 1.0)
        assert abs(result["var_diff"] - 3.0) < 0.01  # var = λ1 + λ2


class TestMonteCarlo:
    """Test Monte Carlo simulation."""

    def test_mc_probabilities_sum_to_one(self):
        from footy.models.advanced_math import monte_carlo_simulate
        result = monte_carlo_simulate(1.5, 1.2, rho=-0.13, n_sims=5000)
        total = result["mc_p_home"] + result["mc_p_draw"] + result["mc_p_away"]
        assert abs(total - 1.0) < 0.01

    def test_mc_btts_in_range(self):
        from footy.models.advanced_math import monte_carlo_simulate
        result = monte_carlo_simulate(1.5, 1.2, rho=-0.13, n_sims=5000)
        assert 0.0 <= result["mc_btts"] <= 1.0

    def test_mc_o25_in_range(self):
        from footy.models.advanced_math import monte_carlo_simulate
        result = monte_carlo_simulate(1.5, 1.2, rho=-0.13, n_sims=5000)
        assert 0.0 <= result["mc_o25"] <= 1.0

    def test_mc_consistent_with_poisson(self):
        from footy.models.advanced_math import monte_carlo_simulate
        # MC results should be roughly consistent with analytical Poisson
        result = monte_carlo_simulate(1.8, 0.8, rho=0.0, n_sims=10000)
        # Home should win much more often than away
        assert result["mc_p_home"] > result["mc_p_away"]


class TestLogitSpace:
    """Test logit-space probability manipulation."""

    def test_logit_roundtrip(self):
        from footy.models.advanced_math import logit, inv_logit
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert abs(inv_logit(logit(p)) - p) < 1e-8

    def test_logit_space_delta(self):
        from footy.models.advanced_math import logit_space_delta
        # Model thinks higher probability → positive delta (value)
        delta = logit_space_delta(0.6, 0.5)
        assert delta > 0

    def test_remove_overround(self):
        from footy.models.advanced_math import remove_overround
        # Typical odds with overround
        fair = remove_overround([2.5, 3.3, 2.8])
        assert abs(sum(fair) - 1.0) < 1e-8

    def test_odds_entropy(self):
        from footy.models.advanced_math import odds_entropy
        # Uniform distribution has maximum entropy
        uniform = odds_entropy([1/3, 1/3, 1/3])
        skewed = odds_entropy([0.8, 0.1, 0.1])
        assert uniform > skewed


class TestNonlinearTransforms:
    """Test nonlinear strength transforms."""

    def test_tanh_transform_bounded(self):
        from footy.models.advanced_math import tanh_transform
        # Should be bounded in [-1, 1]
        assert -1.0 <= tanh_transform(500.0, 400.0) <= 1.0
        assert -1.0 <= tanh_transform(-500.0, 400.0) <= 1.0

    def test_tanh_transform_zero(self):
        from footy.models.advanced_math import tanh_transform
        assert tanh_transform(0.0, 400.0) == 0.0

    def test_log_transform_sign_preserving(self):
        from footy.models.advanced_math import log_transform
        assert log_transform(100.0) > 0
        assert log_transform(-100.0) < 0
        assert log_transform(0.0) == 0.0


class TestKLDivergence:
    """Test KL divergence computation."""

    def test_kl_identical_distributions(self):
        from footy.models.advanced_math import kl_divergence
        # KL(P || P) = 0
        kl = kl_divergence([0.5, 0.3, 0.2], [0.5, 0.3, 0.2])
        assert abs(kl) < 1e-8

    def test_kl_different_distributions(self):
        from footy.models.advanced_math import kl_divergence
        kl = kl_divergence([0.8, 0.1, 0.1], [0.4, 0.3, 0.3])
        assert kl > 0

    def test_kl_non_negative(self):
        from footy.models.advanced_math import kl_divergence
        kl = kl_divergence([0.6, 0.2, 0.2], [0.3, 0.4, 0.3])
        assert kl >= 0


class TestScheduleDifficulty:
    """Test schedule difficulty."""

    def test_basic_schedule_difficulty(self):
        from footy.models.advanced_math import schedule_difficulty
        strengths = [1600, 1500, 1400, 1550, 1650]
        result = schedule_difficulty(strengths, window=3)
        # Composite index: weighted by recency → hard recent schedule lifts result
        # Must be above simple mean (1533) because 1650 is most recent + max-opp boost
        assert result > np.mean([1400, 1550, 1650])
        # Must stay in realistic Elo range
        assert 1200 < result < 1800

    def test_empty_schedule(self):
        from footy.models.advanced_math import schedule_difficulty
        assert schedule_difficulty([], window=5) == 1500.0


class TestEWMA:
    """Test adaptive EWMA."""

    def test_ewma_basic(self):
        from footy.models.advanced_math import adaptive_ewma
        result = adaptive_ewma([1.0, 2.0, 3.0])
        assert result > 1.0  # weighted towards recent

    def test_ewma_empty(self):
        from footy.models.advanced_math import adaptive_ewma
        assert adaptive_ewma([]) == 0.0

    def test_ewma_single(self):
        from footy.models.advanced_math import adaptive_ewma
        assert adaptive_ewma([5.0]) == 5.0
