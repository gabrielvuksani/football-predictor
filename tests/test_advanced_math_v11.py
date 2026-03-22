"""Tests for v11 advanced math improvements.

Covers: Bivariate Poisson, Frank Copula, RPS, COM-Poisson,
        Bradley-Terry, Platt Scaling, JSD, extract_match_probs.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from footy.models.advanced_math import (
    bivariate_poisson_pmf,
    build_bivariate_poisson_matrix,
    build_copula_score_matrix,
    build_dc_score_matrix,
    build_com_poisson_matrix,
    com_poisson_pmf,
    ranked_probability_score,
    rps_from_result,
    bradley_terry_probs,
    platt_scale,
    find_optimal_temperature,
    extract_match_probs,
    jensen_shannon_divergence,
    multi_expert_jsd,
)


# ═══════════════════════════════════════════════════════════════════
# BIVARIATE POISSON
# ═══════════════════════════════════════════════════════════════════
class TestBivariatePoisson:
    def test_pmf_nonnegative(self):
        for x in range(5):
            for y in range(5):
                p = bivariate_poisson_pmf(x, y, 1.0, 0.8, 0.15)
                assert p >= 0, f"pmf({x},{y}) = {p}"

    def test_pmf_sums_approx_one(self):
        total = sum(
            bivariate_poisson_pmf(x, y, 1.2, 0.9, 0.1)
            for x in range(12) for y in range(12)
        )
        assert abs(total - 1.0) < 0.01

    def test_lambda3_zero_is_independent(self):
        """λ₃ = 0 should give independent Poisson."""
        from scipy.stats import poisson
        for x in range(4):
            for y in range(4):
                bp = bivariate_poisson_pmf(x, y, 1.5, 1.1, 0.0)
                indep = poisson.pmf(x, 1.5) * poisson.pmf(y, 1.1)
                assert abs(bp - indep) < 1e-6

    def test_matrix_sums_to_one(self):
        mat = build_bivariate_poisson_matrix(1.5, 1.1, 0.12)
        assert abs(mat.sum() - 1.0) < 1e-6

    def test_matrix_shape(self):
        mat = build_bivariate_poisson_matrix(1.3, 0.9, 0.1, max_goals=6)
        assert mat.shape == (7, 7)

    def test_positive_correlation_increases_draws(self):
        """Higher λ₃ should increase probability of equal-score draws."""
        mat_low = build_bivariate_poisson_matrix(1.5, 1.5, lambda_3=0.0)
        mat_high = build_bivariate_poisson_matrix(1.5, 1.5, lambda_3=0.3)
        draw_low = float(np.trace(mat_low))
        draw_high = float(np.trace(mat_high))
        assert draw_high > draw_low


# ═══════════════════════════════════════════════════════════════════
# FRANK COPULA
# ═══════════════════════════════════════════════════════════════════
class TestFrankCopula:
    def test_matrix_sums_to_one(self):
        mat = build_copula_score_matrix(1.5, 1.1, theta=-2.0)
        assert abs(mat.sum() - 1.0) < 1e-6

    def test_independence_theta_zero(self):
        """θ = 0 should approximate independent Poisson."""
        mat_cop = build_copula_score_matrix(1.5, 1.1, theta=0.0)
        mat_ind = build_dc_score_matrix(1.5, 1.1, rho=0.0)  # rho=0 = independent
        # Should be close
        assert np.allclose(mat_cop, mat_ind, atol=0.005)

    def test_negative_theta_changes_draw_prob(self):
        """Negative θ (like negative ρ) should affect draw probability."""
        mat_indep = build_copula_score_matrix(1.5, 1.1, theta=0.0)
        mat_neg = build_copula_score_matrix(1.5, 1.1, theta=-3.0)
        # Probabilities should differ
        p_home_0 = float(np.tril(mat_indep, -1).sum())
        p_home_n = float(np.tril(mat_neg, -1).sum())
        assert abs(p_home_0 - p_home_n) > 0.001

    def test_nonnegative_entries(self):
        mat = build_copula_score_matrix(1.3, 0.9, theta=-2.5)
        assert (mat >= 0).all()


# ═══════════════════════════════════════════════════════════════════
# RANKED PROBABILITY SCORE
# ═══════════════════════════════════════════════════════════════════
class TestRPS:
    def test_perfect_prediction(self):
        """Perfect prediction should have RPS close to 0."""
        rps = ranked_probability_score([1.0, 0.0, 0.0], 0)
        assert rps == 0.0

    def test_worst_prediction(self):
        """Predicting opposite extreme should have max RPS."""
        rps = ranked_probability_score([1.0, 0.0, 0.0], 2)
        assert rps == 1.0

    def test_ordering_matters(self):
        """RPS should penalize distance: draw pred when home wins < away pred when home wins."""
        rps_draw = ranked_probability_score([0.1, 0.8, 0.1], 0)  # predict draw, actual home
        rps_away = ranked_probability_score([0.1, 0.1, 0.8], 0)  # predict away, actual home
        assert rps_draw < rps_away

    def test_rps_bounds(self):
        """RPS should be in [0, 1]."""
        for _ in range(100):
            probs = np.random.dirichlet([1, 1, 1]).tolist()
            outcome = np.random.randint(0, 3)
            rps = ranked_probability_score(probs, outcome)
            assert 0.0 <= rps <= 1.0

    def test_rps_from_result_home_win(self):
        rps = rps_from_result([0.7, 0.2, 0.1], 3, 1)
        assert rps < 0.1  # good prediction, low RPS

    def test_rps_from_result_draw(self):
        rps = rps_from_result([0.2, 0.6, 0.2], 1, 1)
        assert rps < 0.15

    def test_rps_symmetric_for_uniform(self):
        """Uniform prediction should give same RPS for any outcome."""
        rps_h = ranked_probability_score([1/3, 1/3, 1/3], 0)
        rps_a = ranked_probability_score([1/3, 1/3, 1/3], 2)
        assert abs(rps_h - rps_a) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# COM-POISSON
# ═══════════════════════════════════════════════════════════════════
class TestCOMPoisson:
    def test_nu_one_is_poisson(self):
        """ν=1 should give standard Poisson PMF."""
        from scipy.stats import poisson
        for k in range(6):
            cmp = com_poisson_pmf(k, 1.5, nu=1.0)
            std = poisson.pmf(k, 1.5)
            assert abs(cmp - std) < 0.01, f"k={k}: CMP={cmp:.4f}, Poisson={std:.4f}"

    def test_overdispersed_has_heavier_tails(self):
        """ν < 1 (over-dispersed) should have heavier tails than Poisson."""
        p_std_5 = com_poisson_pmf(5, 1.5, nu=1.0)
        p_over_5 = com_poisson_pmf(5, 1.5, nu=0.8)
        assert p_over_5 > p_std_5

    def test_matrix_sums_to_one(self):
        mat = build_com_poisson_matrix(1.3, 1.0, nu_h=0.9, nu_a=0.95)
        assert abs(mat.sum() - 1.0) < 1e-5

    def test_nonnegative(self):
        for k in range(10):
            assert com_poisson_pmf(k, 2.0, nu=0.85) >= 0


# ═══════════════════════════════════════════════════════════════════
# BRADLEY-TERRY
# ═══════════════════════════════════════════════════════════════════
class TestBradleyTerry:
    def test_sums_to_one(self):
        ph, pd, pa = bradley_terry_probs(0.5, -0.3, 0.25, 0.3)
        assert abs(ph + pd + pa - 1.0) < 1e-10

    def test_home_advantage(self):
        """With equal strength, home team should be favoured."""
        ph, _, pa = bradley_terry_probs(0.0, 0.0, home_adv=0.3)
        assert ph > pa

    def test_stronger_team_wins(self):
        """Much stronger home team should have high probability."""
        ph, _, _ = bradley_terry_probs(2.0, -1.0, 0.3, 0.2)
        assert ph > 0.7

    def test_equal_no_home_is_symmetric(self):
        """Equal teams with no home advantage, draw factor → symmetric."""
        ph, pd, pa = bradley_terry_probs(0.0, 0.0, home_adv=0.0, draw_factor=0.3)
        assert abs(ph - pa) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# PLATT SCALING
# ═══════════════════════════════════════════════════════════════════
class TestPlattScaling:
    def test_identity_at_temp_one(self):
        probs = np.array([0.6, 0.25, 0.15])
        scaled = platt_scale(probs, temperature=1.0)
        assert np.allclose(probs, scaled, atol=0.005)

    def test_sharper_at_low_temp(self):
        probs = np.array([0.6, 0.25, 0.15])
        scaled = platt_scale(probs, temperature=0.5)
        assert scaled[0] > probs[0]  # max gets more confident

    def test_softer_at_high_temp(self):
        probs = np.array([0.6, 0.25, 0.15])
        scaled = platt_scale(probs, temperature=2.0)
        assert scaled[0] < probs[0]  # max gets less confident

    def test_sums_to_one(self):
        probs = np.array([[0.7, 0.2, 0.1], [0.33, 0.34, 0.33]])
        for t in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
            scaled = platt_scale(probs, temperature=t)
            for row in scaled:
                assert abs(row.sum() - 1.0) < 1e-6

    def test_batch_mode(self):
        probs = np.array([[0.5, 0.3, 0.2], [0.8, 0.1, 0.1]])
        scaled = platt_scale(probs, temperature=1.5)
        assert scaled.shape == (2, 3)

    def test_optimal_temperature(self):
        """Optimal temperature should be between bounds."""
        np.random.seed(42)
        probs = np.random.dirichlet([2, 1, 1], size=50)
        labels = np.random.randint(0, 3, size=50)
        t_opt = find_optimal_temperature(probs, labels)
        assert 0.5 <= t_opt <= 3.0


# ═══════════════════════════════════════════════════════════════════
# EXTRACT MATCH PROBS
# ═══════════════════════════════════════════════════════════════════
class TestExtractMatchProbs:
    def test_from_dc_matrix(self):
        mat = build_dc_score_matrix(1.5, 1.1, rho=-0.13)
        p = extract_match_probs(mat)
        assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1.0) < 1e-6
        assert 0 <= p["p_btts"] <= 1
        assert 0 <= p["p_o25"] <= 1
        assert p["eg_home"] > 0
        assert p["eg_away"] > 0

    def test_from_copula_matrix(self):
        mat = build_copula_score_matrix(1.5, 1.1, theta=-2.0)
        p = extract_match_probs(mat)
        assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1.0) < 1e-6

    def test_dispersion_index(self):
        """Standard Poisson should have dispersion ≈ 1.0."""
        mat = build_dc_score_matrix(1.5, 1.1, rho=0.0)
        p = extract_match_probs(mat)
        assert 0.8 < p["disp_home"] < 1.2  # close to 1 for Poisson

    def test_most_likely_score(self):
        mat = build_dc_score_matrix(2.5, 0.5, rho=-0.13)
        p = extract_match_probs(mat)
        assert p["ml_score_h"] >= 1  # strong home team, >0 goals likely
        assert p["eg_home"] > p["eg_away"]


# ═══════════════════════════════════════════════════════════════════
# JENSEN-SHANNON DIVERGENCE
# ═══════════════════════════════════════════════════════════════════
class TestJSD:
    def test_same_distribution_is_zero(self):
        jsd = jensen_shannon_divergence([0.5, 0.3, 0.2], [0.5, 0.3, 0.2])
        assert abs(jsd) < 1e-10

    def test_symmetric(self):
        p = [0.7, 0.2, 0.1]
        q = [0.3, 0.4, 0.3]
        assert abs(jensen_shannon_divergence(p, q) - jensen_shannon_divergence(q, p)) < 1e-10

    def test_bounded(self):
        """JSD should be in [0, ln(2)]."""
        jsd = jensen_shannon_divergence([0.9, 0.05, 0.05], [0.05, 0.05, 0.9])
        assert 0 <= jsd <= math.log(2) + 1e-10

    def test_more_different_is_higher(self):
        jsd_close = jensen_shannon_divergence([0.5, 0.3, 0.2], [0.45, 0.35, 0.2])
        jsd_far = jensen_shannon_divergence([0.5, 0.3, 0.2], [0.1, 0.1, 0.8])
        assert jsd_far > jsd_close


class TestMultiExpertJSD:
    def test_consensus_is_low(self):
        """All experts agree → JSD ≈ 0."""
        jsd = multi_expert_jsd([
            [0.5, 0.3, 0.2],
            [0.5, 0.3, 0.2],
            [0.5, 0.3, 0.2],
        ])
        assert abs(jsd) < 1e-10

    def test_disagreement_is_high(self):
        jsd = multi_expert_jsd([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        assert jsd > 0.1

    def test_nonnegative(self):
        for _ in range(50):
            experts = [np.random.dirichlet([1, 1, 1]).tolist() for _ in range(5)]
            assert multi_expert_jsd(experts) >= -1e-10
