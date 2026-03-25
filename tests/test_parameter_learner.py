"""Tests for data-driven parameter learning module."""

import numpy as np
import pytest

from footy.models.parameter_learner import (
    learn_com_poisson_nu,
    learn_copula_theta,
    learn_gas_params,
    learn_hmm_params,
    _default_hmm_params,
    _poisson_pmf,
)


# ---------------------------------------------------------------------------
# Helper sanity
# ---------------------------------------------------------------------------

class TestPoissonPmf:
    def test_sums_approximately_to_one(self):
        lam = 2.5
        total = sum(_poisson_pmf(k, lam) for k in range(30))
        assert abs(total - 1.0) < 1e-6

    def test_zero_rate(self):
        assert _poisson_pmf(0, 0.0) == 1.0
        assert _poisson_pmf(3, 0.0) == 0.0


class TestDefaultHmm:
    def test_shapes(self):
        params = _default_hmm_params(3)
        assert params["emission_rates"].shape == (3,)
        assert params["transition_matrix"].shape == (3, 3)

    def test_rows_sum_to_one(self):
        params = _default_hmm_params(4)
        row_sums = params["transition_matrix"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 1. HMM EM
# ---------------------------------------------------------------------------

class TestHmmEm:
    def test_hmm_em_learns_from_data(self):
        """Synthetic 3-phase goal sequence: low(0-1), medium(1-2), high(2-4).

        The EM algorithm should recover emission rates that span the
        range, with the highest state > 1.5.
        """
        rng = np.random.default_rng(42)
        # Phase 1: low scoring (rate ~0.5)
        phase1 = rng.poisson(0.5, size=50).tolist()
        # Phase 2: medium scoring (rate ~1.5)
        phase2 = rng.poisson(1.5, size=50).tolist()
        # Phase 3: high scoring (rate ~3.0)
        phase3 = rng.poisson(3.0, size=50).tolist()

        goals = phase1 + phase2 + phase3

        result = learn_hmm_params(goals, n_states=3, n_iter=100)

        assert "emission_rates" in result
        assert "transition_matrix" in result
        assert result["emission_rates"].shape == (3,)
        assert result["transition_matrix"].shape == (3, 3)

        # States should be sorted ascending
        rates = result["emission_rates"]
        assert np.all(rates[:-1] <= rates[1:]), "Rates should be sorted ascending"

        # Highest state should capture the high-scoring phase
        assert rates[-1] > 1.5, f"Max emission rate {rates[-1]} should exceed 1.5"

        # Transition matrix rows should sum to 1
        row_sums = result["transition_matrix"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_hmm_returns_defaults_for_small_data(self):
        result = learn_hmm_params([1, 0, 2])
        defaults = _default_hmm_params(3)
        np.testing.assert_array_equal(
            result["emission_rates"], defaults["emission_rates"]
        )


# ---------------------------------------------------------------------------
# 2. GAS MLE
# ---------------------------------------------------------------------------

class TestGasMle:
    def test_gas_mle_learns_persistence(self):
        """A stable team scoring ~1.5 goals/match should yield high B
        (autoregressive persistence close to 1).
        """
        rng = np.random.default_rng(123)
        goals = rng.poisson(1.5, size=200)

        result = learn_gas_params(goals, home=True)

        assert "omega" in result
        assert "A" in result
        assert "B" in result

        # B should be high (persistent intensity) for stable data
        assert 0.8 < result["B"] < 1.0, (
            f"B={result['B']:.4f} should indicate persistence (0.8, 1.0)"
        )

        # A should be small for stable data
        assert result["A"] < 0.3, f"A={result['A']:.4f} should be small for stable data"

    def test_gas_returns_defaults_for_small_data(self):
        result = learn_gas_params(np.array([1, 2, 0]), home=True)
        assert result == {"omega": 0.15, "A": 0.05, "B": 0.95}

    def test_gas_away_defaults(self):
        result = learn_gas_params(np.array([1]), home=False)
        assert result["omega"] == 0.10


# ---------------------------------------------------------------------------
# 3. Copula theta
# ---------------------------------------------------------------------------

class TestCopulaTheta:
    def test_copula_theta_from_goals(self):
        """Copula theta should be a finite float in a reasonable range."""
        rng = np.random.default_rng(99)
        home = rng.poisson(1.4, size=100)
        away = rng.poisson(1.1, size=100)

        theta = learn_copula_theta(home, away, family="frank")

        assert isinstance(theta, float)
        assert -10.0 < theta < 10.0, f"Frank theta={theta} out of range"

    def test_copula_clayton(self):
        rng = np.random.default_rng(77)
        home = rng.poisson(1.5, size=100)
        away = rng.poisson(1.2, size=100)

        theta = learn_copula_theta(home, away, family="clayton")
        assert isinstance(theta, float)
        assert 0.01 <= theta <= 20.0

    def test_copula_gumbel(self):
        rng = np.random.default_rng(55)
        home = rng.poisson(1.5, size=100)
        away = rng.poisson(1.2, size=100)

        theta = learn_copula_theta(home, away, family="gumbel")
        assert isinstance(theta, float)
        assert 1.0 <= theta <= 20.0

    def test_copula_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown copula family"):
            learn_copula_theta(
                np.array([1, 2, 3] * 20),
                np.array([0, 1, 2] * 20),
                family="unknown",
            )

    def test_copula_returns_default_for_small_data(self):
        theta = learn_copula_theta(np.array([1, 2]), np.array([0, 1]))
        assert theta == -1.0  # Frank default


# ---------------------------------------------------------------------------
# 4. COM-Poisson nu
# ---------------------------------------------------------------------------

class TestComPoissonNu:
    def test_com_poisson_nu(self):
        """nu should be a float in [0.3, 3.0] for typical football data."""
        rng = np.random.default_rng(42)
        goals = rng.poisson(1.3, size=200)

        nu = learn_com_poisson_nu(goals)

        assert isinstance(nu, float)
        assert 0.3 <= nu <= 3.0, f"nu={nu} outside [0.3, 3.0]"

    def test_com_poisson_nu_overdispersed(self):
        """Negative-binomial-like data (over-dispersed) should give nu < 1."""
        rng = np.random.default_rng(10)
        # Mix of low and high to create over-dispersion
        goals = np.concatenate([
            rng.poisson(0.5, size=100),
            rng.poisson(3.5, size=100),
        ])
        nu = learn_com_poisson_nu(goals)
        assert nu < 1.0, f"Over-dispersed data should give nu < 1, got {nu}"

    def test_com_poisson_returns_default_for_small_data(self):
        nu = learn_com_poisson_nu(np.array([1, 0, 2]))
        assert nu == 0.93


# ---------------------------------------------------------------------------
# 5. Defaults on small data — comprehensive
# ---------------------------------------------------------------------------

class TestDefaultsOnSmallData:
    """All functions should return sensible defaults when given fewer than
    20 observations (HMM) or 30 observations (GAS, copula, COM-Poisson).
    """

    def test_hmm_defaults(self):
        result = learn_hmm_params([1, 2])
        assert result["emission_rates"].shape == (3,)
        assert result["transition_matrix"].shape == (3, 3)
        # Should match the default helper
        defaults = _default_hmm_params(3)
        np.testing.assert_array_equal(
            result["emission_rates"], defaults["emission_rates"]
        )

    def test_gas_defaults(self):
        result = learn_gas_params(np.array([1, 0]), home=True)
        assert result == {"omega": 0.15, "A": 0.05, "B": 0.95}

    def test_copula_defaults(self):
        theta = learn_copula_theta(np.array([1]), np.array([0]))
        assert theta == -1.0

    def test_com_poisson_defaults(self):
        nu = learn_com_poisson_nu(np.array([2, 1]))
        assert nu == 0.93
