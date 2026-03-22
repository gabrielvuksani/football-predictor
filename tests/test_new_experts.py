"""Tests for v13 new expert models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Build a minimal test DataFrame that all experts can process
def _make_test_df(n=50):
    """Create a minimal DataFrame with required columns for expert testing."""
    np.random.seed(42)
    teams = ["TeamA", "TeamB", "TeamC", "TeamD", "TeamE", "TeamF"]
    comps = ["PL", "PD"]
    rows = []
    for i in range(n):
        ht = teams[i % len(teams)]
        at = teams[(i + 1) % len(teams)]
        comp = comps[i % len(comps)]
        date = f"2024-01-{(i % 28) + 1:02d}"
        if i < n - 5:
            # Finished matches
            hg = np.random.poisson(1.3)
            ag = np.random.poisson(1.1)
        else:
            # Upcoming matches (NaN goals)
            hg = float('nan')
            ag = float('nan')
        rows.append({
            "utc_date": date,
            "competition": comp,
            "home_team": ht,
            "away_team": at,
            "home_goals": hg,
            "away_goals": ag,
        })
    return pd.DataFrame(rows)


class TestCopulaExpert:
    def test_basic_compute(self):
        from footy.models.experts.copula_expert import CopulaExpert
        expert = CopulaExpert()
        df = _make_test_df()
        result = expert.compute(df)
        assert result.probs.shape == (len(df), 3)
        assert result.confidence.shape == (len(df),)
        assert np.all(result.probs >= 0)
        assert np.allclose(result.probs.sum(axis=1), 1.0, atol=0.01)

    def test_features_present(self):
        from footy.models.experts.copula_expert import CopulaExpert
        expert = CopulaExpert()
        result = expert.compute(_make_test_df())
        expected_features = ["cop_lambda_h", "cop_lambda_a", "cop_p00", "cop_btts", "cop_o25", "cop_entropy"]
        for feat in expected_features:
            assert feat in result.features, f"Missing feature: {feat}"


class TestDoublePoissonExpert:
    def test_basic_compute(self):
        from footy.models.experts.double_poisson_expert import DoublePoissonExpert
        expert = DoublePoissonExpert()
        df = _make_test_df()
        result = expert.compute(df)
        assert result.probs.shape == (len(df), 3)
        assert np.all(result.probs >= 0)
        assert np.allclose(result.probs.sum(axis=1), 1.0, atol=0.01)

    def test_features_present(self):
        from footy.models.experts.double_poisson_expert import DoublePoissonExpert
        expert = DoublePoissonExpert()
        result = expert.compute(_make_test_df())
        for feat in ["dpois_lambda_h", "dpois_phi_h", "dpois_p00", "dpois_btts"]:
            assert feat in result.features


class TestWeibullExpert:
    def test_basic_compute(self):
        from footy.models.experts.weibull_expert import WeibullExpert
        expert = WeibullExpert()
        df = _make_test_df()
        result = expert.compute(df)
        assert result.probs.shape == (len(df), 3)
        assert np.all(result.probs >= 0)
        assert np.allclose(result.probs.sum(axis=1), 1.0, atol=0.01)

    def test_features_present(self):
        from footy.models.experts.weibull_expert import WeibullExpert
        expert = WeibullExpert()
        result = expert.compute(_make_test_df())
        for feat in ["weib_lambda_h", "weib_c_h", "weib_p00"]:
            assert feat in result.features


class TestMotivationExpert:
    def test_basic_compute(self):
        from footy.models.experts.motivation_expert import MotivationExpert
        expert = MotivationExpert()
        df = _make_test_df()
        result = expert.compute(df)
        assert result.probs.shape == (len(df), 3)
        assert np.all(result.probs >= 0)
        assert np.allclose(result.probs.sum(axis=1), 1.0, atol=0.01)

    def test_features_present(self):
        from footy.models.experts.motivation_expert import MotivationExpert
        expert = MotivationExpert()
        result = expert.compute(_make_test_df())
        for feat in ["mot_home_motivation", "mot_away_motivation", "mot_motivation_diff", "mot_season_progress"]:
            assert feat in result.features


class TestSquadRotationExpert:
    def test_basic_compute(self):
        from footy.models.experts.squad_rotation_expert import SquadRotationExpert
        expert = SquadRotationExpert()
        df = _make_test_df()
        result = expert.compute(df)
        assert result.probs.shape == (len(df), 3)
        assert np.all(result.probs >= 0)
        assert np.allclose(result.probs.sum(axis=1), 1.0, atol=0.01)

    def test_features_present(self):
        from footy.models.experts.squad_rotation_expert import SquadRotationExpert
        expert = SquadRotationExpert()
        result = expert.compute(_make_test_df())
        for feat in ["rot_home_days_rest", "rot_away_days_rest", "rot_congestion_diff"]:
            assert feat in result.features


class TestAggregatorNewFunctions:
    def test_thompson_sampling_weights(self):
        from footy.prediction_aggregator import thompson_sampling_weights
        w = thompson_sampling_weights(
            {"a": 10, "b": 5, "c": 8},
            {"a": 2, "b": 5, "c": 3}
        )
        assert len(w) == 3
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_online_weight_update(self):
        from footy.prediction_aggregator import online_weight_update
        initial = {"council": 0.5, "bayesian": 0.3, "market": 0.2}
        prediction = {
            "council": (0.6, 0.2, 0.2),
            "bayesian": (0.4, 0.3, 0.3),
            "market": (0.5, 0.25, 0.25),
        }
        updated = online_weight_update(initial, prediction, 0)
        assert len(updated) == 3
        assert abs(sum(updated.values()) - 1.0) < 0.01


class TestAllExpertsIntegration:
    def test_all_experts_count(self):
        from footy.models.experts import ALL_EXPERTS
        # Should have at least 27 experts (22 original + 5 new)
        assert len(ALL_EXPERTS) >= 27

    def test_all_experts_have_name(self):
        from footy.models.experts import ALL_EXPERTS
        for expert in ALL_EXPERTS:
            assert hasattr(expert, 'name'), f"{expert.__class__.__name__} missing name"
            assert expert.name, f"{expert.__class__.__name__} has empty name"
