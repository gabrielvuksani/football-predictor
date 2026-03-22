"""Tests for enhanced prediction aggregator functionality."""
from __future__ import annotations

import numpy as np
import pytest

from footy.prediction_aggregator import (
    UnifiedPrediction,
    aggregate_predictions,
)


class TestUnifiedPredictionObject:
    """Tests for UnifiedPrediction dataclass."""

    def test_default_initialization(self):
        """Test UnifiedPrediction default values."""
        pred = UnifiedPrediction()

        assert pred.p_home == pytest.approx(1/3, abs=0.01)
        assert pred.p_draw == pytest.approx(1/3, abs=0.01)
        assert pred.p_away == pytest.approx(1/3, abs=0.01)
        assert pred.confidence == 0.5
        assert pred.model_agreement == 0.5

    def test_custom_initialization(self):
        """Test UnifiedPrediction with custom values."""
        pred = UnifiedPrediction(
            p_home=0.6,
            p_draw=0.2,
            p_away=0.2,
            confidence=0.8,
            model_agreement=0.75,
        )

        assert pred.p_home == 0.6
        assert pred.p_draw == 0.2
        assert pred.p_away == 0.2
        assert pred.confidence == 0.8
        assert pred.model_agreement == 0.75

    def test_most_likely_outcome(self):
        """Test most_likely_outcome property."""
        pred = UnifiedPrediction(p_home=0.6, p_draw=0.2, p_away=0.2)
        assert pred.most_likely_outcome == "Home"

        pred = UnifiedPrediction(p_home=0.2, p_draw=0.5, p_away=0.3)
        assert pred.most_likely_outcome == "Draw"

        pred = UnifiedPrediction(p_home=0.1, p_draw=0.2, p_away=0.7)
        assert pred.most_likely_outcome == "Away"

    def test_outcome_label(self):
        """Test outcome_label property."""
        pred = UnifiedPrediction(p_home=0.7, p_draw=0.2, p_away=0.1)
        assert pred.outcome_label == "1"

        pred = UnifiedPrediction(p_home=0.2, p_draw=0.7, p_away=0.1)
        assert pred.outcome_label == "X"

        pred = UnifiedPrediction(p_home=0.1, p_draw=0.2, p_away=0.7)
        assert pred.outcome_label == "2"

    def test_to_dict_structure(self):
        """Test to_dict method structure."""
        pred = UnifiedPrediction(
            p_home=0.6,
            p_draw=0.2,
            p_away=0.2,
            eg_home=1.5,
            eg_away=1.0,
        )

        d = pred.to_dict()

        assert isinstance(d, dict)
        assert "p_home" in d
        assert "p_draw" in d
        assert "p_away" in d
        assert "eg_home" in d
        assert "eg_away" in d
        assert "outcome" in d
        assert "component_breakdown" in d
        assert "value_edges" in d

    def test_to_dict_values_rounded(self):
        """Test that to_dict rounds values appropriately."""
        pred = UnifiedPrediction(p_home=0.123456, p_draw=0.234567)
        d = pred.to_dict()

        # Should be rounded to 4 decimals
        assert d["p_home"] == 0.1235
        assert d["p_draw"] == 0.2346

    def test_to_dict_serializable(self):
        """Test that to_dict output is JSON-serializable."""
        import json
        pred = UnifiedPrediction(
            p_home=0.6,
            p_draw=0.2,
            p_away=0.2,
            score_probabilities=[(1, 0, 0.15), (2, 0, 0.1)],
        )

        d = pred.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


class TestAggregatePredictionsBasic:
    """Tests for basic aggregation functionality."""

    def test_aggregate_council_only(self):
        """Test aggregation with council predictions only."""
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
        )

        assert isinstance(result, UnifiedPrediction)
        assert abs(result.p_home - 0.6) < 0.05
        assert abs(result.p_draw - 0.2) < 0.05
        assert abs(result.p_away - 0.2) < 0.05

    def test_aggregate_no_sources(self):
        """Test aggregation with no input sources."""
        result = aggregate_predictions()

        # Should default to uniform
        assert abs(result.p_home - 1/3) < 0.1
        assert abs(result.p_draw - 1/3) < 0.1
        assert abs(result.p_away - 1/3) < 0.1

    def test_aggregate_council_and_bayesian(self):
        """Test aggregation with council and Bayesian."""
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
            bayesian_probs=(0.55, 0.25, 0.2),
        )

        assert isinstance(result, UnifiedPrediction)
        assert 0.5 < result.p_home < 0.65
        assert result.n_models_used == 2

    def test_aggregate_probabilities_sum_to_one(self):
        """Test that aggregated probabilities sum to 1."""
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
            bayesian_probs=(0.55, 0.25, 0.2),
            statistical_probs=(0.5, 0.3, 0.2),
        )

        total = result.p_home + result.p_draw + result.p_away
        assert abs(total - 1.0) < 1e-6

    def test_aggregate_all_sources(self):
        """Test aggregation with all available sources."""
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
            bayesian_probs=(0.55, 0.25, 0.2),
            statistical_probs=(0.5, 0.3, 0.2),
            market_probs=(0.45, 0.35, 0.2),
        )

        assert result.n_models_used >= 3
        assert abs(result.p_home + result.p_draw + result.p_away - 1.0) < 1e-6


class TestAggregateModelAgreement:
    """Tests for model agreement metrics."""

    def test_perfect_agreement(self):
        """Test model agreement when all models agree."""
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
            bayesian_probs=(0.6, 0.2, 0.2),
            statistical_probs=(0.6, 0.2, 0.2),
        )

        assert result.model_agreement > 0.7  # Should be high when models agree

    def test_disagreement_detected(self):
        """Test model agreement when models disagree."""
        result = aggregate_predictions(
            council_probs=(0.7, 0.2, 0.1),
            bayesian_probs=(0.3, 0.3, 0.4),
        )

        # With disagreement, agreement should be measurably less than perfect
        assert result.model_agreement <= 1.0
        assert result.model_agreement < 0.95  # Not perfect agreement

    def test_disagreement_risk_characteristics(self):
        """Test that disagreement changes risk metrics."""
        agree = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
            bayesian_probs=(0.58, 0.22, 0.2),
        )

        disagree = aggregate_predictions(
            council_probs=(0.7, 0.2, 0.1),
            bayesian_probs=(0.3, 0.3, 0.4),
        )

        # Disagreement should have measurably different metrics
        assert agree.model_agreement != disagree.model_agreement


class TestValueEdges:
    """Tests for value edge computation."""

    def test_value_edge_positive_home(self):
        """Test positive value edge for home."""
        result = aggregate_predictions(
            council_probs=(0.7, 0.15, 0.15),
            market_probs=(0.5, 0.25, 0.25),
        )

        assert result.value_edge_home > 0

    def test_value_edge_negative_home(self):
        """Test negative value edge for home."""
        result = aggregate_predictions(
            council_probs=(0.4, 0.3, 0.3),
            market_probs=(0.5, 0.25, 0.25),
        )

        assert result.value_edge_home < 0

    def test_value_edge_draw(self):
        """Test value edge for draw."""
        result = aggregate_predictions(
            council_probs=(0.45, 0.35, 0.2),
            market_probs=(0.4, 0.25, 0.35),
        )

        assert result.value_edge_draw > 0

    def test_value_edge_away(self):
        """Test value edge for away."""
        result = aggregate_predictions(
            council_probs=(0.3, 0.2, 0.5),
            market_probs=(0.4, 0.25, 0.35),
        )

        assert result.value_edge_away > 0

    def test_value_edges_sum_to_zero_approximately(self):
        """Test that value edges balance out."""
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
            market_probs=(0.5, 0.25, 0.25),
        )

        edges_sum = result.value_edge_home + result.value_edge_draw + result.value_edge_away
        # Should be close to zero (edges are relative)
        assert abs(edges_sum) < 0.2


class TestUpsetRisk:
    """Tests for upset risk scoring."""

    def test_upset_risk_strong_favourite(self):
        """Test upset risk for strong favourite."""
        result = aggregate_predictions(
            council_probs=(0.85, 0.1, 0.05),
            bayesian_probs=(0.85, 0.1, 0.05),
        )

        # Strong agreement on strong favourite should have moderate risk
        assert 0 <= result.upset_risk <= 1

    def test_upset_risk_wide_disagreement(self):
        """Test upset risk for wide disagreement."""
        result = aggregate_predictions(
            council_probs=(0.7, 0.15, 0.15),
            bayesian_probs=(0.3, 0.3, 0.4),
        )

        # Wide disagreement should produce measurable risk
        assert 0 <= result.upset_risk <= 1

    def test_upset_risk_bounds(self):
        """Test that upset risk is bounded."""
        for _ in range(20):
            probs1 = np.random.dirichlet([2, 1.5, 1.5])
            probs2 = np.random.dirichlet([2, 1.5, 1.5])

            result = aggregate_predictions(
                council_probs=tuple(probs1),
                bayesian_probs=tuple(probs2),
            )

            assert 0 <= result.upset_risk <= 1


class TestConfidenceMetric:
    """Tests for confidence metric."""

    def test_confidence_high_agreement(self):
        """Test confidence increases with agreement."""
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
            bayesian_probs=(0.6, 0.2, 0.2),
            statistical_probs=(0.6, 0.2, 0.2),
        )

        # High agreement should produce higher confidence
        assert result.confidence > 0.5

    def test_confidence_low_uncertainty(self):
        """Test confidence with low uncertainty."""
        result = aggregate_predictions(
            council_probs=(0.8, 0.1, 0.1),
        )

        assert result.confidence > 0.5  # Clear prediction should have confidence

    def test_confidence_high_uncertainty(self):
        """Test confidence with high uncertainty."""
        result = aggregate_predictions(
            council_probs=(0.35, 0.35, 0.3),
        )

        # Uncertain prediction should still produce valid confidence
        assert 0 <= result.confidence <= 1


class TestComponentBreakdown:
    """Tests for component breakdown."""

    def test_component_breakdown_included(self):
        """Test that component breakdown is included in output."""
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
            bayesian_probs=(0.55, 0.25, 0.2),
        )

        d = result.to_dict()
        assert "component_breakdown" in d
        assert "council" in d["component_breakdown"]
        assert "bayesian" in d["component_breakdown"]

    def test_component_breakdown_values(self):
        """Test that component breakdown preserves source values."""
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
        )

        d = result.to_dict()
        breakdown = d["component_breakdown"]
        assert abs(breakdown["council"][0] - 0.6) < 0.01


class TestPredictionSet:
    """Tests for prediction set (conformal prediction)."""

    def test_prediction_set_includes_likely_outcome(self):
        """Test that prediction set includes most likely outcome."""
        result = aggregate_predictions(
            council_probs=(0.7, 0.2, 0.1),
        )

        assert 0 in result.prediction_set  # Home is most likely

    def test_prediction_set_bounds(self):
        """Test that prediction set is valid."""
        result = aggregate_predictions(
            council_probs=(0.5, 0.3, 0.2),
        )

        # Each element should be 0, 1, or 2
        assert all(outcome in [0, 1, 2] for outcome in result.prediction_set)
        assert len(result.prediction_set) >= 1

    def test_prediction_interval(self):
        """Test that prediction interval is valid."""
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
        )

        assert result.prediction_interval[0] >= 0
        assert result.prediction_interval[1] <= 1
        assert result.prediction_interval[0] <= result.prediction_interval[1]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_model_prediction(self):
        """Test aggregation with single model."""
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
        )

        assert result.n_models_used == 1
        assert abs(result.p_home - 0.6) < 0.05

    def test_identical_models_prediction(self):
        """Test aggregation when all models are identical."""
        result = aggregate_predictions(
            council_probs=(0.5, 0.3, 0.2),
            bayesian_probs=(0.5, 0.3, 0.2),
            statistical_probs=(0.5, 0.3, 0.2),
        )

        assert result.model_agreement > 0.8
        assert abs(result.p_home - 0.5) < 0.05

    def test_extremely_disagreeing_models(self):
        """Test aggregation with extremely disagreeing models."""
        result = aggregate_predictions(
            council_probs=(0.9, 0.05, 0.05),
            bayesian_probs=(0.05, 0.05, 0.9),
        )

        # Should be valid aggregation
        assert 0 < result.p_home < 1
        assert 0 <= result.upset_risk <= 1

    def test_uniform_market_prediction(self):
        """Test with uniform market prediction."""
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
            market_probs=(1/3, 1/3, 1/3),
        )

        # Should have 2 models
        assert result.n_models_used >= 2

    def test_extreme_probabilities(self):
        """Test with extreme probability values."""
        result = aggregate_predictions(
            council_probs=(0.99, 0.005, 0.005),
        )

        assert result.p_home > 0.9
        assert result.confidence > 0.5

    def test_very_close_probabilities(self):
        """Test with very close probability values."""
        result = aggregate_predictions(
            council_probs=(0.334, 0.333, 0.333),
        )

        assert 0.3 < result.p_home < 0.4
        assert 0 <= result.confidence <= 1


class TestMultipleAggregations:
    """Tests for multiple predictions in sequence."""

    def test_consistency_across_calls(self):
        """Test that same input gives consistent output."""
        result1 = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
            bayesian_probs=(0.55, 0.25, 0.2),
        )

        result2 = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
            bayesian_probs=(0.55, 0.25, 0.2),
        )

        assert abs(result1.p_home - result2.p_home) < 1e-6

    def test_different_inputs_different_outputs(self):
        """Test that different inputs produce different outputs."""
        result1 = aggregate_predictions(council_probs=(0.6, 0.2, 0.2))
        result2 = aggregate_predictions(council_probs=(0.3, 0.3, 0.4))

        assert result1.p_home > result2.p_home
        assert result1.p_away < result2.p_away

    def test_monotonic_probability_change(self):
        """Test that increasing council home probability increases output."""
        for prob_h in [0.3, 0.5, 0.7, 0.9]:
            result = aggregate_predictions(
                council_probs=(prob_h, (1-prob_h)*0.4, (1-prob_h)*0.6),
            )
            assert result.p_home == pytest.approx(prob_h, abs=0.05)
