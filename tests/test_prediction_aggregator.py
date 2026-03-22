"""Tests for the Unified Prediction Aggregator."""
from __future__ import annotations

import numpy as np
import pytest

from footy.prediction_aggregator import (
    UnifiedPrediction,
    logarithmic_opinion_pool,
    linear_opinion_pool,
    compute_value_edges,
    compute_upset_risk,
    aggregate_predictions,
)


class TestLogarithmicOpinionPool:
    def test_single_source(self):
        result = logarithmic_opinion_pool([(0.6, 0.2, 0.2)], [1.0])
        assert abs(result[0] - 0.6) < 0.01
        assert abs(sum(result) - 1.0) < 1e-6

    def test_equal_weights(self):
        probs = [(0.6, 0.2, 0.2), (0.4, 0.3, 0.3)]
        result = logarithmic_opinion_pool(probs, [1.0, 1.0])
        assert abs(sum(result) - 1.0) < 1e-6
        # Geometric mean should be between the two
        assert 0.4 < result[0] < 0.6

    def test_dominates_with_high_weight(self):
        probs = [(0.8, 0.1, 0.1), (0.2, 0.4, 0.4)]
        result = logarithmic_opinion_pool(probs, [10.0, 1.0])
        assert result[0] > 0.6  # First source dominates

    def test_empty_input(self):
        result = logarithmic_opinion_pool([], [])
        assert abs(sum(result) - 1.0) < 1e-6

    def test_always_sums_to_one(self):
        for _ in range(20):
            probs = [
                (np.random.uniform(0.1, 0.9), np.random.uniform(0.05, 0.4), 0)
                for _ in range(3)
            ]
            probs = [(p[0], p[1], 1 - p[0] - p[1]) for p in probs]
            probs = [(max(0.01, p[0]), max(0.01, p[1]), max(0.01, p[2])) for p in probs]
            weights = [np.random.uniform(0.1, 2.0) for _ in range(3)]
            result = logarithmic_opinion_pool(probs, weights)
            assert abs(sum(result) - 1.0) < 1e-6


class TestLinearOpinionPool:
    def test_basic(self):
        result = linear_opinion_pool(
            [(0.6, 0.2, 0.2), (0.4, 0.3, 0.3)], [1.0, 1.0]
        )
        assert abs(result[0] - 0.5) < 0.01
        assert abs(sum(result) - 1.0) < 1e-6


class TestComputeValueEdges:
    def test_positive_edge(self):
        edges = compute_value_edges((0.6, 0.2, 0.2), (0.5, 0.25, 0.25))
        assert edges[0] > 0  # Model thinks home more likely
        assert edges[1] < 0

    def test_no_market_data(self):
        edges = compute_value_edges((0.5, 0.3, 0.2), (0.0, 0.0, 0.0))
        assert edges == (0.0, 0.0, 0.0)


class TestComputeUpsetRisk:
    def test_strong_favourite_low_risk(self):
        risk = compute_upset_risk((0.8, 0.1, 0.1), (0.8, 0.1, 0.1), 0.9)
        assert risk < 0.3

    def test_disagreement_high_risk(self):
        risk = compute_upset_risk((0.5, 0.3, 0.2), (0.7, 0.15, 0.15), 0.4)
        assert risk > 0.3


class TestUnifiedPrediction:
    def test_to_dict(self):
        pred = UnifiedPrediction(p_home=0.5, p_draw=0.3, p_away=0.2)
        d = pred.to_dict()
        assert d["p_home"] == 0.5
        assert d["outcome"] == "1"
        assert "component_breakdown" in d

    def test_most_likely_outcome(self):
        pred = UnifiedPrediction(p_home=0.2, p_draw=0.5, p_away=0.3)
        assert pred.most_likely_outcome == "Draw"

    def test_outcome_label(self):
        pred = UnifiedPrediction(p_home=0.1, p_draw=0.1, p_away=0.8)
        assert pred.outcome_label == "2"


class TestAggregatePredictions:
    def test_council_only(self):
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
        )
        assert isinstance(result, UnifiedPrediction)
        assert abs(result.p_home + result.p_draw + result.p_away - 1.0) < 1e-6

    def test_multiple_sources(self):
        result = aggregate_predictions(
            council_probs=(0.6, 0.2, 0.2),
            bayesian_probs=(0.55, 0.25, 0.2),
            market_probs=(0.5, 0.25, 0.25),
        )
        assert result.n_models_used == 3
        assert result.model_agreement > 0

    def test_no_sources(self):
        result = aggregate_predictions()
        assert result.p_home == pytest.approx(1 / 3, abs=0.01)

    def test_value_edges_computed(self):
        result = aggregate_predictions(
            council_probs=(0.7, 0.15, 0.15),
            market_probs=(0.5, 0.25, 0.25),
        )
        assert result.value_edge_home > 0

    def test_upset_risk(self):
        result = aggregate_predictions(
            council_probs=(0.5, 0.3, 0.2),
            market_probs=(0.7, 0.15, 0.15),
        )
        assert result.upset_risk > 0
