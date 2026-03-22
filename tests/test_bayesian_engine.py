"""Tests for the Bayesian State-Space Engine."""
from __future__ import annotations

import numpy as np
import pytest

from footy.models.bayesian_engine import (
    BayesianStateSpaceEngine,
    BayesianPrediction,
    ConformalPredictor,
    TeamState,
    unified_bayesian_prediction,
)


class TestTeamState:
    def test_default_state(self):
        ts = TeamState()
        assert ts.attack_mean == 0.0
        assert ts.defense_mean == 0.0
        assert ts.matches_played == 0

    def test_credible_interval(self):
        ts = TeamState(attack_mean=0.5, attack_var=0.25)
        lo, hi = ts.attack_ci
        assert lo < 0.5 < hi
        assert hi - lo > 0.5  # Wide interval with high variance


class TestBayesianEngine:
    def test_initial_prediction(self):
        engine = BayesianStateSpaceEngine()
        pred = engine.predict("Team A", "Team B")
        assert isinstance(pred, BayesianPrediction)
        assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 1e-6

    def test_update_changes_state(self):
        engine = BayesianStateSpaceEngine()
        engine.update("Home", "Away", 3, 0)
        state_h = engine.teams["Home"]
        state_a = engine.teams["Away"]
        assert state_h.matches_played == 1
        assert state_a.matches_played == 1
        # Home scored 3, should have higher attack than initial
        assert state_h.attack_mean > engine.league_attack_mean - 0.1

    def test_prediction_after_updates(self):
        engine = BayesianStateSpaceEngine()
        # Strong home team
        for _ in range(5):
            engine.update("Strong", "Weak", 3, 0)
        pred = engine.predict("Strong", "Weak")
        assert pred.p_home > pred.p_away
        assert pred.confidence > 0.3

    def test_batch_update(self):
        engine = BayesianStateSpaceEngine()
        matches = [
            {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1},
            {"home_team": "B", "away_team": "A", "home_goals": 0, "away_goals": 1},
            {"home_team": "A", "away_team": "C", "home_goals": 3, "away_goals": 0},
        ]
        engine.batch_update(matches)
        assert engine._match_count == 3
        assert "A" in engine.teams
        assert "B" in engine.teams
        assert "C" in engine.teams

    def test_score_matrix_sums_to_one(self):
        engine = BayesianStateSpaceEngine()
        pred = engine.predict("X", "Y")
        assert abs(pred.score_matrix.sum() - 1.0) < 1e-6

    def test_most_likely_outcome(self):
        engine = BayesianStateSpaceEngine()
        pred = engine.predict("A", "B")
        assert pred.most_likely_outcome in ("Home", "Draw", "Away")

    def test_most_likely_score(self):
        engine = BayesianStateSpaceEngine()
        pred = engine.predict("A", "B")
        scores = pred.most_likely_score(top_n=3)
        assert len(scores) <= 3
        for h, a, p in scores:
            assert isinstance(h, int)
            assert isinstance(a, int)
            assert p >= 0

    def test_rankings(self):
        engine = BayesianStateSpaceEngine()
        engine.batch_update([
            {"home_team": "X", "away_team": "Y", "home_goals": 5, "away_goals": 0},
            {"home_team": "Y", "away_team": "Z", "home_goals": 1, "away_goals": 1},
        ])
        rankings = engine.get_rankings(top_n=5)
        assert len(rankings) > 0
        assert rankings[0]["team"] == "X"  # Strongest team

    def test_adaptive_process_variance(self):
        engine = BayesianStateSpaceEngine(learning_rate_adapt=True)
        base_var = engine.process_variance
        # Add many surprising results
        engine._surprise_history = [3.0] * 20
        adapted = engine._adaptive_process_variance()
        assert adapted > base_var  # Should increase for surprising results

    def test_zero_inflation(self):
        engine = BayesianStateSpaceEngine(zero_inflation_base=0.05)
        pred = engine.predict("A", "B")
        # Score matrix (0,0) should be slightly inflated
        assert pred.score_matrix[0, 0] > 0

    def test_probabilities_are_valid(self):
        engine = BayesianStateSpaceEngine()
        for _ in range(10):
            engine.update("A", "B", 1, 1)
        pred = engine.predict("A", "B")
        assert 0 <= pred.p_home <= 1
        assert 0 <= pred.p_draw <= 1
        assert 0 <= pred.p_away <= 1
        assert 0 <= pred.p_btts <= 1
        assert 0 <= pred.p_over_25 <= 1
        assert 0 <= pred.confidence <= 1


class TestConformalPredictor:
    def test_calibrate_and_predict(self):
        cp = ConformalPredictor(coverage=0.9)
        # Simulate calibration data
        probs = [np.array([0.6, 0.2, 0.2])] * 50
        outcomes = [0] * 30 + [1] * 10 + [2] * 10
        cp.calibrate(probs, outcomes)
        assert len(cp.nonconformity_scores) == 50

    def test_prediction_set(self):
        cp = ConformalPredictor(coverage=0.9)
        probs = [np.array([0.7, 0.2, 0.1])] * 100
        outcomes = [0] * 70 + [1] * 20 + [2] * 10
        cp.calibrate(probs, outcomes)

        pred_set, adj_probs, set_size = cp.predict_set(np.array([0.7, 0.2, 0.1]))
        assert 0 in pred_set  # Most likely should be included
        assert len(pred_set) >= 1
        assert set_size >= 1

    def test_empty_calibration(self):
        cp = ConformalPredictor()
        pred_set, adj_probs, set_size = cp.predict_set(np.array([0.5, 0.3, 0.2]))
        assert pred_set == [0, 1, 2]  # All outcomes when no calibration


class TestUnifiedBayesianPrediction:
    def test_basic_prediction(self):
        result = unified_bayesian_prediction(1.5, 1.0)
        assert "p_home" in result
        assert "p_draw" in result
        assert "p_away" in result
        assert abs(result["p_home"] + result["p_draw"] + result["p_away"] - 1.0) < 0.01

    def test_with_market_probs(self):
        result = unified_bayesian_prediction(
            1.5, 1.0,
            market_probs=(0.5, 0.25, 0.25),
        )
        assert result["p_home"] > 0
        assert result["n_models"] >= 2

    def test_model_agreement(self):
        result = unified_bayesian_prediction(1.3, 1.1)
        assert 0 <= result["model_agreement"] <= 1
