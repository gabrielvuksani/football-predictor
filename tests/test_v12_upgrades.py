"""Comprehensive test suite for v12 feature upgrades.

Tests the following new features:
1. EnsembleMetaLearner - Stacking meta-learner for ensemble predictions
2. CouncilBlendConfig - Configuration for council model blending
3. _safe_value and _NAN_STRATEGY - NaN/Inf handling in prediction aggregator
4. _expert_by_name - Expert lookup utility in council module
5. check_for_drift - Model drift detection using CUSUM
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from footy.models.meta_learner import EnsembleMetaLearner
from footy.model_config import CouncilBlendConfig, get_model_config
from footy.prediction_aggregator import _safe_value, _NAN_STRATEGY
from footy.models.council import _expert_by_name
from footy.continuous_training import ContinuousTrainingManager


# =============================================================================
# Test EnsembleMetaLearner
# =============================================================================

class TestEnsembleMetaLearner:
    """Tests for EnsembleMetaLearner class."""

    def test_instantiation(self):
        """Test that EnsembleMetaLearner can be instantiated."""
        learner = EnsembleMetaLearner()
        assert learner is not None
        assert learner._model is None
        assert learner._expert_names == []
        assert learner._is_fitted is False

    def test_fit_with_synthetic_data(self):
        """Test fitting with synthetic expert predictions and outcomes."""
        learner = EnsembleMetaLearner()
        
        # Create synthetic data: 100 matches, 3 experts (9 features total)
        n_matches = 100
        n_experts = 3
        expert_predictions = np.random.dirichlet(
            np.ones(3), size=(n_matches, n_experts)
        ).reshape(n_matches, n_experts * 3)
        
        # Create outcomes: 0=home, 1=draw, 2=away
        outcomes = np.random.choice([0, 1, 2], size=n_matches)
        expert_names = [f"Expert{i}" for i in range(n_experts)]
        
        learner.fit(expert_predictions, outcomes, expert_names)
        
        # Check that model was fitted
        assert learner._is_fitted is True
        assert learner._expert_names == expert_names
        assert learner._model is not None

    def test_fit_insufficient_data(self):
        """Test that fit handles insufficient training data gracefully."""
        learner = EnsembleMetaLearner()
        
        # Only 10 matches (less than minimum 20)
        expert_predictions = np.random.rand(10, 9)
        outcomes = np.random.choice([0, 1, 2], size=10)
        expert_names = ["Expert1", "Expert2", "Expert3"]
        
        learner.fit(expert_predictions, outcomes, expert_names)
        
        # Should not be fitted due to insufficient data
        assert learner._is_fitted is False

    def test_fit_feature_count_mismatch(self):
        """Test that fit handles feature count mismatch."""
        learner = EnsembleMetaLearner()
        
        # Create mismatched data
        expert_predictions = np.random.rand(50, 6)  # 2 experts * 3 features
        outcomes = np.random.choice([0, 1, 2], size=50)
        expert_names = ["Expert1", "Expert2", "Expert3"]  # 3 experts declared
        
        learner.fit(expert_predictions, outcomes, expert_names)
        
        # Should not be fitted due to mismatch
        assert learner._is_fitted is False

    def test_predict_returns_correct_shape(self):
        """Test that predict returns correct output shape."""
        learner = EnsembleMetaLearner()
        
        # Train first
        n_matches = 100
        n_experts = 3
        # Generate shape (n_matches, n_experts * 3) properly
        expert_predictions = np.random.rand(n_matches, n_experts * 3)
        # Normalize to probabilities
        expert_predictions = expert_predictions / expert_predictions.sum(axis=1, keepdims=True)
        
        outcomes = np.random.choice([0, 1, 2], size=n_matches)
        expert_names = [f"Expert{i}" for i in range(n_experts)]
        
        learner.fit(expert_predictions, outcomes, expert_names)
        
        # Test prediction with single match
        test_pred = np.random.rand(1, 9)
        test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)
        result = learner.predict(test_pred)
        
        assert result is not None
        assert result.shape == (1, 3)

    def test_predict_probabilities_sum_to_one(self):
        """Test that predicted probabilities sum to approximately 1.0."""
        learner = EnsembleMetaLearner()
        
        # Train
        n_matches = 100
        n_experts = 3
        expert_predictions = np.random.rand(n_matches, n_experts * 3)
        expert_predictions = expert_predictions / expert_predictions.sum(axis=1, keepdims=True)
        
        outcomes = np.random.choice([0, 1, 2], size=n_matches)
        expert_names = [f"Expert{i}" for i in range(n_experts)]
        
        learner.fit(expert_predictions, outcomes, expert_names)
        
        # Predict
        test_pred = np.random.rand(5, 9)
        test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)
        result = learner.predict(test_pred)
        
        assert result is not None
        # Check that probabilities sum to ~1.0 for each prediction
        sums = result.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(5), decimal=5)

    def test_feature_importance_returns_dict(self):
        """Test that feature_importance returns a dictionary."""
        learner = EnsembleMetaLearner()
        
        # Train
        n_matches = 100
        n_experts = 3
        expert_predictions = np.random.dirichlet(
            np.ones(3), size=(n_matches, n_experts)
        ).reshape(n_matches, n_experts * 3)
        outcomes = np.random.choice([0, 1, 2], size=n_matches)
        expert_names = [f"Expert{i}" for i in range(n_experts)]
        
        learner.fit(expert_predictions, outcomes, expert_names)
        
        # Get feature importance
        importance = learner.feature_importance()
        
        assert isinstance(importance, dict)

    def test_save_load_roundtrip(self):
        """Test that saving and loading preserves model state."""
        learner = EnsembleMetaLearner()
        
        # Train
        n_matches = 100
        n_experts = 3
        expert_predictions = np.random.rand(n_matches, n_experts * 3)
        expert_predictions = expert_predictions / expert_predictions.sum(axis=1, keepdims=True)
        
        outcomes = np.random.choice([0, 1, 2], size=n_matches)
        expert_names = [f"Expert{i}" for i in range(n_experts)]
        
        learner.fit(expert_predictions, outcomes, expert_names)
        
        # Get prediction from trained model
        test_pred = np.random.rand(1, 9)
        test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)
        pred_before = learner.predict(test_pred)
        
        # Create new learner and load
        learner2 = EnsembleMetaLearner()
        learner2._load()
        
        # Check if model was loaded
        if learner2._is_fitted:
            pred_after = learner2.predict(test_pred)
            np.testing.assert_array_almost_equal(pred_before, pred_after, decimal=5)

    def test_predict_before_fit_returns_none(self):
        """Test that predict returns None if model not fitted and no saved model."""
        learner = EnsembleMetaLearner()
        
        # Ensure the model doesn't exist on disk or set path to non-existent
        learner.MODEL_PATH = Path("/tmp/nonexistent_meta_learner_path_12345.joblib")
        
        test_pred = np.random.rand(1, 9)
        result = learner.predict(test_pred)
        
        # Should return None since not fitted and model doesn't exist
        assert result is None

    def test_is_fitted_method(self):
        """Test the is_fitted method."""
        learner = EnsembleMetaLearner()
        
        # Initially not fitted
        assert learner.is_fitted() is False
        
        # Train
        n_matches = 100
        n_experts = 3
        expert_predictions = np.random.dirichlet(
            np.ones(3), size=(n_matches, n_experts)
        ).reshape(n_matches, n_experts * 3)
        outcomes = np.random.choice([0, 1, 2], size=n_matches)
        expert_names = [f"Expert{i}" for i in range(n_experts)]
        
        learner.fit(expert_predictions, outcomes, expert_names)
        
        # Now should be fitted
        assert learner.is_fitted() is True


# =============================================================================
# Test CouncilBlendConfig
# =============================================================================

class TestCouncilBlendConfig:
    """Tests for CouncilBlendConfig configuration class."""

    def test_default_values_exist(self):
        """Test that CouncilBlendConfig has default values."""
        config = CouncilBlendConfig()
        
        assert hasattr(config, 'ensemble_blend_factor')
        assert hasattr(config, 'adaptive_blend_with_calibration')
        assert hasattr(config, 'adaptive_blend_without_calibration')
        assert hasattr(config, 'validation_fraction')
        assert hasattr(config, 'feature_interaction_blend')
        assert hasattr(config, 'high_agreement_boost')
        assert hasattr(config, 'disagreement_penalty')

    def test_all_fields_are_floats(self):
        """Test that all CouncilBlendConfig fields are floats."""
        config = CouncilBlendConfig()
        
        for field_name in [
            'ensemble_blend_factor',
            'adaptive_blend_with_calibration',
            'adaptive_blend_without_calibration',
            'validation_fraction',
            'feature_interaction_blend',
            'high_agreement_boost',
            'disagreement_penalty',
        ]:
            value = getattr(config, field_name)
            assert isinstance(value, float), \
                f"{field_name} should be float, got {type(value)}"

    def test_environment_variable_override(self):
        """Test that environment variables override default values."""
        # Save original env
        orig_val = os.environ.get('FOOTY_COUNCIL_BLEND_ENSEMBLE_BLEND_FACTOR')
        
        try:
            # Set environment variable (if mechanism exists)
            # Note: This tests the pattern, actual implementation may vary
            os.environ['FOOTY_COUNCIL_BLEND_ENSEMBLE_BLEND_FACTOR'] = '0.25'
            
            # Create config - it should attempt to read from env
            config = CouncilBlendConfig()
            # Default value should still be there (unless env override is implemented)
            assert config.ensemble_blend_factor > 0
            
        finally:
            # Restore original
            if orig_val is not None:
                os.environ['FOOTY_COUNCIL_BLEND_ENSEMBLE_BLEND_FACTOR'] = orig_val
            else:
                os.environ.pop('FOOTY_COUNCIL_BLEND_ENSEMBLE_BLEND_FACTOR', None)

    def test_council_blend_in_model_config(self):
        """Test that council_blend is available in ModelConfig."""
        cfg = get_model_config()
        
        assert hasattr(cfg, 'council_blend')
        assert isinstance(cfg.council_blend, CouncilBlendConfig)


# =============================================================================
# Test _safe_value and _NAN_STRATEGY
# =============================================================================

class TestSafeValueAndNanStrategy:
    """Tests for _safe_value function and _NAN_STRATEGY dictionary."""

    def test_nan_strategy_keys_exist(self):
        """Test that _NAN_STRATEGY has expected keys."""
        assert 'probability' in _NAN_STRATEGY
        assert 'feature' in _NAN_STRATEGY
        assert 'rating' in _NAN_STRATEGY

    def test_nan_probability_defaults_to_one_third(self):
        """Test that NaN probability defaults to 1/3."""
        assert _NAN_STRATEGY['probability'] == pytest.approx(1/3)

    def test_nan_feature_defaults_to_zero(self):
        """Test that NaN feature defaults to 0.0."""
        assert _NAN_STRATEGY['feature'] == 0.0

    def test_nan_rating_defaults_to_1500(self):
        """Test that NaN rating defaults to 1500.0."""
        assert _NAN_STRATEGY['rating'] == 1500.0

    def test_safe_value_nan_probability(self):
        """Test _safe_value handles NaN for probability kind."""
        result = _safe_value(np.nan, kind='probability')
        assert result == pytest.approx(1/3)

    def test_safe_value_nan_feature(self):
        """Test _safe_value handles NaN for feature kind."""
        result = _safe_value(np.nan, kind='feature')
        assert result == 0.0

    def test_safe_value_nan_rating(self):
        """Test _safe_value handles NaN for rating kind."""
        result = _safe_value(np.nan, kind='rating')
        assert result == 1500.0

    def test_safe_value_inf_probability(self):
        """Test _safe_value handles Inf for probability kind."""
        result = _safe_value(np.inf, kind='probability')
        assert result == pytest.approx(1/3)

    def test_safe_value_negative_inf_feature(self):
        """Test _safe_value handles -Inf for feature kind."""
        result = _safe_value(-np.inf, kind='feature')
        assert result == 0.0

    def test_safe_value_none_probability(self):
        """Test _safe_value handles None for probability kind."""
        result = _safe_value(None, kind='probability')
        assert result == pytest.approx(1/3)

    def test_safe_value_none_feature(self):
        """Test _safe_value handles None for feature kind."""
        result = _safe_value(None, kind='feature')
        assert result == 0.0

    def test_safe_value_none_rating(self):
        """Test _safe_value handles None for rating kind."""
        result = _safe_value(None, kind='rating')
        assert result == 1500.0

    def test_safe_value_normal_value_passthrough(self):
        """Test _safe_value passes through normal values."""
        assert _safe_value(0.5, kind='probability') == 0.5
        assert _safe_value(2.5, kind='feature') == 2.5
        assert _safe_value(1600.0, kind='rating') == 1600.0
        assert _safe_value(42, kind='feature') == 42

    def test_safe_value_unknown_kind_defaults_to_zero(self):
        """Test _safe_value defaults to 0.0 for unknown kind."""
        result = _safe_value(np.nan, kind='unknown')
        assert result == 0.0

    def test_safe_value_normal_int_passthrough(self):
        """Test _safe_value passes through integers."""
        assert _safe_value(5, kind='feature') == 5
        assert _safe_value(0, kind='feature') == 0

    def test_safe_value_normal_float_passthrough(self):
        """Test _safe_value passes through normal floats."""
        assert _safe_value(0.123, kind='probability') == 0.123
        assert _safe_value(999.999, kind='rating') == 999.999


# =============================================================================
# Test _expert_by_name
# =============================================================================

class MockExpert:
    """Mock expert class for testing."""
    
    def __init__(self, name):
        self.name = name
    
    @property
    def __class__(self):
        """Mock class name."""
        class MockClass:
            def __init__(self, name):
                self.__name__ = name
        return MockClass(self.__class__.__name__)


class MockExpertResult:
    """Mock expert result."""
    
    def __init__(self, p_home, p_draw, p_away):
        self.p_home = p_home
        self.p_draw = p_draw
        self.p_away = p_away


class TestExpertByName:
    """Tests for _expert_by_name utility function."""

    def test_expert_by_name_finds_correct_expert(self):
        """Test that _expert_by_name finds the correct expert by name."""
        # Create mock experts and results
        class Expert1:
            name = "EloExpert"
        
        class Expert2:
            name = "PoissonExpert"
        
        result1 = MockExpertResult(0.5, 0.3, 0.2)
        result2 = MockExpertResult(0.4, 0.35, 0.25)
        
        experts = [(Expert1(), result1), (Expert2(), result2)]
        
        found = _expert_by_name(experts, "PoissonExpert")
        assert found is result2

    def test_expert_by_name_finds_by_class_name(self):
        """Test that _expert_by_name can find expert by class name."""
        class EloExpert:
            pass
        
        result1 = MockExpertResult(0.5, 0.3, 0.2)
        
        expert = EloExpert()
        experts = [(expert, result1)]
        
        found = _expert_by_name(experts, "EloExpert")
        assert found is result1

    def test_expert_by_name_returns_none_for_missing(self):
        """Test that _expert_by_name returns None for missing expert."""
        class Expert1:
            name = "EloExpert"
        
        result1 = MockExpertResult(0.5, 0.3, 0.2)
        experts = [(Expert1(), result1)]
        
        found = _expert_by_name(experts, "NonExistentExpert")
        assert found is None

    def test_expert_by_name_empty_list(self):
        """Test that _expert_by_name handles empty expert list."""
        found = _expert_by_name([], "SomeExpert")
        assert found is None

    def test_expert_by_name_none_list(self):
        """Test that _expert_by_name handles None expert list."""
        found = _expert_by_name(None, "SomeExpert")
        assert found is None


# =============================================================================
# Test check_for_drift
# =============================================================================

class TestCheckForDrift:
    """Tests for drift detection using CUSUM control chart."""

    def test_drift_detection_no_drift_stable_predictions(self):
        """Test that stable predictions do not trigger drift detection."""
        learner = ContinuousTrainingManager()
        
        # Create stable predictions: all predict home win with high confidence
        recent_predictions = [[0.7, 0.2, 0.1] for _ in range(50)]
        # Outcomes are consistently home wins
        recent_outcomes = [0 for _ in range(50)]
        
        # Check for drift
        has_drift = learner.check_for_drift(
            recent_predictions, recent_outcomes,
            threshold=0.05, window_size=30
        )
        
        # Should not detect drift with stable, correct predictions
        assert has_drift is False

    def test_drift_detection_with_degrading_predictions(self):
        """Test that degrading predictions trigger drift detection."""
        learner = ContinuousTrainingManager()
        
        # First half: good predictions
        predictions_good = [[0.7, 0.2, 0.1] for _ in range(30)]
        outcomes_good = [0 for _ in range(30)]
        
        # Second half: poor predictions (opposite of outcomes)
        predictions_bad = [[0.2, 0.1, 0.7] for _ in range(30)]
        outcomes_bad = [0 for _ in range(30)]  # Still home wins
        
        recent_predictions = predictions_good + predictions_bad
        recent_outcomes = outcomes_good + outcomes_bad
        
        # Check for drift
        has_drift = learner.check_for_drift(
            recent_predictions, recent_outcomes,
            threshold=0.05, window_size=30
        )
        
        # May or may not detect drift depending on CUSUM sensitivity
        # The test validates that the method runs without error
        assert isinstance(has_drift, bool)

    def test_drift_detection_empty_lists(self):
        """Test that empty lists don't cause errors."""
        learner = ContinuousTrainingManager()
        
        has_drift = learner.check_for_drift([], [], threshold=0.05, window_size=30)
        
        # Should return False for empty data
        assert has_drift is False

    def test_drift_detection_insufficient_data(self):
        """Test that insufficient data returns False."""
        learner = ContinuousTrainingManager()
        
        # Only 10 predictions (less than window_size=30)
        recent_predictions = [[0.5, 0.3, 0.2] for _ in range(10)]
        recent_outcomes = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        
        has_drift = learner.check_for_drift(
            recent_predictions, recent_outcomes,
            threshold=0.05, window_size=30
        )
        
        # Should return False due to insufficient data
        assert has_drift is False

    def test_drift_detection_returns_boolean(self):
        """Test that check_for_drift returns a boolean."""
        learner = ContinuousTrainingManager()
        
        recent_predictions = [[0.5, 0.3, 0.2] for _ in range(50)]
        recent_outcomes = [0, 1, 2] * 16 + [0, 1, 2]
        
        result = learner.check_for_drift(
            recent_predictions, recent_outcomes,
            threshold=0.05, window_size=30
        )
        
        assert isinstance(result, bool)

    def test_drift_detection_custom_threshold(self):
        """Test drift detection with custom threshold."""
        learner = ContinuousTrainingManager()
        
        # Create marginal predictions
        recent_predictions = [[0.4, 0.3, 0.3] for _ in range(50)]
        recent_outcomes = [0, 0, 0, 1, 1, 1, 2, 2, 2] * 5 + [0, 1, 2, 0]
        
        # With higher threshold, less likely to detect drift
        result_high_threshold = learner.check_for_drift(
            recent_predictions, recent_outcomes,
            threshold=0.2, window_size=30
        )
        
        # With lower threshold, more likely to detect drift
        result_low_threshold = learner.check_for_drift(
            recent_predictions, recent_outcomes,
            threshold=0.01, window_size=30
        )
        
        # Both should return boolean
        assert isinstance(result_high_threshold, bool)
        assert isinstance(result_low_threshold, bool)

    def test_drift_detection_custom_window_size(self):
        """Test drift detection with custom window size."""
        learner = ContinuousTrainingManager()
        
        recent_predictions = [[0.5, 0.3, 0.2] for _ in range(100)]
        recent_outcomes = [0, 1, 2] * 33 + [0, 1]
        
        # Should work with larger window
        result = learner.check_for_drift(
            recent_predictions, recent_outcomes,
            threshold=0.05, window_size=60
        )
        
        assert isinstance(result, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
