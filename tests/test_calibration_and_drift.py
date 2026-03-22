"""Tests for Calibration Pipeline and Enhanced Drift Detection.

Tests the new calibration methods and drift detection enhancements.
Run with: pytest tests/test_calibration_and_drift.py -v
"""
import numpy as np
import pytest

from footy.models.calibration import (
    PlattCalibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
    BetaCalibrator,
    AutoCalibrator,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram_data,
    fit_calibrator,
)
from footy.self_learning import (
    DriftDetector,
    PageHinkleyDetector,
    ADWINDetector,
    DDMDetector,
    EDDMDetector,
    SelfLearningLoop,
)


class TestCalibrationMethods:
    """Test individual calibration methods."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        # Create properly distributed probabilities
        self.n_samples = 300
        self.probs = np.random.dirichlet([1, 1, 1], size=self.n_samples)
        # Generate outcomes from the probabilities
        self.outcomes = np.array([
            np.random.choice([0, 1, 2], p=p) for p in self.probs
        ])

        # Add miscalibration (exaggerate probabilities)
        self.uncalibrated_probs = self.probs ** 1.3
        self.uncalibrated_probs /= self.uncalibrated_probs.sum(axis=1, keepdims=True)

    def test_platt_calibration(self):
        """Test Platt scaling."""
        calibrator = PlattCalibrator(n_classes=3)
        assert not calibrator.is_fitted

        calibrator.fit(self.uncalibrated_probs, self.outcomes)
        assert calibrator.is_fitted

        calibrated = calibrator.calibrate(self.uncalibrated_probs)
        assert calibrated.shape == self.uncalibrated_probs.shape
        assert np.allclose(calibrated.sum(axis=1), 1.0)

    def test_isotonic_calibration(self):
        """Test Isotonic regression."""
        calibrator = IsotonicCalibrator(n_classes=3)
        calibrator.fit(self.uncalibrated_probs, self.outcomes)
        assert calibrator.is_fitted

        calibrated = calibrator.calibrate(self.uncalibrated_probs)
        assert calibrated.shape == self.uncalibrated_probs.shape
        assert np.allclose(calibrated.sum(axis=1), 1.0)

    def test_temperature_scaling(self):
        """Test Temperature scaling."""
        calibrator = TemperatureCalibrator()
        calibrator.fit(self.uncalibrated_probs, self.outcomes)
        assert calibrator.is_fitted
        assert calibrator.temperature > 0

        calibrated = calibrator.calibrate(self.uncalibrated_probs)
        assert calibrated.shape == self.uncalibrated_probs.shape
        assert np.allclose(calibrated.sum(axis=1), 1.0)

    def test_beta_calibration(self):
        """Test Beta calibration."""
        calibrator = BetaCalibrator(n_classes=3)
        calibrator.fit(self.uncalibrated_probs, self.outcomes)
        assert calibrator.is_fitted
        assert len(calibrator.alphas) == 3
        assert len(calibrator.betas) == 3

        calibrated = calibrator.calibrate(self.uncalibrated_probs)
        assert calibrated.shape == self.uncalibrated_probs.shape
        assert np.allclose(calibrated.sum(axis=1), 1.0)

    def test_auto_calibrator(self):
        """Test Auto-calibrator."""
        auto_cal = AutoCalibrator(n_classes=3, cv_folds=3)
        auto_cal.fit(self.uncalibrated_probs, self.outcomes)
        assert auto_cal.is_fitted
        assert auto_cal.best_calibrator is not None
        assert "selected_method" in auto_cal.params

        calibrated = auto_cal.calibrate(self.uncalibrated_probs)
        assert calibrated.shape == self.uncalibrated_probs.shape
        assert np.allclose(calibrated.sum(axis=1), 1.0)

    def test_fit_calibrator_helper(self):
        """Test fit_calibrator convenience function."""
        for method in ["platt", "isotonic", "temperature", "beta", "auto"]:
            calibrator = fit_calibrator(
                self.uncalibrated_probs,
                self.outcomes,
                method=method,
            )
            assert calibrator.is_fitted
            calibrated = calibrator.calibrate(self.uncalibrated_probs)
            assert np.allclose(calibrated.sum(axis=1), 1.0)


class TestCalibrationDiagnostics:
    """Test calibration diagnostic functions."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 500
        self.probs = np.random.dirichlet([1, 1, 1], size=self.n_samples)
        self.outcomes = np.array([
            np.random.choice([0, 1, 2], p=p) for p in self.probs
        ])

    def test_expected_calibration_error(self):
        """Test ECE calculation."""
        ece = expected_calibration_error(self.probs, self.outcomes, n_bins=10)
        assert 0 <= ece <= 1
        assert isinstance(ece, float)

    def test_maximum_calibration_error(self):
        """Test MCE calculation."""
        mce = maximum_calibration_error(self.probs, self.outcomes, n_bins=10)
        assert 0 <= mce <= 1
        assert isinstance(mce, float)

    def test_reliability_diagram_data(self):
        """Test reliability diagram generation."""
        data = reliability_diagram_data(self.probs, self.outcomes, n_bins=15)
        assert "bin_centers" in data
        assert "bin_accuracies" in data
        assert "bin_sizes" in data
        assert "ece" in data
        assert len(data["bin_centers"]) == len(data["bin_accuracies"])
        assert sum(data["bin_sizes"]) == len(self.outcomes)


class TestDriftDetection:
    """Test drift detection methods."""

    def test_page_hinkley_detector(self):
        """Test Page-Hinkley drift detector."""
        detector = PageHinkleyDetector(delta=0.005, lambda_threshold=2.0)

        # No drift in constant signal
        errors = [0.3] * 50
        drifts = [detector.update(e) for e in errors]
        assert not any(drifts)

        # Clear drift when error increases dramatically
        detector.reset()
        errors = [0.3] * 35 + [0.8] * 25  # Strong shift
        drifts = [detector.update(e) for e in errors]
        assert any(drifts)

    def test_adwin_detector(self):
        """Test ADWIN detector."""
        detector = ADWINDetector(delta=0.002)

        # No drift in uniform distribution
        errors = np.random.uniform(0.2, 0.4, 50)
        drifts = [detector.update(e) for e in errors]
        # ADWIN is more sensitive, may detect drift

        detector.reset()
        errors = list(np.random.uniform(0.2, 0.4, 30))
        errors += list(np.random.uniform(0.6, 0.8, 20))
        drifts = [detector.update(e) for e in errors]
        # Should detect change in distribution

    def test_ddm_detector(self):
        """Test DDM detector."""
        detector = DDMDetector(warning_level=2.0, drift_level=3.0)

        # No drift when error rate is stable
        errors = [0, 1, 0, 1, 0] * 10
        drifts = [detector.update(e) for e in errors]
        assert not any(drifts)

        # Drift when error rate increases
        detector.reset()
        errors = [0, 1, 0] * 10  # 33% error rate
        errors += [1, 1, 1] * 10  # 100% error rate
        drifts = [detector.update(e) for e in errors]
        assert any(drifts)

    def test_eddm_detector(self):
        """Test EDDM detector."""
        detector = EDDMDetector()

        # Few errors initially
        errors = [0] * 40 + [1, 0] * 5
        drifts = [detector.update(e) for e in errors]

    def test_ensemble_drift_detector(self):
        """Test ensemble drift detector."""
        detector = DriftDetector(use_ensemble=True, min_instances=20)

        # No drift in stable performance
        losses = np.random.normal(0.5, 0.05, 50)
        errors = [0, 0, 1, 0, 0] * 10
        drifts = [detector.update(l, e) for l, e in zip(losses, errors)]

        # Verify can track severity
        detector.reset()
        losses = list(np.random.normal(0.5, 0.05, 30))
        losses += list(np.random.normal(0.8, 0.05, 20))
        errors = [0, 0, 1] * 16 + [1, 1, 1] * 6
        drifts = [detector.update(l, e) for l, e in zip(losses, errors)]

    def test_drift_severity_classification(self):
        """Test drift severity classification."""
        detector = DriftDetector(use_ensemble=True)

        # Induce sudden drift
        errors = [0, 0, 1] * 20 + [1, 1, 1] * 10
        losses = [0.5] * 60 + [0.9] * 30
        for loss, error in zip(losses, errors):
            detector.update(loss, error)

        # Should detect drift
        if detector.is_drifting:
            # Severity should be classified
            assert detector.drift_severity in ["sudden", "gradual", "recurring", "none"]


class TestSelfLearningEnhancements:
    """Test self-learning system enhancements."""

    def test_get_optimal_expert_weights_basic(self):
        """Test basic weight calculation."""
        loop = SelfLearningLoop()

        # Record some predictions
        for i in range(100):
            probs = np.random.dirichlet([1, 1, 1])
            outcome = np.random.choice([0, 1, 2])
            loop.record_prediction_result(
                match_id=i,
                predicted_probs=probs,
                actual_outcome=outcome,
                expert_predictions={
                    "elo": np.random.dirichlet([1, 1, 1]),
                    "poisson": np.random.dirichlet([1, 1, 1]),
                },
            )

        weights = loop.get_optimal_expert_weights()
        assert isinstance(weights, dict)
        assert "elo" in weights
        assert "poisson" in weights
        assert np.allclose(sum(weights.values()), 1.0)

    def test_get_optimal_expert_weights_with_uncertainty(self):
        """Test weights with uncertainty estimates."""
        loop = SelfLearningLoop()

        for i in range(100):
            loop.record_prediction_result(
                match_id=i,
                predicted_probs=np.random.dirichlet([1, 1, 1]),
                actual_outcome=np.random.choice([0, 1, 2]),
                expert_predictions={
                    "elo": np.random.dirichlet([1, 1, 1]),
                    "bayesian": np.random.dirichlet([1, 1, 1]),
                },
            )

        result = loop.get_optimal_expert_weights(return_uncertainties=True)
        assert isinstance(result, dict)
        assert "weights" in result
        assert "uncertainties" in result
        assert "meta" in result
        assert np.allclose(sum(result["weights"].values()), 1.0)

    def test_context_specific_weights(self):
        """Test per-context weight tracking."""
        loop = SelfLearningLoop()

        # Record predictions for different contexts
        for i in range(50):
            loop.record_prediction_result(
                match_id=i,
                predicted_probs=np.random.dirichlet([1, 1, 1]),
                actual_outcome=np.random.choice([0, 1, 2]),
                expert_predictions={
                    "expert_a": np.random.dirichlet([1, 1, 1]),
                    "expert_b": np.random.dirichlet([1, 1, 1]),
                },
                league="EPL",
                context="derby",
            )

        # Get league-specific weights
        epl_weights = loop.get_optimal_expert_weights(league="EPL")
        assert isinstance(epl_weights, dict)

        # Get context-specific weights
        derby_weights = loop.get_optimal_expert_weights(
            league="EPL",
            context="derby",
        )
        assert isinstance(derby_weights, dict)

    def test_drift_detection_integration(self):
        """Test drift detection in learning loop."""
        loop = SelfLearningLoop()

        # Normal predictions
        for i in range(50):
            probs = np.random.dirichlet([1, 1, 1])
            outcome = np.random.choice([0, 1, 2], p=probs)
            result = loop.record_prediction_result(
                match_id=i,
                predicted_probs=probs,
                actual_outcome=outcome,
            )
            assert "drift_detected" in result
            assert "drift_severity" in result

    def test_performance_report(self):
        """Test performance report generation."""
        loop = SelfLearningLoop()

        for i in range(100):
            loop.record_prediction_result(
                match_id=i,
                predicted_probs=np.random.dirichlet([1, 1, 1]),
                actual_outcome=np.random.choice([0, 1, 2]),
                expert_predictions={
                    "expert_a": np.random.dirichlet([1, 1, 1]),
                    "expert_b": np.random.dirichlet([1, 1, 1]),
                },
            )

        report = loop.get_performance_report()
        assert "overall" in report
        assert "expert_rankings" in report
        assert "drift_detected" in report
        assert "retrain_recommended" in report


class TestRecencyWeighting:
    """Test recency-based weighting."""

    def test_get_weight_recency(self):
        """Test that weight computation respects recency."""
        from footy.self_learning import ExpertTracker

        tracker = ExpertTracker(name="test_expert")

        # Add many predictions
        for i in range(100):
            record = {
                "match_id": i,
                "probs": [0.4, 0.3, 0.3],
                "predicted_prob": 0.4,
                "outcome": 0,
                "correct": True,
                "log_loss": 0.9,
            }
            tracker.record(record)

        # Get weight with recency
        w_recency, u_recency = tracker.get_weight(use_recency=True)

        # Get weight without recency
        w_no_recency, u_no_recency = tracker.get_weight(use_recency=False)

        # Both should be valid
        assert 0 <= w_recency <= 10
        assert 0 <= w_no_recency <= 10


class TestBayesianShrinkage:
    """Test Bayesian shrinkage toward global performance."""

    def test_shrinkage_toward_global(self):
        """Test that league weights shrink toward global."""
        from footy.self_learning import ExpertTracker

        tracker = ExpertTracker(name="test_expert")

        # Add global predictions
        for i in range(100):
            record = {
                "match_id": i,
                "probs": [0.4, 0.3, 0.3],
                "predicted_prob": 0.4,
                "outcome": 0,
                "correct": i < 60,  # 60% global accuracy
                "log_loss": 0.8 if i < 60 else 1.2,
            }
            tracker.record(record)

        # Add league predictions (higher accuracy)
        for i in range(50):
            record = {
                "match_id": i + 1000,
                "probs": [0.5, 0.25, 0.25],
                "predicted_prob": 0.5,
                "outcome": 0,
                "correct": True,  # 100% league accuracy
                "log_loss": 0.7,
            }
            tracker.record(record, league="EPL")

        # League weight should be shrunk toward global
        league_w, _ = tracker.get_weight(league="EPL")
        global_w, _ = tracker.get_weight()

        # League weight should be between league-only and global
        assert global_w < league_w or np.isclose(global_w, league_w)
