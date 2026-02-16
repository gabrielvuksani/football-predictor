"""Tests for walk-forward cross-validation module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footy.walkforward import (
    FoldResult,
    WalkForwardResult,
    _compute_ece,
    analyze_calibration,
    feature_stability_report,
    walk_forward_cv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 2000, n_features: int = 20, seed: int = 42):
    """Generate a synthetic time-series classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    # Labels correlated with first few features
    logit = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2] + rng.standard_normal(n) * 0.5
    probs_h = 1 / (1 + np.exp(-logit))
    y = np.zeros(n, dtype=int)
    for i in range(n):
        r = rng.random()
        if r < probs_h[i] * 0.5:
            y[i] = 0  # home
        elif r < probs_h[i] * 0.5 + 0.25:
            y[i] = 1  # draw
        else:
            y[i] = 2  # away

    # Dates: one match per day over ~5.5 years
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return X, y, dates


# ---------------------------------------------------------------------------
# _compute_ece
# ---------------------------------------------------------------------------

class TestComputeECE:
    def test_perfect_calibration(self):
        """When predicted confidence matches accuracy, ECE ≈ 0."""
        # 100% confident and 100% correct → ECE = 0
        probs = np.array([[1.0, 0.0, 0.0]] * 50)
        labels = np.zeros(50, dtype=int)
        ece = _compute_ece(probs, labels)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_overconfident(self):
        """When model is overconfident, ECE > 0."""
        # 90% confident but always wrong
        probs = np.array([[0.9, 0.05, 0.05]] * 100)
        labels = np.ones(100, dtype=int)
        ece = _compute_ece(probs, labels)
        assert ece > 0.5

    def test_uniform_predictions(self):
        """Uniform predictions → moderate ECE."""
        probs = np.full((100, 3), 1 / 3)
        labels = np.random.default_rng(42).integers(0, 3, 100)
        ece = _compute_ece(probs, labels)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# analyze_calibration
# ---------------------------------------------------------------------------

class TestAnalyzeCalibration:
    def test_returns_list_of_dicts(self):
        probs = np.random.default_rng(42).dirichlet([2, 1.5, 1.5], size=200)
        actuals = np.random.default_rng(42).integers(0, 3, 200)
        result = analyze_calibration(probs, actuals)
        assert isinstance(result, list)
        assert all(isinstance(d, dict) for d in result)
        assert len(result) > 0

    def test_required_keys(self):
        probs = np.random.default_rng(1).dirichlet([2, 2, 2], size=300)
        actuals = np.random.default_rng(1).integers(0, 3, 300)
        result = analyze_calibration(probs, actuals, n_bins=5)
        for d in result:
            assert "bin_lower" in d
            assert "bin_upper" in d
            assert "avg_confidence" in d
            assert "avg_accuracy" in d
            assert "count" in d
            assert "fraction" in d

    def test_counts_sum_to_total(self):
        n = 500
        probs = np.random.default_rng(7).dirichlet([3, 2, 1], size=n)
        actuals = np.random.default_rng(7).integers(0, 3, n)
        result = analyze_calibration(probs, actuals)
        total = sum(d["count"] for d in result)
        assert total == n


# ---------------------------------------------------------------------------
# FoldResult / WalkForwardResult dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_fold_result_creation(self):
        fr = FoldResult(
            fold=1, train_size=500, test_size=100,
            train_start="2020-01-01", train_end="2021-06-30",
            test_start="2021-07-01", test_end="2021-09-30",
            logloss=0.95, brier=0.22, accuracy=0.42, ece=0.05,
        )
        assert fr.fold == 1
        assert fr.train_size == 500
        assert fr.btts_accuracy is None
        assert isinstance(fr.feature_importances, dict)

    def test_walk_forward_result_creation(self):
        wfr = WalkForwardResult(
            n_folds=3, total_test_samples=300,
            mean_logloss=0.90, mean_brier=0.20,
            mean_accuracy=0.45, mean_ece=0.04,
            std_logloss=0.02, std_accuracy=0.03,
            folds=[],
        )
        assert wfr.n_folds == 3
        assert wfr.all_predictions is None
        assert isinstance(wfr.feature_importances, dict)


# ---------------------------------------------------------------------------
# feature_stability_report
# ---------------------------------------------------------------------------

class TestFeatureStabilityReport:
    def test_empty_importances(self):
        wfr = WalkForwardResult(
            n_folds=2, total_test_samples=100,
            mean_logloss=1.0, mean_brier=0.3,
            mean_accuracy=0.33, mean_ece=0.1,
            std_logloss=0.1, std_accuracy=0.05,
            folds=[],
        )
        result = feature_stability_report(wfr)
        assert result == []

    def test_stable_features_flagged(self):
        # Create folds with consistent feature importances
        folds = [
            FoldResult(
                fold=i, train_size=500, test_size=100,
                train_start="2020-01-01", train_end="2021-01-01",
                test_start="2021-01-01", test_end="2021-04-01",
                logloss=0.9, brier=0.2, accuracy=0.45, ece=0.05,
                feature_importances={"feat_a": 0.1 + i * 0.001, "feat_b": 0.05},
            )
            for i in range(4)
        ]
        wfr = WalkForwardResult(
            n_folds=4, total_test_samples=400,
            mean_logloss=0.9, mean_brier=0.2,
            mean_accuracy=0.45, mean_ece=0.05,
            std_logloss=0.02, std_accuracy=0.01,
            folds=folds,
            feature_importances={"feat_a": 0.1015, "feat_b": 0.05},
        )
        report = feature_stability_report(wfr, top_n=5)
        assert len(report) == 2
        assert report[0]["name"] == "feat_a"
        assert report[0]["stable"] is True

    def test_report_keys(self):
        folds = [
            FoldResult(
                fold=1, train_size=500, test_size=100,
                train_start="2020-01-01", train_end="2021-01-01",
                test_start="2021-01-01", test_end="2021-04-01",
                logloss=0.9, brier=0.2, accuracy=0.45, ece=0.05,
                feature_importances={"x": 0.5},
            ),
        ]
        wfr = WalkForwardResult(
            n_folds=1, total_test_samples=100,
            mean_logloss=0.9, mean_brier=0.2,
            mean_accuracy=0.45, mean_ece=0.05,
            std_logloss=0.0, std_accuracy=0.0,
            folds=folds,
            feature_importances={"x": 0.5},
        )
        report = feature_stability_report(wfr)
        for item in report:
            assert "name" in item
            assert "mean_importance" in item
            assert "std_importance" in item
            assert "cv" in item
            assert "stable" in item


# ---------------------------------------------------------------------------
# walk_forward_cv (integration)
# ---------------------------------------------------------------------------

class TestWalkForwardCV:
    """Integration tests with synthetic data — these train actual models."""

    @pytest.fixture()
    def dataset(self):
        return _make_dataset(n=2000, n_features=15)

    def test_basic_run(self, dataset):
        """Walk-forward CV completes and returns valid result."""
        X, y, dates = dataset
        result = walk_forward_cv(
            X, y, dates,
            n_folds=3,
            min_train_frac=0.3,
            test_months=4,
            calibrate=False,
            verbose=False,
            model_params={"max_iter": 100, "max_depth": 3},
        )
        assert isinstance(result, WalkForwardResult)
        assert result.n_folds >= 1
        assert result.total_test_samples > 0
        assert 0.0 < result.mean_logloss < 5.0
        assert 0.0 <= result.mean_accuracy <= 1.0
        assert 0.0 <= result.mean_ece <= 1.0

    def test_feature_names_tracked(self, dataset):
        """Feature importances are tracked when names are provided."""
        X, y, dates = dataset
        names = [f"f_{i}" for i in range(X.shape[1])]
        result = walk_forward_cv(
            X, y, dates,
            n_folds=2,
            min_train_frac=0.4,
            test_months=6,
            calibrate=False,
            verbose=False,
            feature_names=names,
            model_params={"max_iter": 50, "max_depth": 3},
        )
        assert len(result.feature_importances) > 0
        for fname in result.feature_importances:
            assert fname in names

    def test_predictions_collected(self, dataset):
        """All out-of-sample predictions are concatenated."""
        X, y, dates = dataset
        result = walk_forward_cv(
            X, y, dates,
            n_folds=2,
            min_train_frac=0.4,
            test_months=6,
            calibrate=False,
            verbose=False,
            model_params={"max_iter": 50, "max_depth": 3},
        )
        if result.all_predictions is not None:
            assert result.all_predictions.shape[1] == 3
            assert len(result.all_predictions) == result.total_test_samples

    def test_too_little_data_raises(self):
        """Should raise ValueError if not enough folds complete."""
        X = np.random.default_rng(1).standard_normal((50, 5))
        y = np.random.default_rng(1).integers(0, 3, 50)
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        with pytest.raises(ValueError, match="No valid folds"):
            walk_forward_cv(
                X, y, dates,
                n_folds=5,
                min_train_frac=0.8,
                calibrate=False,
                verbose=False,
            )

    def test_sliding_window(self, dataset):
        """Sliding window mode works (expanding=False)."""
        X, y, dates = dataset
        result = walk_forward_cv(
            X, y, dates,
            n_folds=2,
            min_train_frac=0.3,
            test_months=6,
            expanding=False,
            max_train_months=12,
            calibrate=False,
            verbose=False,
            model_params={"max_iter": 50, "max_depth": 3},
        )
        assert result.n_folds >= 1

    def test_auxiliary_btts(self, dataset):
        """BTTS auxiliary labels are evaluated when provided."""
        X, y, dates = dataset
        rng = np.random.default_rng(99)
        y_btts = rng.integers(0, 2, len(y))
        result = walk_forward_cv(
            X, y, dates,
            n_folds=2,
            min_train_frac=0.4,
            test_months=6,
            calibrate=False,
            verbose=False,
            y_btts=y_btts,
            model_params={"max_iter": 50, "max_depth": 3},
        )
        # At least one fold should have BTTS accuracy
        btts_accs = [f.btts_accuracy for f in result.folds if f.btts_accuracy is not None]
        assert len(btts_accs) > 0
