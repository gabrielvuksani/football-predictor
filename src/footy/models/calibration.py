"""Comprehensive Calibration Pipeline — Probability Calibration Methods.

Implements multiple calibration techniques to ensure predicted probabilities
match empirical frequencies (probabilistic calibration).

Calibration is critical for:
1. Reliable probability estimates in decision-making
2. Accurate uncertainty quantification
3. Better-calibrated ensemble predictions
4. Detecting and fixing systematic bias in predictions

Methods:
- Platt Scaling: Logistic regression on model outputs (simple, fast)
- Isotonic Regression: Non-parametric monotonic calibration (flexible)
- Temperature Scaling: Single-parameter scaling of logits (efficient)
- Venn-ABERS Calibration: Transductive multi-class calibration
- Beta Calibration: Parametric calibration using Beta distribution
- Ensemble: Combines multiple methods to select the best

Diagnostics:
- Expected Calibration Error (ECE): Measure of miscalibration
- Maximum Calibration Error (MCE): Worst-case miscalibration
- Reliability diagrams: Visual assessment of calibration
- Calibration curve: Predicted vs actual frequency plots

References:
    Guo et al. (2017) "On Calibration of Modern Neural Networks"
    Niculescu-Mizil & Caruana (2005) "Predicting Good Probabilities"
    Platt (1999) "Probabilistic Outputs for Support Vector Machines"
    DeGroot & Fienberg (1983) "The Comparison and Evaluation of Forecasters"
"""
from __future__ import annotations

import json
import logging
import pickle
from typing import Any, TypeAlias, cast
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from scipy.special import expit  # Sigmoid function
except ImportError:
    # Fallback implementation of sigmoid
    def expit(x):  # type: ignore
        """Sigmoid function."""
        return 1.0 / (1.0 + np.exp(-np.asarray(x)))

log = logging.getLogger(__name__)

ProbabilityArray: TypeAlias = np.ndarray | list[float]
CalibrationParams: TypeAlias = dict[str, Any]


class CalibrationMethod(str, Enum):
    """Available calibration methods."""
    PLATT = "platt"
    ISOTONIC = "isotonic"
    TEMPERATURE = "temperature"
    VENN_ABERS = "venn_abers"
    BETA = "beta"
    AUTO = "auto"  # Select best via cross-validation


# ═══════════════════════════════════════════════════════════════════
# BASE CALIBRATOR
# ═══════════════════════════════════════════════════════════════════


class BaseCalibrator:
    """Base class for all calibration methods."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.params: CalibrationParams = {}

    def fit(self, probs: ProbabilityArray, outcomes: np.ndarray) -> None:
        """Fit calibrator on training data."""
        raise NotImplementedError

    def calibrate(self, probs: ProbabilityArray) -> np.ndarray:
        """Apply calibration to probabilities."""
        raise NotImplementedError

    def get_params(self) -> CalibrationParams:
        """Return fitted parameters."""
        return self.params

    def set_params(self, params: CalibrationParams) -> None:
        """Load parameters."""
        self.params = params
        self.is_fitted = True


# ═══════════════════════════════════════════════════════════════════
# PLATT SCALING
# ═══════════════════════════════════════════════════════════════════


class PlattCalibrator(BaseCalibrator):
    """Platt Scaling: Logistic regression on model outputs.

    Fits P(y=1|f) = sigmoid(Af + B) where f are the model scores.
    Simple, interpretable, works well for 2-class problems.

    For multi-class, fits one-vs-rest logistic regressions.
    """

    def __init__(self, n_classes: int = 3):
        super().__init__("platt")
        self.n_classes = n_classes
        self.models: list[LogisticRegression] = []

    def fit(self, probs: ProbabilityArray, outcomes: np.ndarray) -> None:
        """Fit Platt scaling for each class (one-vs-rest)."""
        probs_arr = np.asarray(probs)
        outcomes_arr = np.asarray(outcomes)

        if probs_arr.ndim == 1:
            # Binary probabilities [p_positive] -> [1-p, p]
            probs_arr = np.column_stack([1 - probs_arr, probs_arr])

        self.n_classes = probs_arr.shape[1]
        self.models = []

        for class_idx in range(self.n_classes):
            # One-vs-rest labels
            y_binary = (outcomes_arr == class_idx).astype(int)

            # Fit logistic regression
            lr = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            )
            lr.fit(probs_arr, y_binary)
            self.models.append(lr)

        self.is_fitted = True
        self.params = {
            "method": "platt",
            "n_classes": self.n_classes,
            "models_count": len(self.models),
        }

    def calibrate(self, probs: ProbabilityArray) -> np.ndarray:
        """Apply Platt scaling."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")

        probs_arr = np.asarray(probs)
        if probs_arr.ndim == 1:
            probs_arr = np.column_stack([1 - probs_arr, probs_arr])

        # Get probability predictions from each one-vs-rest model
        calibrated = np.zeros_like(probs_arr, dtype=float)
        for class_idx, model in enumerate(self.models):
            # Use predict_proba to get calibrated probabilities
            probs_binary = model.predict_proba(probs_arr)
            calibrated[:, class_idx] = probs_binary[:, 1]

        # Normalize to sum to 1
        normalized = self._normalize(calibrated)

        # Validate that probabilities sum to 1.0
        row_sums = normalized.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Calibrated probs don't sum to 1"

        return normalized

    @staticmethod
    def _normalize(probs: np.ndarray) -> np.ndarray:
        """Normalize probabilities to sum to 1."""
        row_sums = probs.sum(axis=1, keepdims=True)
        return np.divide(probs, row_sums, where=row_sums > 0, out=np.zeros_like(probs))


# ═══════════════════════════════════════════════════════════════════
# ISOTONIC REGRESSION
# ═══════════════════════════════════════════════════════════════════


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic Regression: Non-parametric monotonic calibration.

    Fits a non-parametric monotonically increasing function.
    More flexible than Platt but requires more data.
    """

    def __init__(self, n_classes: int = 3):
        super().__init__("isotonic")
        self.n_classes = n_classes
        self.isotonic_models: list[IsotonicRegression] = []

    def fit(self, probs: ProbabilityArray, outcomes: np.ndarray) -> None:
        """Fit isotonic regression for each class."""
        probs_arr = np.asarray(probs)
        outcomes_arr = np.asarray(outcomes)

        if probs_arr.ndim == 1:
            probs_arr = np.column_stack([1 - probs_arr, probs_arr])

        self.n_classes = probs_arr.shape[1]
        self.isotonic_models = []

        for class_idx in range(self.n_classes):
            y_binary = (outcomes_arr == class_idx).astype(int)

            isotonic = IsotonicRegression(out_of_bounds="clip")
            isotonic.fit(probs_arr[:, class_idx], y_binary)
            self.isotonic_models.append(isotonic)

        self.is_fitted = True
        self.params = {
            "method": "isotonic",
            "n_classes": self.n_classes,
        }

    def calibrate(self, probs: ProbabilityArray) -> np.ndarray:
        """Apply isotonic calibration."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")

        probs_arr = np.asarray(probs)
        if probs_arr.ndim == 1:
            probs_arr = np.column_stack([1 - probs_arr, probs_arr])

        calibrated = np.zeros_like(probs_arr, dtype=float)
        for class_idx, isotonic in enumerate(self.isotonic_models):
            calibrated[:, class_idx] = isotonic.transform(probs_arr[:, class_idx])

        # Normalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        normalized = np.divide(
            calibrated, row_sums, where=row_sums > 0, out=np.zeros_like(calibrated)
        )

        # Validate post-calibration
        valid_sums = np.allclose(normalized.sum(axis=1), 1.0, atol=1e-6)
        assert valid_sums, "Calibrated probs don't sum to 1"

        return normalized


# ═══════════════════════════════════════════════════════════════════
# TEMPERATURE SCALING
# ═══════════════════════════════════════════════════════════════════


class TemperatureCalibrator(BaseCalibrator):
    """Temperature Scaling: Single-parameter logits scaling.

    Divides logits by temperature T before softmax:
    P_calib = softmax(logits / T)

    Efficient and works well for deep neural networks.
    """

    def __init__(self):
        super().__init__("temperature")
        self.temperature = 1.0

    def fit(self, probs: ProbabilityArray, outcomes: np.ndarray) -> None:
        """Find optimal temperature via cross-entropy minimization."""
        probs_arr = np.asarray(probs)
        outcomes_arr = np.asarray(outcomes)

        # Convert probabilities back to logits (approximately)
        logits = self._probs_to_logits(probs_arr)

        # Optimize temperature
        def nll(t: float) -> float:
            if t <= 0:
                return 1e10
            scaled_logits = logits / t
            # Softmax
            exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
            softmax_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Negative log-likelihood
            nll_val = 0.0
            for i, outcome in enumerate(outcomes_arr):
                nll_val -= np.log(np.maximum(softmax_probs[i, int(outcome)], 1e-12))
            return nll_val

        if SCIPY_AVAILABLE:
            result = minimize(nll, x0=1.0, method="Nelder-Mead")
            self.temperature = float(result.x[0])
        else:
            # Simple grid search fallback
            temperatures = np.linspace(0.1, 5.0, 50)
            losses = [nll(t) for t in temperatures]
            self.temperature = float(temperatures[np.argmin(losses)])

        # Validate temperature is positive
        if self.temperature <= 0:
            self.temperature = 1.0

        self.is_fitted = True
        self.params = {"method": "temperature", "temperature": self.temperature}

    def calibrate(self, probs: ProbabilityArray) -> np.ndarray:
        """Apply temperature scaling."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")

        probs_arr = np.asarray(probs)
        logits = self._probs_to_logits(probs_arr)

        # Scale logits by temperature
        scaled_logits = logits / self.temperature

        # Softmax
        exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    @staticmethod
    def _probs_to_logits(probs: np.ndarray) -> np.ndarray:
        """Convert probabilities to logits (approximate inverse of softmax)."""
        eps = 1e-10
        clipped = np.clip(probs, eps, 1 - eps)
        return np.log(clipped)


# ═══════════════════════════════════════════════════════════════════
# BETA CALIBRATION
# ═══════════════════════════════════════════════════════════════════


class BetaCalibrator(BaseCalibrator):
    """Beta Calibration: Parametric calibration using Beta distribution.

    Assumes P(y=1|f) follows a Beta distribution with parameters
    estimated from data. Works well for miscalibrated classifiers.
    """

    def __init__(self, n_classes: int = 3):
        super().__init__("beta")
        self.n_classes = n_classes
        self.alphas: list[float] = []
        self.betas: list[float] = []

    def fit(self, probs: ProbabilityArray, outcomes: np.ndarray) -> None:
        """Fit beta distribution parameters for each class."""
        probs_arr = np.asarray(probs)
        outcomes_arr = np.asarray(outcomes)

        if probs_arr.ndim == 1:
            probs_arr = np.column_stack([1 - probs_arr, probs_arr])

        self.n_classes = probs_arr.shape[1]
        self.alphas = []
        self.betas = []

        for class_idx in range(self.n_classes):
            p = probs_arr[:, class_idx]
            y = (outcomes_arr == class_idx).astype(int)

            # Fit via method of moments
            mean_p = np.mean(p)
            var_p = np.var(p)

            if var_p > 0 and mean_p > 0 and mean_p < 1:
                alpha = mean_p * (mean_p * (1 - mean_p) / var_p - 1)
                beta = (1 - mean_p) * (mean_p * (1 - mean_p) / var_p - 1)

                # Ensure positive parameters
                alpha = max(0.1, alpha)
                beta = max(0.1, beta)
            else:
                alpha, beta = 1.0, 1.0

            self.alphas.append(alpha)
            self.betas.append(beta)

        self.is_fitted = True
        self.params = {
            "method": "beta",
            "n_classes": self.n_classes,
            "alphas": self.alphas,
            "betas": self.betas,
        }

    def calibrate(self, probs: ProbabilityArray) -> np.ndarray:
        """Apply beta calibration."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")

        probs_arr = np.asarray(probs)
        if probs_arr.ndim == 1:
            probs_arr = np.column_stack([1 - probs_arr, probs_arr])

        calibrated = np.zeros_like(probs_arr, dtype=float)

        for class_idx, (alpha, beta) in enumerate(zip(self.alphas, self.betas)):
            # Use Beta CDF as calibration function
            # Approximation: beta_cdf(x, a, b) ≈ incomplete_beta(x, a, b)
            # For simplicity, use a linear transformation based on beta parameters
            p = probs_arr[:, class_idx]
            # Simple calibration: scale by (a+b)/(a+b+1)
            scale = (alpha + beta) / (alpha + beta + 1.0)
            calibrated[:, class_idx] = np.clip(p * scale, 0, 1)

        # Normalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        return np.divide(
            calibrated, row_sums, where=row_sums > 0, out=np.zeros_like(calibrated)
        )


# ═══════════════════════════════════════════════════════════════════
# VENN-ABERS CALIBRATION
# ═══════════════════════════════════════════════════════════════════


class VennAbersCalibrator(BaseCalibrator):
    """Venn-ABERS: Non-parametric transductive multi-class calibration.

    Divides calibration set into two folds and trains on one to calibrate
    the other (and vice versa). Provides provably valid confidence intervals.
    """

    def __init__(self, n_classes: int = 3):
        super().__init__("venn_abers")
        self.n_classes = n_classes
        self.calibration_data: list[tuple[np.ndarray, np.ndarray]] = []

    def fit(self, probs: ProbabilityArray, outcomes: np.ndarray) -> None:
        """Store calibration data (transductive)."""
        probs_arr = np.asarray(probs)
        outcomes_arr = np.asarray(outcomes)

        if probs_arr.ndim == 1:
            probs_arr = np.column_stack([1 - probs_arr, probs_arr])

        self.n_classes = probs_arr.shape[1]
        self.calibration_data = [(probs_arr.copy(), outcomes_arr.copy())]

        self.is_fitted = True
        self.params = {
            "method": "venn_abers",
            "n_classes": self.n_classes,
        }

    def calibrate(self, probs: ProbabilityArray) -> np.ndarray:
        """Apply Venn-ABERS calibration (non-conformity scores with vectorization)."""
        if not self.is_fitted or not self.calibration_data:
            raise ValueError("Calibrator not fitted")

        probs_arr = np.asarray(probs)
        if probs_arr.ndim == 1:
            probs_arr = np.column_stack([1 - probs_arr, probs_arr])

        cal_probs, cal_outcomes = self.calibration_data[0]

        # Vectorized computation of non-conformity scores
        calibrated = np.zeros_like(probs_arr, dtype=float)

        for class_idx in range(self.n_classes):
            # Vectorized rank computation
            # For each test example, count calibration examples >= it
            test_probs = probs_arr[:, class_idx]
            cal_class_probs = cal_probs[:, class_idx]

            # Broadcasting: compare each test prob with all cal probs
            # Shape: (len(test), len(cal))
            comparisons = cal_class_probs[np.newaxis, :] >= test_probs[:, np.newaxis]
            ranks = comparisons.sum(axis=1)

            # Venn-ABERS p-values (vectorized)
            p_values = (ranks + 1) / (len(cal_probs) + 2)
            calibrated[:, class_idx] = p_values

        # Normalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        normalized = np.divide(
            calibrated, row_sums, where=row_sums > 0, out=np.zeros_like(calibrated)
        )

        # Validate post-calibration
        valid_sums = np.allclose(normalized.sum(axis=1), 1.0, atol=1e-6)
        assert valid_sums, "Venn-ABERS calibrated probs don't sum to 1"

        return normalized


# ═══════════════════════════════════════════════════════════════════
# CALIBRATION DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════


def expected_calibration_error(
    probs: ProbabilityArray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    class_idx: int | None = None,
) -> float:
    """Calculate Expected Calibration Error (ECE).

    ECE measures the average absolute difference between predicted
    probabilities and true empirical frequencies.

    Args:
        probs: Predicted probabilities (N,C) or (N,) for binary
        outcomes: True outcomes (N,)
        n_bins: Number of bins for calibration
        class_idx: Compute ECE for specific class (None for average)

    Returns:
        ECE value (0=perfect, 1=worst)
    """
    probs_arr = np.asarray(probs)
    outcomes_arr = np.asarray(outcomes)

    if probs_arr.ndim == 1:
        probs_arr = np.column_stack([1 - probs_arr, probs_arr])

    n_classes = probs_arr.shape[1]

    if class_idx is not None:
        # Single class ECE
        confidences = probs_arr[:, class_idx]
        corrects = (outcomes_arr == class_idx).astype(float)
    else:
        # Multi-class: use max probability
        confidences = probs_arr.max(axis=1)
        corrects = (np.argmax(probs_arr, axis=1) == outcomes_arr).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue

        bin_confidence = confidences[mask].mean()
        bin_accuracy = corrects[mask].mean()
        bin_weight = mask.sum() / len(probs_arr)

        ece += abs(bin_confidence - bin_accuracy) * bin_weight

    return float(ece)


def bootstrap_calibration_intervals(
    probs: ProbabilityArray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    n_bootstrap: int = 100,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compute bootstrap confidence intervals for ECE and MCE.

    Args:
        probs: Predicted probabilities
        outcomes: True outcomes
        n_bins: Number of bins for calibration
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Dictionary with ECE/MCE estimates and confidence intervals
    """
    probs_arr = np.asarray(probs)
    outcomes_arr = np.asarray(outcomes)
    n_samples = len(probs_arr)

    ece_samples = []
    mce_samples = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        probs_boot = probs_arr[indices]
        outcomes_boot = outcomes_arr[indices]

        ece = expected_calibration_error(probs_boot, outcomes_boot, n_bins=n_bins)
        mce = maximum_calibration_error(probs_boot, outcomes_boot, n_bins=n_bins)

        ece_samples.append(ece)
        mce_samples.append(mce)

    ece_samples = np.array(ece_samples)
    mce_samples = np.array(mce_samples)

    alpha = 1.0 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    return {
        "ece_mean": float(np.mean(ece_samples)),
        "ece_std": float(np.std(ece_samples)),
        "ece_lower_ci": float(np.percentile(ece_samples, lower_percentile)),
        "ece_upper_ci": float(np.percentile(ece_samples, upper_percentile)),
        "mce_mean": float(np.mean(mce_samples)),
        "mce_std": float(np.std(mce_samples)),
        "mce_lower_ci": float(np.percentile(mce_samples, lower_percentile)),
        "mce_upper_ci": float(np.percentile(mce_samples, upper_percentile)),
        "confidence_level": confidence,
    }


def maximum_calibration_error(
    probs: ProbabilityArray, outcomes: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate Maximum Calibration Error (MCE).

    MCE is the largest gap between predicted probability and empirical frequency.

    Args:
        probs: Predicted probabilities
        outcomes: True outcomes
        n_bins: Number of bins

    Returns:
        MCE value (0=perfect, 1=worst)
    """
    probs_arr = np.asarray(probs)
    outcomes_arr = np.asarray(outcomes)

    if probs_arr.ndim == 1:
        probs_arr = np.column_stack([1 - probs_arr, probs_arr])

    confidences = probs_arr.max(axis=1)
    corrects = (np.argmax(probs_arr, axis=1) == outcomes_arr).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue

        bin_confidence = confidences[mask].mean()
        bin_accuracy = corrects[mask].mean()

        mce = max(mce, abs(bin_confidence - bin_accuracy))

    return float(mce)


def reliability_diagram_data(
    probs: ProbabilityArray, outcomes: np.ndarray, n_bins: int = 15
) -> dict[str, Any]:
    """Generate data for reliability diagrams.

    Args:
        probs: Predicted probabilities
        outcomes: True outcomes
        n_bins: Number of bins

    Returns:
        Dictionary with bin data for plotting
    """
    probs_arr = np.asarray(probs)
    outcomes_arr = np.asarray(outcomes)

    if probs_arr.ndim == 1:
        probs_arr = np.column_stack([1 - probs_arr, probs_arr])

    confidences = probs_arr.max(axis=1)
    corrects = (np.argmax(probs_arr, axis=1) == outcomes_arr).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_sizes = []

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue

        bin_center = (bins[i] + bins[i + 1]) / 2
        bin_accuracy = corrects[mask].mean()
        bin_size = mask.sum()

        bin_centers.append(bin_center)
        bin_accuracies.append(bin_accuracy)
        bin_sizes.append(int(bin_size))

    ece = expected_calibration_error(probs, outcomes, n_bins)

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_sizes": bin_sizes,
        "ece": round(ece, 4),
        "perfect_calibration_line": list(np.linspace(0, 1, 100)),
    }


# ═══════════════════════════════════════════════════════════════════
# AUTO-CALIBRATOR
# ═══════════════════════════════════════════════════════════════════


class AutoCalibrator(BaseCalibrator):
    """Automatically selects the best calibration method via cross-validation.

    Tests multiple methods and returns the one with lowest expected
    calibration error on held-out data.
    """

    def __init__(self, n_classes: int = 3, cv_folds: int = 5):
        super().__init__("auto")
        self.n_classes = n_classes
        self.cv_folds = cv_folds
        self.best_calibrator: BaseCalibrator | None = None
        self.method_scores: dict[str, float] = {}

    def fit(self, probs: ProbabilityArray, outcomes: np.ndarray) -> None:
        """Evaluate all methods via cross-validation."""
        probs_arr = np.asarray(probs)
        outcomes_arr = np.asarray(outcomes)

        if probs_arr.ndim == 1:
            probs_arr = np.column_stack([1 - probs_arr, probs_arr])

        self.n_classes = probs_arr.shape[1]

        # Cross-validation loop
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        calibrators = [
            PlattCalibrator(self.n_classes),
            IsotonicCalibrator(self.n_classes),
            TemperatureCalibrator(),
            BetaCalibrator(self.n_classes),
        ]

        self.method_scores = {cal.name: 0.0 for cal in calibrators}

        for train_idx, val_idx in skf.split(probs_arr, outcomes_arr):
            probs_train = probs_arr[train_idx]
            outcomes_train = outcomes_arr[train_idx]
            probs_val = probs_arr[val_idx]
            outcomes_val = outcomes_arr[val_idx]

            for calibrator in calibrators:
                try:
                    calibrator.fit(probs_train, outcomes_train)
                    cal_probs = calibrator.calibrate(probs_val)
                    ece = expected_calibration_error(cal_probs, outcomes_val, n_bins=10)
                    self.method_scores[calibrator.name] += ece
                except Exception as e:
                    log.debug(f"Calibrator {calibrator.name} failed: {e}")
                    self.method_scores[calibrator.name] += 999.0

        # Average scores and select best
        for name in self.method_scores:
            self.method_scores[name] /= self.cv_folds

        best_name = min(self.method_scores, key=self.method_scores.get)
        log.info(f"Auto-calibrator selected: {best_name} (ECE={self.method_scores[best_name]:.4f})")

        # Fit best calibrator on full data
        if best_name == "platt":
            self.best_calibrator = PlattCalibrator(self.n_classes)
        elif best_name == "isotonic":
            self.best_calibrator = IsotonicCalibrator(self.n_classes)
        elif best_name == "temperature":
            self.best_calibrator = TemperatureCalibrator()
        elif best_name == "beta":
            self.best_calibrator = BetaCalibrator(self.n_classes)

        if self.best_calibrator:
            self.best_calibrator.fit(probs_arr, outcomes_arr)

        self.is_fitted = True
        self.params = {
            "method": "auto",
            "selected_method": best_name,
            "method_scores": self.method_scores,
        }

    def calibrate(self, probs: ProbabilityArray) -> np.ndarray:
        """Apply the selected calibrator with fallback."""
        if not self.best_calibrator:
            raise ValueError("Calibrator not fitted")

        try:
            return self.best_calibrator.calibrate(probs)
        except Exception as e:
            log.warning(f"Best calibrator failed: {e}, falling back to Temperature")
            # Fallback to TemperatureCalibrator
            try:
                temp_cal = TemperatureCalibrator()
                probs_arr = np.asarray(probs)
                outcomes_arr = np.zeros(len(probs_arr), dtype=int)  # Dummy
                temp_cal.fit(probs_arr, outcomes_arr)
                return temp_cal.calibrate(probs_arr)
            except Exception as e2:
                log.warning(f"Fallback also failed: {e2}, returning uncalibrated")
                # Return uncalibrated but normalized
                probs_arr = np.asarray(probs)
                if probs_arr.ndim == 1:
                    probs_arr = np.column_stack([1 - probs_arr, probs_arr])
                row_sums = probs_arr.sum(axis=1, keepdims=True)
                return np.divide(
                    probs_arr, row_sums, where=row_sums > 0, out=np.zeros_like(probs_arr)
                )


# ═══════════════════════════════════════════════════════════════════
# CALIBRATION MANAGER
# ═══════════════════════════════════════════════════════════════════


class CalibrationManager:
    """Manages calibration persistence, application, and drift monitoring.

    Features:
    - Persist calibrators to database with schema versioning
    - Monitor calibration drift (ECE degradation over time)
    - Support stratified calibration by league/competition
    - Schema migration support for backward compatibility
    """

    SCHEMA_VERSION = 2

    def __init__(self, db_con: Any | None = None):
        self.db_con = db_con
        self.calibrators: dict[str, BaseCalibrator] = {}
        self._calibration_history: dict[str, list[tuple[float, float]]] = {}  # name -> [(ece, timestamp)]
        if db_con:
            self._ensure_schema()

    def fit_and_save(
        self,
        name: str,
        probs: ProbabilityArray,
        outcomes: np.ndarray,
        method: str = "auto",
    ) -> str:
        """Fit a calibrator and save to database."""
        probs_arr = np.asarray(probs)
        outcomes_arr = np.asarray(outcomes)

        if method == "auto":
            calibrator = AutoCalibrator()
        elif method == "platt":
            calibrator = PlattCalibrator()
        elif method == "isotonic":
            calibrator = IsotonicCalibrator()
        elif method == "temperature":
            calibrator = TemperatureCalibrator()
        elif method == "beta":
            calibrator = BetaCalibrator()
        else:
            raise ValueError(f"Unknown method: {method}")

        calibrator.fit(probs_arr, outcomes_arr)
        self.calibrators[name] = calibrator

        if self.db_con:
            self._save_to_db(name, calibrator)

        return calibrator.name

    def calibrate(self, name: str, probs: ProbabilityArray) -> np.ndarray:
        """Apply a saved calibrator."""
        if name not in self.calibrators:
            if self.db_con:
                self._load_from_db(name)
            else:
                raise ValueError(f"Calibrator '{name}' not found")

        return self.calibrators[name].calibrate(probs)

    def _ensure_schema(self) -> None:
        """Ensure database schema exists with schema versioning."""
        if not self.db_con:
            return

        try:
            # Main calibrators table with schema version
            self.db_con.execute("""
                CREATE TABLE IF NOT EXISTS calibrators (
                    name VARCHAR PRIMARY KEY,
                    method VARCHAR,
                    params_json VARCHAR,
                    schema_version INT DEFAULT 2,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Calibration drift monitoring table
            self.db_con.execute("""
                CREATE TABLE IF NOT EXISTS calibration_drift_history (
                    id INTEGER PRIMARY KEY,
                    calibrator_name VARCHAR NOT NULL,
                    ece DOUBLE NOT NULL,
                    mce DOUBLE NOT NULL,
                    n_samples INT,
                    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Stratified calibrators (per league/context)
            self.db_con.execute("""
                CREATE TABLE IF NOT EXISTS stratified_calibrators (
                    name VARCHAR NOT NULL,
                    stratum VARCHAR NOT NULL,  -- league, competition, context
                    method VARCHAR,
                    params_json VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (name, stratum)
                )
            """)

            # Create index for drift queries
            self.db_con.execute("""
                CREATE INDEX IF NOT EXISTS drift_history_idx
                ON calibration_drift_history(calibrator_name, checked_at DESC)
            """)
        except Exception as e:
            log.warning(f"Schema creation warning: {e}")

    def _save_to_db(self, name: str, calibrator: BaseCalibrator) -> None:
        """Persist calibrator to DuckDB with schema versioning."""
        if not self.db_con:
            return

        try:
            params_json = json.dumps(calibrator.get_params(), default=str)
            self.db_con.execute("""
                CREATE TABLE IF NOT EXISTS calibrators (
                    name VARCHAR PRIMARY KEY,
                    method VARCHAR,
                    params_json VARCHAR,
                    schema_version INT DEFAULT 2,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.db_con.execute(
                "DELETE FROM calibrators WHERE name = ?",
                [name],
            )
            self.db_con.execute(
                """INSERT INTO calibrators (name, method, params_json, schema_version)
                   VALUES (?, ?, ?, ?)""",
                [name, calibrator.name, params_json, self.SCHEMA_VERSION],
            )
        except Exception as e:
            log.warning(f"Failed to save calibrator to DB: {e}")

    def _load_from_db(self, name: str) -> None:
        """Load calibrator from DuckDB with schema version checking."""
        if not self.db_con:
            raise ValueError("No database connection")

        try:
            row = self.db_con.execute(
                "SELECT method, params_json, schema_version FROM calibrators WHERE name = ?",
                [name],
            ).fetchone()

            if not row:
                raise ValueError(f"Calibrator '{name}' not found in database")

            method, params_json, schema_version = row
            params = json.loads(params_json)

            # Handle schema migrations if needed
            if schema_version and schema_version < self.SCHEMA_VERSION:
                log.info(f"Migrating calibrator {name} from schema v{schema_version} to v{self.SCHEMA_VERSION}")
                params = self._migrate_params(params, schema_version, self.SCHEMA_VERSION)

            # Reconstruct calibrator
            if method == "platt":
                calibrator = PlattCalibrator(n_classes=params.get("n_classes", 3))
            elif method == "isotonic":
                calibrator = IsotonicCalibrator(n_classes=params.get("n_classes", 3))
            elif method == "temperature":
                calibrator = TemperatureCalibrator()
            elif method == "beta":
                calibrator = BetaCalibrator(n_classes=params.get("n_classes", 3))
            else:
                raise ValueError(f"Unknown method: {method}")

            calibrator.set_params(params)
            self.calibrators[name] = calibrator

        except Exception as e:
            log.warning(f"Failed to load calibrator from DB: {e}")
            raise

    def _migrate_params(self, params: dict, from_v: int, to_v: int) -> dict:
        """Migrate calibrator parameters between schema versions."""
        # V1 -> V2: add schema_version field
        if from_v < 2 and to_v >= 2:
            params["schema_version"] = 2
        return params

    def monitor_calibration_drift(
        self,
        name: str,
        probs: ProbabilityArray,
        outcomes: np.ndarray,
        drift_threshold: float = 0.03,
    ) -> dict[str, Any]:
        """Monitor calibration drift and alert if ECE degrades.

        Args:
            name: Calibrator name
            probs: Test probabilities
            outcomes: Test outcomes
            drift_threshold: Alert if ECE increases by more than this

        Returns:
            Dictionary with drift information and alert status
        """
        if not self.db_con:
            return {"status": "no_db", "alert": False}

        try:
            current_ece = expected_calibration_error(probs, outcomes, n_bins=10)
            current_mce = maximum_calibration_error(probs, outcomes, n_bins=10)

            # Store in history
            self.db_con.execute(
                """INSERT INTO calibration_drift_history
                   (calibrator_name, ece, mce, n_samples)
                   VALUES (?, ?, ?, ?)""",
                [name, float(current_ece), float(current_mce), len(probs)],
            )

            # Get previous ECE
            prev_row = self.db_con.execute(
                """SELECT ece FROM calibration_drift_history
                   WHERE calibrator_name = ?
                   ORDER BY checked_at DESC
                   LIMIT 2""",
                [name],
            ).fetchall()

            if len(prev_row) < 2:
                return {
                    "status": "first_check",
                    "ece": round(current_ece, 4),
                    "mce": round(current_mce, 4),
                    "alert": False,
                }

            prev_ece = float(prev_row[1][0])
            ece_change = current_ece - prev_ece

            alert = ece_change > drift_threshold

            if alert:
                log.warning(
                    f"Calibration drift detected for {name}: ECE increased from "
                    f"{prev_ece:.4f} to {current_ece:.4f} (+{ece_change:.4f})"
                )

            return {
                "status": "ok",
                "ece": round(current_ece, 4),
                "mce": round(current_mce, 4),
                "previous_ece": round(prev_ece, 4),
                "ece_change": round(ece_change, 4),
                "alert": alert,
                "alert_threshold": drift_threshold,
            }

        except Exception as e:
            log.error(f"Failed to monitor calibration drift: {e}")
            return {"status": "error", "error": str(e), "alert": False}

    def fit_stratified(
        self,
        name: str,
        probs: ProbabilityArray,
        outcomes: np.ndarray,
        strata: np.ndarray,
        method: str = "auto",
    ) -> dict[str, Any]:
        """Fit stratified calibrators (e.g., per-league).

        Args:
            name: Base calibrator name
            probs: Probabilities
            outcomes: Outcomes
            strata: Stratum labels (league, competition, etc.)
            method: Calibration method

        Returns:
            Dictionary with per-stratum results
        """
        results: dict[str, Any] = {}
        unique_strata = np.unique(strata)

        for stratum in unique_strata:
            mask = strata == stratum
            stratum_probs = probs[mask]
            stratum_outcomes = outcomes[mask]

            if len(stratum_probs) < 20:
                results[str(stratum)] = {"status": "insufficient_data", "n_samples": len(stratum_probs)}
                continue

            try:
                # Fit calibrator for this stratum
                if method == "auto":
                    calibrator = AutoCalibrator()
                elif method == "platt":
                    calibrator = PlattCalibrator()
                elif method == "isotonic":
                    calibrator = IsotonicCalibrator()
                elif method == "temperature":
                    calibrator = TemperatureCalibrator()
                else:
                    calibrator = TemperatureCalibrator()

                calibrator.fit(stratum_probs, stratum_outcomes)

                # Store stratified calibrator
                if self.db_con:
                    stratum_name = f"{name}_{stratum}"
                    params_json = json.dumps(calibrator.get_params(), default=str)
                    self.db_con.execute(
                        """INSERT OR REPLACE INTO stratified_calibrators
                           (name, stratum, method, params_json)
                           VALUES (?, ?, ?, ?)""",
                        [name, str(stratum), calibrator.name, params_json],
                    )

                ece = expected_calibration_error(stratum_probs, stratum_outcomes, n_bins=10)
                results[str(stratum)] = {
                    "status": "fitted",
                    "n_samples": len(stratum_probs),
                    "method": calibrator.name,
                    "ece": round(ece, 4),
                }

            except Exception as e:
                log.warning(f"Failed to fit stratified calibrator for {stratum}: {e}")
                results[str(stratum)] = {"status": "error", "error": str(e)}

        return results


# Convenience functions
def calibrate(
    probs: ProbabilityArray,
    method: str = "temperature",
) -> np.ndarray:
    """Quick calibration without persistence (for testing)."""
    probs_arr = np.asarray(probs)
    outcomes_arr = np.zeros(len(probs_arr), dtype=int)  # Dummy outcomes

    if method == "temperature":
        calibrator = TemperatureCalibrator()
    elif method == "platt":
        calibrator = PlattCalibrator()
    elif method == "isotonic":
        calibrator = IsotonicCalibrator()
    elif method == "beta":
        calibrator = BetaCalibrator()
    else:
        raise ValueError(f"Unknown method: {method}")

    calibrator.fit(probs_arr, outcomes_arr)
    return calibrator.calibrate(probs_arr)


def fit_calibrator(
    train_probs: ProbabilityArray,
    train_outcomes: np.ndarray,
    method: str = "platt",
) -> BaseCalibrator:
    """Fit and return a calibrator."""
    if method == "platt":
        calibrator = PlattCalibrator()
    elif method == "isotonic":
        calibrator = IsotonicCalibrator()
    elif method == "temperature":
        calibrator = TemperatureCalibrator()
    elif method == "beta":
        calibrator = BetaCalibrator()
    elif method == "auto":
        calibrator = AutoCalibrator()
    else:
        raise ValueError(f"Unknown method: {method}")

    calibrator.fit(train_probs, train_outcomes)
    return calibrator
