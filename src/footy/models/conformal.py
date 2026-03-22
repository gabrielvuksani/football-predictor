"""Conformal prediction for football match outcomes.

Provides distribution-free prediction sets with guaranteed marginal coverage.
Uses split conformal prediction (Vovk et al., 2005) adapted for 3-class
football outcomes.

The key guarantee: if the calibration data and future test data are
exchangeable, the prediction sets produced by ``ConformalPredictor`` will
contain the true outcome with probability >= 1 - alpha, regardless of the
underlying model's accuracy.

Typical workflow::

    from footy.models.conformal import ConformalPredictor

    cp = ConformalPredictor(alpha=0.10)           # 90 % coverage
    cp.calibrate(cal_probs, cal_labels)           # holdout calibration
    sets  = cp.predict_set(test_probs)            # prediction sets
    intervals = cp.prediction_interval(test_probs)  # probability intervals

References:
    Vovk et al. (2005) "Algorithmic Learning in a Random World"
    Romano et al. (2020) "Classification with Valid and Adaptive Coverage"
    Sadinle et al. (2019) "Least Ambiguous Set-Valued Classifiers"
"""
from __future__ import annotations

import numpy as np
import logging

log = logging.getLogger(__name__)

# Outcome labels used throughout the predictor module.
OUTCOME_HOME = 0
OUTCOME_DRAW = 1
OUTCOME_AWAY = 2
OUTCOME_LABELS = {OUTCOME_HOME: "Home", OUTCOME_DRAW: "Draw", OUTCOME_AWAY: "Away"}


class ConformalPredictor:
    """Split conformal predictor for 1X2 outcomes.

    Uses the *Adaptive Prediction Sets* (APS) nonconformity score
    ``s(x, y) = 1 - p_model(y | x)`` which yields the smallest prediction
    sets among common conformal methods for classification problems.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        """
        Args:
            alpha: Miscoverage rate.  ``alpha=0.1`` means a 90 % coverage
                guarantee.  Must be in (0, 1).

        Raises:
            ValueError: If alpha is outside (0, 1).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.calibration_scores: np.ndarray | None = None
        self._qhat: float | None = None

    # ── Calibration ────────────────────────────────────────────────────

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """Calibrate using holdout data.

        Computes nonconformity scores on the calibration set and derives
        the conformal quantile ``q_hat`` used for prediction-set
        construction.

        Args:
            probs: ``(n, 3)`` predicted probabilities
                ``[p_home, p_draw, p_away]``.  Each row must sum to ~1.
            labels: ``(n,)`` true outcomes ``{0=home, 1=draw, 2=away}``.

        Raises:
            ValueError: If shapes are inconsistent or values invalid.
        """
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.intp)

        if probs.ndim != 2 or probs.shape[1] != 3:
            raise ValueError(
                f"probs must have shape (n, 3), got {probs.shape}"
            )
        if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
            raise ValueError(
                f"labels length ({labels.shape[0]}) must match "
                f"probs rows ({probs.shape[0]})"
            )
        if not np.all((labels >= 0) & (labels <= 2)):
            raise ValueError("labels must be in {0, 1, 2}")

        n = len(labels)

        # Nonconformity score: 1 - p(true class)
        true_class_probs = probs[np.arange(n), labels]
        self.calibration_scores = 1.0 - true_class_probs

        # Conformal quantile with finite-sample correction.
        # q_hat = ceil((n+1)(1-alpha)) / n  -th quantile of scores.
        quantile_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        quantile_level = min(quantile_level, 1.0)  # clamp for safety
        self._qhat = float(np.quantile(self.calibration_scores, quantile_level))

        log.info(
            "Calibrated on %d samples.  q_hat=%.4f (alpha=%.2f)",
            n,
            self._qhat,
            self.alpha,
        )

    # ── Prediction sets ────────────────────────────────────────────────

    def predict_set(self, probs: np.ndarray) -> list[list[int]]:
        """Return prediction sets with guaranteed coverage.

        A class ``k`` is included in the prediction set for sample ``i``
        if its nonconformity score ``1 - p(k|x_i)`` does not exceed the
        conformal quantile ``q_hat``.  Equivalently, class ``k`` is
        included when ``p(k|x_i) >= 1 - q_hat``.

        Args:
            probs: ``(n, 3)`` predicted probabilities.

        Returns:
            List of prediction sets.  Each element is a sorted list of
            outcome indices that cannot be rejected at level ``alpha``.
            For example ``[[0], [0, 1], [0, 2]]``.

        Raises:
            RuntimeError: If :meth:`calibrate` has not been called.
            ValueError: If probs has wrong shape.
        """
        self._check_calibrated()
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] != 3:
            raise ValueError(
                f"probs must have shape (n, 3), got {probs.shape}"
            )

        threshold = 1.0 - self._qhat
        prediction_sets: list[list[int]] = []

        for row in probs:
            pset = sorted(
                k for k in range(3) if row[k] >= threshold
            )
            # A conformal predictor must never return the empty set.
            # If all probabilities are below the threshold (rare, only
            # possible with very poor models), include the argmax.
            if not pset:
                pset = [int(np.argmax(row))]
            prediction_sets.append(pset)

        return prediction_sets

    # ── Prediction intervals ───────────────────────────────────────────

    def prediction_interval(
        self, probs: np.ndarray
    ) -> list[tuple[float, float]]:
        """Return prediction intervals for the most likely outcome probability.

        For each sample the interval ``[lower, upper]`` bounds the
        probability of the predicted (argmax) class.  The bounds are
        derived from the conformal quantile:

        * ``lower = max(p_max - q_hat, 0)``
        * ``upper = min(p_max + q_hat, 1)``

        These intervals reflect the model's uncertainty as measured by
        the calibration residuals and are *not* exact coverage intervals
        for the probability itself, but rather a practical summary of
        conformal uncertainty.

        Args:
            probs: ``(n, 3)`` predicted probabilities.

        Returns:
            List of ``(lower, upper)`` tuples, one per sample.

        Raises:
            RuntimeError: If :meth:`calibrate` has not been called.
        """
        self._check_calibrated()
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] != 3:
            raise ValueError(
                f"probs must have shape (n, 3), got {probs.shape}"
            )

        p_max = np.max(probs, axis=1)
        lower = np.clip(p_max - self._qhat, 0.0, 1.0)
        upper = np.clip(p_max + self._qhat, 0.0, 1.0)

        return list(zip(lower.tolist(), upper.tolist()))

    # ── Diagnostics ────────────────────────────────────────────────────

    def empirical_coverage(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> float:
        """Compute empirical coverage on a held-out evaluation set.

        Args:
            probs: ``(n, 3)`` predicted probabilities.
            labels: ``(n,)`` true outcomes.

        Returns:
            Fraction of samples whose true label is inside the
            prediction set.
        """
        labels = np.asarray(labels, dtype=np.intp)
        sets = self.predict_set(probs)
        hits = sum(
            1 for pset, y in zip(sets, labels) if y in pset
        )
        return hits / len(labels) if len(labels) > 0 else 0.0

    def average_set_size(self, probs: np.ndarray) -> float:
        """Mean prediction-set size (lower is more informative).

        Args:
            probs: ``(n, 3)`` predicted probabilities.

        Returns:
            Average number of classes in the prediction sets.
        """
        sets = self.predict_set(probs)
        return sum(len(s) for s in sets) / len(sets) if sets else 0.0

    # ── Internals ──────────────────────────────────────────────────────

    def _check_calibrated(self) -> None:
        if self._qhat is None:
            raise RuntimeError(
                "ConformalPredictor has not been calibrated.  "
                "Call calibrate() first."
            )

    def __repr__(self) -> str:
        cal = (
            f"q_hat={self._qhat:.4f}"
            if self._qhat is not None
            else "uncalibrated"
        )
        return f"ConformalPredictor(alpha={self.alpha}, {cal})"
