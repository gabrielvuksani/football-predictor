"""Ordinal Regression (Proportional Odds) model for goal differences.

Models goal difference as an ordered outcome: ..., -2, -1, 0, +1, +2, ...
Derives 1X2 probabilities from the cumulative goal difference distribution.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

log = logging.getLogger(__name__)


def _ordinal_nll(params: np.ndarray, X: np.ndarray, y: np.ndarray,
                 n_classes: int) -> float:
    """Negative log-likelihood for ordinal regression (proportional odds model).

    Parameters
    ----------
    params : 1-D array of (n_classes - 1) thresholds + n_features coefficients
    X : (n_samples, n_features) feature matrix
    y : (n_samples,) integer class labels 0..n_classes-1
    n_classes : number of ordered categories

    Returns
    -------
    Negative log-likelihood (scalar).
    """
    n_thresholds = n_classes - 1
    thresholds = params[:n_thresholds]
    beta = params[n_thresholds:]

    # Ensure thresholds are ordered
    thresholds = np.sort(thresholds)

    eta = X @ beta  # (n_samples,)

    nll = 0.0
    for k in range(n_classes):
        mask = y == k
        if not mask.any():
            continue
        eta_k = eta[mask]
        if k == 0:
            p = expit(thresholds[0] - eta_k)
        elif k == n_classes - 1:
            p = 1.0 - expit(thresholds[-1] - eta_k)
        else:
            p = expit(thresholds[k] - eta_k) - expit(thresholds[k - 1] - eta_k)
        p = np.clip(p, 1e-12, 1 - 1e-12)  # avoid log(0)
        nll -= np.sum(np.log(p))

    # L2 regularization on beta
    nll += 0.01 * np.sum(beta ** 2)
    return float(nll)


class OrdinalModel:
    """Proportional odds ordinal regression for goal difference prediction."""

    def __init__(self, gd_range: int = 4):
        """
        Parameters
        ----------
        gd_range : int
            Maximum absolute goal difference to model.
            E.g. gd_range=4 models GD in [-4, -3, -2, -1, 0, 1, 2, 3, 4].
        """
        self.gd_range = gd_range
        self.n_classes = 2 * gd_range + 1  # e.g., 9 classes for gd_range=4
        self.thresholds: np.ndarray | None = None
        self.beta: np.ndarray | None = None
        self._fitted = False

    def _gd_to_class(self, gd: int) -> int:
        """Map goal difference to class index."""
        return min(max(gd + self.gd_range, 0), self.n_classes - 1)

    def _class_to_gd(self, cls: int) -> int:
        """Map class index back to goal difference."""
        return cls - self.gd_range

    def fit(self, X: np.ndarray, goal_diffs: np.ndarray) -> dict[str, Any]:
        """Fit ordinal regression model.

        Parameters
        ----------
        X : (n_samples, n_features) feature matrix
        goal_diffs : (n_samples,) integer goal differences (home - away)

        Returns
        -------
        Dict with fitting statistics.
        """
        y = np.array([self._gd_to_class(int(gd)) for gd in goal_diffs])
        n_features = X.shape[1]

        # Initial parameters: evenly spaced thresholds + zero coefficients
        init_thresholds = np.linspace(-2, 2, self.n_classes - 1)
        init_beta = np.zeros(n_features)
        x0 = np.concatenate([init_thresholds, init_beta])

        result = minimize(
            _ordinal_nll,
            x0,
            args=(X, y, self.n_classes),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-8},
        )

        n_thresholds = self.n_classes - 1
        self.thresholds = np.sort(result.x[:n_thresholds])
        self.beta = result.x[n_thresholds:]
        self._fitted = True

        return {
            "converged": result.success,
            "nll": float(result.fun),
            "n_samples": len(y),
            "n_features": n_features,
            "n_classes": self.n_classes,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for each goal difference.

        Returns
        -------
        (n_samples, n_classes) probability matrix.
        """
        if not self._fitted or self.thresholds is None or self.beta is None:
            n = X.shape[0]
            return np.full((n, self.n_classes), 1.0 / self.n_classes)

        eta = X @ self.beta
        probs = np.zeros((len(eta), self.n_classes))

        for k in range(self.n_classes):
            if k == 0:
                probs[:, k] = expit(self.thresholds[0] - eta)
            elif k == self.n_classes - 1:
                probs[:, k] = 1.0 - expit(self.thresholds[-1] - eta)
            else:
                probs[:, k] = expit(self.thresholds[k] - eta) - expit(self.thresholds[k - 1] - eta)

        # Clip and normalize
        probs = np.clip(probs, 0, 1)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = np.where(row_sums > 0, probs / row_sums, 1.0 / self.n_classes)

        return probs

    def predict_1x2(self, X: np.ndarray) -> np.ndarray:
        """Predict 1X2 probabilities from goal difference distribution.

        Returns
        -------
        (n_samples, 3) array of [P(Home), P(Draw), P(Away)].
        """
        gd_probs = self.predict_proba(X)

        home_idx = [i for i in range(self.n_classes) if self._class_to_gd(i) > 0]
        draw_idx = [i for i in range(self.n_classes) if self._class_to_gd(i) == 0]
        away_idx = [i for i in range(self.n_classes) if self._class_to_gd(i) < 0]

        p_home = gd_probs[:, home_idx].sum(axis=1)
        p_draw = gd_probs[:, draw_idx].sum(axis=1)
        p_away = gd_probs[:, away_idx].sum(axis=1)

        result = np.column_stack([p_home, p_draw, p_away])
        # Normalize
        row_sums = result.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0, result / row_sums, 1.0 / 3.0)

    def predict_expected_goals(self, X: np.ndarray, base_home: float = 1.5,
                                base_away: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
        """Estimate expected goals from predicted goal difference distribution.

        Uses the mean predicted GD plus base league scoring rates.
        """
        gd_probs = self.predict_proba(X)
        gd_values = np.array([self._class_to_gd(i) for i in range(self.n_classes)])
        mean_gd = (gd_probs * gd_values).sum(axis=1)

        # Distribute expected goals: home gets base + half of GD, away gets base - half
        eg_home = np.maximum(0.1, base_home + mean_gd * 0.5)
        eg_away = np.maximum(0.1, base_away - mean_gd * 0.5)

        return eg_home, eg_away
