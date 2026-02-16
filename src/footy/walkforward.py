"""
Walk-forward cross-validation for time-series football prediction models.

Unlike standard k-fold CV, walk-forward respects temporal ordering:
    Fold 1: Train on months 1-12,  Test on month 13
    Fold 2: Train on months 1-13,  Test on month 14
    Fold 3: Train on months 1-14,  Test on month 15
    ...

This gives **unbiased** performance estimates since the model never sees
future data during training.

Key features:
    - Expanding window (train grows each fold) or sliding window
    - Per-fold and aggregate metrics (logloss, brier, accuracy, ECE)
    - Fold-level feature importance tracking
    - Calibration analysis across folds
    - Automatic hyperparameter stability analysis
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

log = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold: int
    train_size: int
    test_size: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    logloss: float
    brier: float
    accuracy: float
    ece: float
    # Optional detailed metrics
    btts_accuracy: float | None = None
    ou25_accuracy: float | None = None
    feature_importances: dict[str, float] = field(default_factory=dict)
    predictions: np.ndarray | None = None  # (n_test, 3)
    actuals: np.ndarray | None = None  # (n_test,)


@dataclass
class WalkForwardResult:
    """Aggregate results from walk-forward cross-validation."""
    n_folds: int
    total_test_samples: int
    # Aggregate metrics (weighted by fold size)
    mean_logloss: float
    mean_brier: float
    mean_accuracy: float
    mean_ece: float
    # Stability metrics
    std_logloss: float
    std_accuracy: float
    # Per-fold results
    folds: list[FoldResult]
    # Combined predictions (all OOS predictions)
    all_predictions: np.ndarray | None = None
    all_actuals: np.ndarray | None = None
    # Feature importance (averaged across folds)
    feature_importances: dict[str, float] = field(default_factory=dict)


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if mask.any():
            ece += abs(conf[mask].mean() - correct[mask].mean()) * mask.mean()
    return float(ece)


def walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray | pd.Series,
    *,
    n_folds: int = 5,
    min_train_frac: float = 0.4,
    test_months: int = 3,
    expanding: bool = True,
    max_train_months: int | None = None,
    model_params: dict[str, Any] | None = None,
    calibrate: bool = True,
    feature_names: list[str] | None = None,
    y_btts: np.ndarray | None = None,
    y_ou25: np.ndarray | None = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Perform walk-forward cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,) with values 0, 1, 2
        dates: Datetime array aligned with X/y
        n_folds: Number of evaluation folds
        min_train_frac: Minimum fraction of data used for first training set
        test_months: Months per test fold
        expanding: If True, training window expands; if False, slides
        max_train_months: If set with expanding=False, max training window size
        model_params: Override HistGradientBoosting hyperparameters
        calibrate: Whether to apply isotonic calibration
        feature_names: Optional feature names for importance tracking
        y_btts: Optional BTTS labels for auxiliary evaluation
        y_ou25: Optional Over 2.5 labels for auxiliary evaluation
        verbose: Print progress

    Returns:
        WalkForwardResult with per-fold and aggregate metrics
    """
    dates = pd.to_datetime(dates)
    n = len(X)

    # Compute fold boundaries
    total_range = (dates.max() - dates.min()).days
    test_days = test_months * 30
    min_train_days = int(total_range * min_train_frac)

    # Generate fold cutoffs
    fold_cutoffs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    end_date = dates.max()
    cursor = end_date - pd.Timedelta(days=test_days * n_folds)

    # Ensure minimum training data
    if (cursor - dates.min()).days < min_train_days:
        cursor = dates.min() + pd.Timedelta(days=min_train_days)

    for fold in range(n_folds):
        test_start = cursor
        test_end = cursor + pd.Timedelta(days=test_days)
        if test_end > end_date:
            test_end = end_date
        fold_cutoffs.append((test_start, test_end))
        cursor = test_end

    if not fold_cutoffs:
        raise ValueError(f"Not enough data for {n_folds} folds")

    # Default model parameters
    params = {
        "learning_rate": 0.02,
        "max_depth": 5,
        "max_iter": 1800,
        "l2_regularization": 0.5,
        "min_samples_leaf": 50,
        "max_bins": 255,
        "max_leaf_nodes": 31,
        "early_stopping": True,
        "validation_fraction": 0.12,
        "n_iter_no_change": 30,
        "random_state": 42,
    }
    if model_params:
        params.update(model_params)

    fold_results: list[FoldResult] = []
    all_preds_list: list[np.ndarray] = []
    all_actuals_list: list[np.ndarray] = []
    eps = 1e-12

    for fold_i, (test_start, test_end) in enumerate(fold_cutoffs):
        # Define train/test masks
        test_mask = (dates >= test_start) & (dates < test_end)
        if expanding:
            train_mask = dates < test_start
        else:
            if max_train_months:
                train_start = test_start - pd.Timedelta(days=max_train_months * 30)
                train_mask = (dates >= train_start) & (dates < test_start)
            else:
                train_mask = dates < test_start

        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte = X[test_mask], y[test_mask]

        if len(Xtr) < 200 or len(Xte) < 20:
            if verbose:
                log.info("Fold %d: skipped (train=%d, test=%d)", fold_i + 1, len(Xtr), len(Xte))
            continue

        if verbose:
            log.info(
                "Fold %d/%d: train=%d (%s→%s) test=%d (%s→%s)",
                fold_i + 1, n_folds, len(Xtr),
                dates[train_mask].min().strftime("%Y-%m-%d"),
                dates[train_mask].max().strftime("%Y-%m-%d"),
                len(Xte),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            )

        # Train model
        base = HistGradientBoostingClassifier(**params)
        if calibrate:
            cv_folds = min(5, max(2, len(Xtr) // 200))
            model = CalibratedClassifierCV(base, method="isotonic", cv=cv_folds)
        else:
            model = base

        try:
            model.fit(Xtr, ytr)
        except Exception as e:
            log.warning("Fold %d: training failed: %s", fold_i + 1, e)
            continue

        P = model.predict_proba(Xte)

        # Metrics
        logloss = float(np.mean(-np.log(P[np.arange(len(yte)), yte] + eps)))
        Y_onehot = np.zeros_like(P)
        Y_onehot[np.arange(len(yte)), yte] = 1.0
        brier = float(np.mean(np.sum((P - Y_onehot) ** 2, axis=1) / 3))
        accuracy = float(np.mean(P.argmax(axis=1) == yte))
        ece = _compute_ece(P, yte)

        # BTTS / O2.5 accuracy if labels provided
        btts_acc = None
        ou25_acc = None
        if y_btts is not None:
            yte_btts = y_btts[test_mask]
            try:
                btts_base = HistGradientBoostingClassifier(
                    learning_rate=0.03, max_depth=4, max_iter=800,
                    l2_regularization=0.5, min_samples_leaf=80,
                    early_stopping=True, validation_fraction=0.12,
                    n_iter_no_change=20, random_state=42,
                )
                btts_model = CalibratedClassifierCV(btts_base, method="isotonic", cv=min(5, max(2, len(Xtr) // 200)))
                btts_model.fit(Xtr, y_btts[train_mask])
                P_btts = btts_model.predict_proba(Xte)
                btts_acc = float(np.mean((P_btts[:, 1] >= 0.5) == yte_btts))
            except Exception:
                pass

        if y_ou25 is not None:
            yte_ou25 = y_ou25[test_mask]
            try:
                ou25_base = HistGradientBoostingClassifier(
                    learning_rate=0.03, max_depth=4, max_iter=800,
                    l2_regularization=0.5, min_samples_leaf=80,
                    early_stopping=True, validation_fraction=0.12,
                    n_iter_no_change=20, random_state=42,
                )
                ou25_model = CalibratedClassifierCV(ou25_base, method="isotonic", cv=min(5, max(2, len(Xtr) // 200)))
                ou25_model.fit(Xtr, y_ou25[train_mask])
                P_ou25 = ou25_model.predict_proba(Xte)
                ou25_acc = float(np.mean((P_ou25[:, 1] >= 0.5) == yte_ou25))
            except Exception:
                pass

        # Feature importance (from base estimator)
        feat_imp: dict[str, float] = {}
        try:
            if hasattr(model, "estimators_"):
                # CalibratedClassifierCV wraps estimators
                importances = np.mean([
                    est.estimator.feature_importances_
                    for est in model.calibrated_classifiers_
                    if hasattr(est, "estimator") and hasattr(est.estimator, "feature_importances_")
                ], axis=0)
            elif hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                importances = None

            if importances is not None and feature_names:
                for fname, imp in zip(feature_names, importances):
                    feat_imp[fname] = float(imp)
        except Exception:
            pass

        fold_result = FoldResult(
            fold=fold_i + 1,
            train_size=len(Xtr),
            test_size=len(Xte),
            train_start=dates[train_mask].min().strftime("%Y-%m-%d"),
            train_end=dates[train_mask].max().strftime("%Y-%m-%d"),
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
            logloss=logloss,
            brier=brier,
            accuracy=accuracy,
            ece=ece,
            btts_accuracy=btts_acc,
            ou25_accuracy=ou25_acc,
            feature_importances=feat_imp,
            predictions=P,
            actuals=yte,
        )
        fold_results.append(fold_result)
        all_preds_list.append(P)
        all_actuals_list.append(yte)

        if verbose:
            log.info(
                "  Fold %d: logloss=%.4f brier=%.4f accuracy=%.3f ece=%.4f",
                fold_i + 1, logloss, brier, accuracy, ece,
            )

    if not fold_results:
        raise ValueError("No valid folds completed")

    # Aggregate
    weights = np.array([f.test_size for f in fold_results], dtype=float)
    weights /= weights.sum()

    mean_ll = float(np.average([f.logloss for f in fold_results], weights=weights))
    mean_br = float(np.average([f.brier for f in fold_results], weights=weights))
    mean_acc = float(np.average([f.accuracy for f in fold_results], weights=weights))
    mean_ece = float(np.average([f.ece for f in fold_results], weights=weights))
    std_ll = float(np.std([f.logloss for f in fold_results]))
    std_acc = float(np.std([f.accuracy for f in fold_results]))

    # Combine all predictions
    all_preds = np.vstack(all_preds_list) if all_preds_list else None
    all_actuals = np.concatenate(all_actuals_list) if all_actuals_list else None
    total_test = sum(f.test_size for f in fold_results)

    # Average feature importance across folds
    avg_importance: dict[str, float] = {}
    if feature_names:
        for fname in feature_names:
            vals = [f.feature_importances.get(fname, 0.0) for f in fold_results]
            avg_importance[fname] = float(np.mean(vals))
        # Sort by importance
        avg_importance = dict(sorted(avg_importance.items(), key=lambda x: -x[1]))

    result = WalkForwardResult(
        n_folds=len(fold_results),
        total_test_samples=total_test,
        mean_logloss=mean_ll,
        mean_brier=mean_br,
        mean_accuracy=mean_acc,
        mean_ece=mean_ece,
        std_logloss=std_ll,
        std_accuracy=std_acc,
        folds=fold_results,
        all_predictions=all_preds,
        all_actuals=all_actuals,
        feature_importances=avg_importance,
    )

    if verbose:
        log.info(
            "Walk-forward CV (%d folds, %d total test):\n"
            "  logloss: %.4f ± %.4f\n"
            "  brier:   %.4f\n"
            "  accuracy: %.3f ± %.3f\n"
            "  ECE:     %.4f",
            result.n_folds, result.total_test_samples,
            mean_ll, std_ll, mean_br, mean_acc, std_acc, mean_ece,
        )
        if avg_importance:
            top_10 = list(avg_importance.items())[:10]
            log.info("  Top 10 features: %s",
                     ", ".join(f"{k}={v:.4f}" for k, v in top_10))

    return result


def analyze_calibration(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """
    Compute reliability diagram data.

    Returns list of dicts with:
        bin_lower, bin_upper, avg_confidence, avg_accuracy, count, fraction
    """
    conf = predictions.max(axis=1)
    pred = predictions.argmax(axis=1)
    correct = (pred == actuals).astype(float)

    bins_edges = np.linspace(0, 1, n_bins + 1)
    calibration: list[dict] = []

    for i in range(n_bins):
        mask = (conf >= bins_edges[i]) & (conf < bins_edges[i + 1])
        if mask.any():
            calibration.append({
                "bin_lower": float(bins_edges[i]),
                "bin_upper": float(bins_edges[i + 1]),
                "avg_confidence": float(conf[mask].mean()),
                "avg_accuracy": float(correct[mask].mean()),
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
            })

    return calibration


def feature_stability_report(
    wf_result: WalkForwardResult,
    top_n: int = 20,
) -> list[dict]:
    """
    Analyze feature importance stability across folds.

    Returns list of features with:
        name, mean_importance, std_importance, cv (coefficient of variation),
        stable (True if CV < 0.5)
    """
    if not wf_result.feature_importances:
        return []

    top_features = list(wf_result.feature_importances.keys())[:top_n]
    report: list[dict] = []

    for fname in top_features:
        vals = [f.feature_importances.get(fname, 0.0) for f in wf_result.folds]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        cv = std_val / max(mean_val, 1e-12)

        report.append({
            "name": fname,
            "mean_importance": round(mean_val, 6),
            "std_importance": round(std_val, 6),
            "cv": round(cv, 3),
            "stable": cv < 0.5,
        })

    return report
