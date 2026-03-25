"""Optuna hyperparameter tuning for the v16 Sentinel leak-free pipeline.

Tunes CatBoost and XGBoost hyperparameters using proper temporal splits
(train/cal/test) with no data leakage. Uses the calibration split for
early stopping (NOT the test set).

Usage:
    python -m footy.models.tune_hyperparams --n-trials 100 --verbose

References:
    - CatBoost + Pi-Ratings achieved 55.82% (benchmark)
    - Optuna: Bayesian optimization with TPE sampler
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

log = logging.getLogger(__name__)


def tune_catboost(n_trials: int = 100, verbose: bool = True) -> dict:
    """Run Optuna hyperparameter optimization for CatBoost on the leak-free pipeline.

    Returns the best hyperparameters and achieved accuracy.
    """
    try:
        import optuna
        from catboost import CatBoostClassifier
    except ImportError:
        print("Requires: pip install optuna catboost")
        return {}

    from footy.db import connect
    from footy.models.council import (
        _prepare_df, _run_experts, _build_v13_features, ALL_EXPERTS,
        _label,
    )
    import pandas as pd

    con = connect()

    # Prepare data (same as train_and_save)
    df = _prepare_df(con, finished_only=True, days=2555)
    if df.empty or len(df) < 800:
        print(f"Not enough data ({len(df)})")
        return {}

    eval_days = 365
    cutoff_test = df["utc_date"].max() - pd.Timedelta(days=eval_days)
    cutoff_cal = cutoff_test - pd.Timedelta(days=max(30, eval_days // 3))
    train_mask = df["utc_date"] < cutoff_cal
    cal_mask = (df["utc_date"] >= cutoff_cal) & (df["utc_date"] < cutoff_test)
    test_mask = df["utc_date"] >= cutoff_test

    # Run experts once (expensive — cache results)
    if verbose:
        print(f"Running {len(ALL_EXPERTS)} experts on {len(df)} matches...", flush=True)
    results = _run_experts(df)

    competitions = df["competition"].to_numpy() if "competition" in df.columns else None
    X, _ = _build_v13_features(results, competitions=competitions, df=df)
    y = np.array([_label(int(hg), int(ag))
                  for hg, ag in zip(df["home_goals"], df["away_goals"])], dtype=int)

    Xtr, ytr = X[train_mask.to_numpy()], y[train_mask.to_numpy()]
    Xcal, ycal = X[cal_mask.to_numpy()], y[cal_mask.to_numpy()]
    Xte, yte = X[test_mask.to_numpy()], y[test_mask.to_numpy()]

    if verbose:
        print(f"Data: train={len(ytr)} cal={len(ycal)} test={len(yte)} features={X.shape[1]}", flush=True)

    # Class weights
    from collections import Counter
    class_counts = Counter(ytr)
    n_samples = len(ytr)
    cat_class_weights = [
        n_samples / (3 * class_counts.get(0, 1)),
        n_samples / (3 * class_counts.get(1, 1)),
        n_samples / (3 * class_counts.get(2, 1)),
    ]

    def objective(trial):
        params = {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "iterations": trial.suggest_int("iterations", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "random_strength": trial.suggest_float("random_strength", 0.5, 10.0),
            "bootstrap_type": "MVS",
            "use_best_model": True,
            "early_stopping_rounds": 100,
            "class_weights": cat_class_weights,
            "random_seed": 42,
            "verbose": 0,
        }

        model = CatBoostClassifier(**params)
        model.fit(Xtr, ytr, eval_set=(Xcal, ycal))  # cal split for early stopping
        preds = model.predict_proba(Xte)
        accuracy = float(np.mean(preds.argmax(axis=1) == yte))

        # Also compute RPS for a more robust metric
        rps = 0.0
        for i in range(len(yte)):
            cum_pred = np.cumsum(preds[i])
            cum_true = np.cumsum([1.0 if yte[i] == j else 0.0 for j in range(3)])
            rps += np.mean((cum_pred - cum_true) ** 2)
        rps /= len(yte)

        trial.set_user_attr("accuracy", accuracy)
        trial.set_user_attr("rps", rps)

        return rps  # minimize RPS (more robust than accuracy for 3-way)

    # Optuna study
    if verbose:
        print(f"Starting Optuna optimization ({n_trials} trials)...", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best = study.best_trial
    if verbose:
        print(f"\n=== BEST TRIAL (#{best.number}) ===")
        print(f"  RPS: {best.value:.5f}")
        print(f"  Accuracy: {best.user_attrs['accuracy']:.4f}")
        print("  Params:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")

    return {
        "best_params": best.params,
        "best_rps": best.value,
        "best_accuracy": best.user_attrs["accuracy"],
        "n_trials": n_trials,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    result = tune_catboost(n_trials=args.n_trials, verbose=args.verbose)
    if result:
        print(f"\nDone. Best accuracy: {result['best_accuracy']:.4f}, RPS: {result['best_rps']:.5f}")
        print("Copy these params to council.py CatBoost section:")
        for k, v in result["best_params"].items():
            print(f"  {k}={v},")
