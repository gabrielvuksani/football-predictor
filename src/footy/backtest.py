"""Historical backtest framework for football prediction models.

Runs the full prediction pipeline on historical data with proper temporal
separation to measure true out-of-sample performance.

Usage:
    from footy.backtest import Backtester
    bt = Backtester(con)
    results = bt.run(seasons=3, verbose=True)
    bt.print_report(results)

    # Compare two model configurations:
    r_a = bt.run(seasons=3, use_catboost=True)
    r_b = bt.run(seasons=3, use_catboost=False)
    diff = bt.compare_models(r_a, r_b)
    bt.print_comparison(diff)
"""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import HistGradientBoostingClassifier

from footy.models.math.scoring import (
    brier_score,
    expected_calibration_error,
    log_loss as single_log_loss,
    ranked_probability_score,
    remove_overround,
)
from footy.models.experts import ALL_EXPERTS, ExpertResult
from footy.models.experts._base import _label
from footy.normalize import canonical_team_name

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Results from a single expanding-window fold."""
    window_idx: int
    train_size: int
    test_size: int
    train_end: str
    test_start: str
    test_end: str
    rps: float
    log_loss: float
    brier: float
    accuracy: float
    ece: float
    predictions: np.ndarray      # (n_test, 3)
    actuals: np.ndarray          # (n_test,) outcome indices
    market_probs: np.ndarray     # (n_test, 3) market implied probs
    competitions: np.ndarray     # (n_test,) competition strings
    feature_importances: dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from a historical backtest run."""
    total_matches: int = 0
    accuracy: float = 0.0
    rps: float = 0.0
    log_loss: float = 0.0
    brier: float = 0.0
    ece: float = 0.0

    # Per-competition breakdown
    by_competition: dict = field(default_factory=dict)

    # Per-period breakdown (monthly)
    by_period: list = field(default_factory=list)

    # Calibration data (for reliability diagrams)
    calibration_bins: list = field(default_factory=list)

    # Upset detection performance
    upset_recall: float = 0.0
    upset_precision: float = 0.0
    upset_f1: float = 0.0

    # Comparison against market odds
    vs_market_rps: float = 0.0  # our RPS minus market RPS (negative = we're better)

    # Feature importance (averaged across windows)
    feature_importance: dict = field(default_factory=dict)

    # Per-window breakdown for stability analysis
    windows: list = field(default_factory=list)

    # Raw arrays for downstream analysis
    all_predictions: Optional[np.ndarray] = None
    all_actuals: Optional[np.ndarray] = None
    all_market_probs: Optional[np.ndarray] = None
    all_competitions: Optional[np.ndarray] = None
    all_dates: Optional[np.ndarray] = None

    # Timing
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Run historical backtests on the prediction pipeline.

    This class orchestrates expanding-window evaluation that mirrors
    how the model would perform in production: for each test window,
    only data strictly before the window is available for training.
    """

    def __init__(self, con):
        """
        Args:
            con: DuckDB connection with the footy database.
        """
        self.con = con

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        seasons: int = 3,
        min_train_matches: int = 2000,
        test_window_days: int = 30,
        embargo_days: int = 3,
        use_catboost: bool = True,
        verbose: bool = True,
    ) -> BacktestResult:
        """Run expanding-window backtest over historical data.

        For each test window:
            1. Train model on all data BEFORE the window (minus embargo gap)
            2. Generate predictions for matches in the window
            3. Compare to actual results
            4. Move window forward by test_window_days

        Args:
            seasons: Number of recent seasons to backtest over.
            min_train_matches: Minimum training set size before first prediction.
            test_window_days: Width of each test window in days.
            embargo_days: Gap between train end and test start to prevent
                same-matchday leakage.
            use_catboost: Whether to try CatBoost as primary (falls back to
                HistGBM if unavailable).
            verbose: Print progress to stdout.

        Returns:
            BacktestResult with aggregate and per-window metrics.
        """
        t0 = time.time()

        # ----------------------------------------------------------
        # 1. Load all finished matches
        # ----------------------------------------------------------
        df = self._load_matches()
        if df.empty or len(df) < min_train_matches + 100:
            raise ValueError(
                f"Not enough data for backtest: {len(df)} matches "
                f"(need at least {min_train_matches + 100})"
            )

        df = df.sort_values("utc_date").reset_index(drop=True)
        dates = pd.to_datetime(df["utc_date"], utc=True)

        # Determine backtest range
        end_date = dates.max()
        start_date = end_date - pd.Timedelta(days=int(seasons * 365))
        # Ensure we have enough training data before start_date
        earliest_possible = dates.iloc[min_train_matches]
        if start_date < earliest_possible:
            start_date = earliest_possible
            if verbose:
                print(
                    f"[backtest] adjusted start to {start_date.strftime('%Y-%m-%d')} "
                    f"(need {min_train_matches} training matches)",
                    flush=True,
                )

        if verbose:
            print(
                f"[backtest] {len(df)} total matches | "
                f"backtest range: {start_date.strftime('%Y-%m-%d')} -> "
                f"{end_date.strftime('%Y-%m-%d')} | "
                f"window={test_window_days}d embargo={embargo_days}d",
                flush=True,
            )

        # ----------------------------------------------------------
        # 2. Generate window boundaries
        # ----------------------------------------------------------
        windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        cursor = start_date
        while cursor < end_date:
            w_end = cursor + pd.Timedelta(days=test_window_days)
            if w_end > end_date:
                w_end = end_date
            windows.append((cursor, w_end))
            cursor = w_end

        if verbose:
            print(f"[backtest] {len(windows)} test windows", flush=True)

        # ----------------------------------------------------------
        # 3. Run all experts ONCE on the full dataset
        # ----------------------------------------------------------
        # Experts compute sequential features (EMA, rolling stats) and
        # must see the full chronological history. We compute them once,
        # then slice the resulting feature matrices per window.
        if verbose:
            print(
                f"[backtest] running {len(ALL_EXPERTS)} experts on "
                f"{len(df)} matches...",
                flush=True,
            )

        results = self._run_experts(df)

        # Build Dixon-Coles models per competition (trained on full data
        # up to each window cutoff; DC is inexpensive so we refit per window).
        # For the single-pass expert features we fit DC on ALL finished data.
        # This is slightly optimistic for DC features; the meta-learner
        # training split below is what provides temporal separation.

        # Build labels
        y = np.array(
            [
                _label(int(hg), int(ag))
                for hg, ag in zip(df["home_goals"], df["away_goals"])
            ],
            dtype=int,
        )

        # Extract market implied probabilities for every match
        market_probs_full = self._extract_market_probs(df)

        # Extract competitions
        competitions_full = df["competition"].to_numpy() if "competition" in df.columns else np.full(len(df), "UNK")

        # v13 feature builder (imported lazily to avoid circular import)
        from footy.models.council import _build_v13_features

        # Build v13 feature matrix (all results, no DC appended here —
        # we skip the per-window DC refit for speed; the experts already
        # include Poisson-derived DC proxies).
        X, feature_names = _build_v13_features(results, competitions=competitions_full, df=df)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if verbose:
            print(
                f"[backtest] feature matrix: {X.shape[0]} x {X.shape[1]}",
                flush=True,
            )

        # ----------------------------------------------------------
        # 4. Expanding-window evaluation
        # ----------------------------------------------------------
        window_results: list[WindowResult] = []
        all_preds_list: list[np.ndarray] = []
        all_actuals_list: list[np.ndarray] = []
        all_market_list: list[np.ndarray] = []
        all_comp_list: list[np.ndarray] = []
        all_dates_list: list[np.ndarray] = []

        for w_idx, (w_start, w_end) in enumerate(windows):
            # temporal masks
            train_cutoff = w_start - pd.Timedelta(days=embargo_days)
            train_mask = (dates < train_cutoff).to_numpy()
            is_last = w_idx == len(windows) - 1
            if is_last:
                test_mask = ((dates >= w_start) & (dates <= w_end)).to_numpy()
            else:
                test_mask = ((dates >= w_start) & (dates < w_end)).to_numpy()

            n_train = int(train_mask.sum())
            n_test = int(test_mask.sum())

            if n_train < min_train_matches or n_test < 5:
                if verbose:
                    log.debug(
                        "Window %d skipped: train=%d test=%d",
                        w_idx + 1, n_train, n_test,
                    )
                continue

            Xtr, ytr = X[train_mask], y[train_mask]
            Xte, yte = X[test_mask], y[test_mask]
            mkt_te = market_probs_full[test_mask]
            comp_te = competitions_full[test_mask]
            dates_te = dates.to_numpy()[test_mask]

            # ---- Train model ----
            P = self._train_and_predict(
                Xtr, ytr, Xte, use_catboost=use_catboost
            )

            # Compute per-match scores
            rps_vals = np.array([
                ranked_probability_score(P[i], int(yte[i]))
                for i in range(len(yte))
            ])
            ll_vals = np.array([
                single_log_loss(P[i], int(yte[i]))
                for i in range(len(yte))
            ])
            brier_vals = np.array([
                brier_score(P[i], int(yte[i]))
                for i in range(len(yte))
            ])

            acc = float(np.mean(P.argmax(axis=1) == yte))
            ece = float(expected_calibration_error(P, yte, n_bins=10))

            # Feature importance from HistGBM (always trained as fallback)
            feat_imp = self._extract_importance(
                Xtr, ytr, feature_names
            )

            wr = WindowResult(
                window_idx=w_idx + 1,
                train_size=n_train,
                test_size=n_test,
                train_end=train_cutoff.strftime("%Y-%m-%d"),
                test_start=w_start.strftime("%Y-%m-%d"),
                test_end=w_end.strftime("%Y-%m-%d"),
                rps=float(np.mean(rps_vals)),
                log_loss=float(np.mean(ll_vals)),
                brier=float(np.mean(brier_vals)),
                accuracy=acc,
                ece=ece,
                predictions=P,
                actuals=yte,
                market_probs=mkt_te,
                competitions=comp_te,
                feature_importances=feat_imp,
            )
            window_results.append(wr)
            all_preds_list.append(P)
            all_actuals_list.append(yte)
            all_market_list.append(mkt_te)
            all_comp_list.append(comp_te)
            all_dates_list.append(dates_te)

            if verbose:
                print(
                    f"  Window {w_idx + 1:3d}/{len(windows)}: "
                    f"train={n_train:5d} test={n_test:4d} | "
                    f"RPS={wr.rps:.4f} LL={wr.log_loss:.4f} "
                    f"Acc={wr.accuracy:.3f}",
                    flush=True,
                )

        if not window_results:
            raise ValueError(
                "No valid windows completed. Check data availability "
                "and min_train_matches."
            )

        # ----------------------------------------------------------
        # 5. Aggregate results
        # ----------------------------------------------------------
        all_preds = np.vstack(all_preds_list)
        all_actuals = np.concatenate(all_actuals_list)
        all_market = np.vstack(all_market_list)
        all_comps = np.concatenate(all_comp_list)
        all_dates_arr = np.concatenate(all_dates_list)
        total = len(all_actuals)

        # Overall metrics (computed on pooled OOS predictions)
        rps_all = np.mean([
            ranked_probability_score(all_preds[i], int(all_actuals[i]))
            for i in range(total)
        ])
        ll_all = np.mean([
            single_log_loss(all_preds[i], int(all_actuals[i]))
            for i in range(total)
        ])
        brier_all = np.mean([
            brier_score(all_preds[i], int(all_actuals[i]))
            for i in range(total)
        ])
        acc_all = float(np.mean(all_preds.argmax(axis=1) == all_actuals))
        ece_all = float(expected_calibration_error(all_preds, all_actuals, n_bins=10))

        # Market comparison
        market_rps_all = np.mean([
            ranked_probability_score(all_market[i], int(all_actuals[i]))
            for i in range(total)
            if all_market[i].sum() > 0.5  # skip matches without odds
        ])
        n_with_odds = int(np.sum(all_market.sum(axis=1) > 0.5))

        # Per-competition breakdown
        by_comp = self._compute_by_competition(
            all_preds, all_actuals, all_market, all_comps
        )

        # Per-period (monthly) breakdown
        by_period = self._compute_by_period(
            all_preds, all_actuals, all_dates_arr
        )

        # Calibration bins
        cal_bins = self._compute_calibration(all_preds, all_actuals, n_bins=10)

        # Upset detection
        upset_metrics = self._compute_upset_metrics(
            all_preds, all_actuals, all_market
        )

        # Feature importance (average across windows)
        avg_importance = self._average_importance(window_results, feature_names)

        # SHAP if available
        shap_importance = self._compute_shap_importance_from_data(
            X, y, dates.to_numpy(), all_dates_arr, feature_names
        )
        if shap_importance:
            # Merge SHAP into importance dict (prefix with "shap_")
            for k, v in shap_importance.items():
                avg_importance[f"shap_{k}"] = v

        elapsed = time.time() - t0

        result = BacktestResult(
            total_matches=total,
            accuracy=float(acc_all),
            rps=float(rps_all),
            log_loss=float(ll_all),
            brier=float(brier_all),
            ece=float(ece_all),
            by_competition=by_comp,
            by_period=by_period,
            calibration_bins=cal_bins,
            upset_recall=upset_metrics.get("recall", 0.0),
            upset_precision=upset_metrics.get("precision", 0.0),
            upset_f1=upset_metrics.get("f1", 0.0),
            vs_market_rps=float(rps_all - market_rps_all) if n_with_odds > 0 else 0.0,
            feature_importance=avg_importance,
            windows=window_results,
            all_predictions=all_preds,
            all_actuals=all_actuals,
            all_market_probs=all_market,
            all_competitions=all_comps,
            all_dates=all_dates_arr,
            elapsed_seconds=elapsed,
        )

        if verbose:
            self.print_report(result)

        return result

    def compare_models(
        self,
        model_a_results: BacktestResult,
        model_b_results: BacktestResult,
    ) -> dict:
        """A/B comparison between two model versions.

        Uses the Diebold-Mariano test to determine if the difference
        in RPS is statistically significant, plus practical effect sizes.

        Args:
            model_a_results: BacktestResult from first model.
            model_b_results: BacktestResult from second model.

        Returns:
            Dictionary with comparison metrics and statistical tests.
        """
        pa = model_a_results.all_predictions
        pb = model_b_results.all_predictions
        ya = model_a_results.all_actuals
        yb = model_b_results.all_actuals

        if pa is None or pb is None or ya is None or yb is None:
            return {"error": "Both results must have raw prediction arrays."}

        # Use the shorter of the two (in case different backtest ranges)
        n = min(len(pa), len(pb))
        pa, pb = pa[:n], pb[:n]
        ya_n = ya[:n]

        # Per-match RPS for both models
        rps_a = np.array([
            ranked_probability_score(pa[i], int(ya_n[i]))
            for i in range(n)
        ])
        rps_b = np.array([
            ranked_probability_score(pb[i], int(ya_n[i]))
            for i in range(n)
        ])

        # Per-match log-loss
        ll_a = np.array([
            single_log_loss(pa[i], int(ya_n[i]))
            for i in range(n)
        ])
        ll_b = np.array([
            single_log_loss(pb[i], int(ya_n[i]))
            for i in range(n)
        ])

        # Diebold-Mariano test on RPS differences
        d = rps_a - rps_b  # negative = A is better
        dm_stat, dm_pval = self._diebold_mariano(d)

        # Accuracy comparison
        acc_a = float(np.mean(pa.argmax(axis=1) == ya_n))
        acc_b = float(np.mean(pb.argmax(axis=1) == ya_n))

        # Effect size (Cohen's d on RPS differences)
        d_std = np.std(d, ddof=1) if len(d) > 1 else 1e-12
        cohens_d = float(np.mean(d) / max(d_std, 1e-12))

        return {
            "n_matches": n,
            "model_a": {
                "rps": float(np.mean(rps_a)),
                "log_loss": float(np.mean(ll_a)),
                "accuracy": acc_a,
            },
            "model_b": {
                "rps": float(np.mean(rps_b)),
                "log_loss": float(np.mean(ll_b)),
                "accuracy": acc_b,
            },
            "rps_diff": float(np.mean(d)),        # negative = A wins
            "ll_diff": float(np.mean(ll_a - ll_b)),
            "acc_diff": acc_a - acc_b,
            "dm_statistic": dm_stat,
            "dm_pvalue": dm_pval,
            "significant_at_005": dm_pval < 0.05,
            "significant_at_001": dm_pval < 0.01,
            "cohens_d": cohens_d,
            "winner": "A" if np.mean(d) < 0 else ("B" if np.mean(d) > 0 else "tie"),
        }

    def print_report(self, result: BacktestResult) -> None:
        """Print a formatted backtest report to stdout."""
        sep = "=" * 70

        print(f"\n{sep}")
        print("  BACKTEST REPORT")
        print(sep)
        print(f"  Total OOS matches:  {result.total_matches:,}")
        print(f"  Windows completed:  {len(result.windows)}")
        print(f"  Elapsed:            {result.elapsed_seconds:.1f}s")
        print(sep)

        # -- Overall metrics --
        print("\n  OVERALL METRICS (pooled out-of-sample)")
        print("  " + "-" * 40)
        print(f"  RPS:        {result.rps:.5f}   (lower = better)")
        print(f"  Log-loss:   {result.log_loss:.5f}")
        print(f"  Brier:      {result.brier:.5f}")
        print(f"  Accuracy:   {result.accuracy:.4f}   ({result.accuracy * 100:.1f}%)")
        print(f"  ECE:        {result.ece:.5f}")

        # Stability across windows
        if result.windows:
            rps_vals = [w.rps for w in result.windows]
            acc_vals = [w.accuracy for w in result.windows]
            print(f"\n  RPS range:  {min(rps_vals):.5f} - {max(rps_vals):.5f}  "
                  f"(std={np.std(rps_vals):.5f})")
            print(f"  Acc range:  {min(acc_vals):.4f} - {max(acc_vals):.4f}  "
                  f"(std={np.std(acc_vals):.4f})")

        # -- Market comparison --
        print("\n  MARKET COMPARISON")
        print("  " + "-" * 40)
        delta = result.vs_market_rps
        direction = "BETTER" if delta < 0 else "WORSE"
        print(f"  vs Market RPS: {delta:+.5f}  ({direction} than bookmakers)")
        if abs(delta) < 0.005:
            print("  Interpretation: roughly on par with market odds")
        elif delta < -0.01:
            print("  Interpretation: meaningfully outperforming market")
        elif delta > 0.01:
            print("  Interpretation: market odds are still superior")

        # -- Upset detection --
        print("\n  UPSET DETECTION")
        print("  " + "-" * 40)
        print(f"  Precision:  {result.upset_precision:.4f}")
        print(f"  Recall:     {result.upset_recall:.4f}")
        print(f"  F1:         {result.upset_f1:.4f}")

        # -- Per-competition --
        if result.by_competition:
            print("\n  PER-COMPETITION BREAKDOWN")
            print("  " + "-" * 60)
            print(f"  {'Competition':<25} {'N':>5} {'RPS':>8} {'Acc':>7} {'vs Mkt':>8}")
            print("  " + "-" * 60)
            for comp, m in sorted(
                result.by_competition.items(),
                key=lambda x: -x[1].get("n", 0),
            ):
                n = m.get("n", 0)
                if n < 10:
                    continue
                rps_c = m.get("rps", 0)
                acc_c = m.get("accuracy", 0)
                vs_mkt = m.get("vs_market_rps", 0)
                mkt_flag = "+" if vs_mkt >= 0 else ""
                print(
                    f"  {comp:<25} {n:5d} {rps_c:8.5f} {acc_c:6.1%} {mkt_flag}{vs_mkt:7.5f}"
                )

        # -- Calibration --
        if result.calibration_bins:
            print("\n  CALIBRATION (reliability diagram data)")
            print("  " + "-" * 50)
            print(f"  {'Bin':>10} {'Pred Conf':>10} {'Act Acc':>10} {'Count':>7}")
            print("  " + "-" * 50)
            for b in result.calibration_bins:
                lo = b["bin_lower"]
                hi = b["bin_upper"]
                print(
                    f"  {lo:.1f}-{hi:.1f}   {b['avg_confidence']:10.4f} "
                    f"{b['avg_accuracy']:10.4f} {b['count']:7d}"
                )

        # -- Feature importance (top 10) --
        if result.feature_importance:
            # Filter to non-SHAP keys for the main table
            main_imp = {
                k: v for k, v in result.feature_importance.items()
                if not k.startswith("shap_")
            }
            if main_imp:
                print("\n  TOP 10 FEATURES (split importance)")
                print("  " + "-" * 40)
                for rank, (fname, imp) in enumerate(
                    sorted(main_imp.items(), key=lambda x: -x[1])[:10], 1
                ):
                    bar = "#" * int(imp * 200)
                    print(f"  {rank:2d}. {fname:<25} {imp:.5f}  {bar}")

            # SHAP importance if available
            shap_imp = {
                k.replace("shap_", ""): v
                for k, v in result.feature_importance.items()
                if k.startswith("shap_")
            }
            if shap_imp:
                print("\n  TOP 10 FEATURES (SHAP mean |value|)")
                print("  " + "-" * 40)
                for rank, (fname, imp) in enumerate(
                    sorted(shap_imp.items(), key=lambda x: -x[1])[:10], 1
                ):
                    print(f"  {rank:2d}. {fname:<25} {imp:.5f}")

        # -- Monthly trend --
        if result.by_period and len(result.by_period) > 3:
            print("\n  MONTHLY TREND (last 6)")
            print("  " + "-" * 50)
            print(f"  {'Period':>10} {'N':>5} {'RPS':>8} {'Acc':>7}")
            for p in result.by_period[-6:]:
                print(
                    f"  {p['period']:>10} {p['n']:5d} "
                    f"{p['rps']:8.5f} {p['accuracy']:6.1%}"
                )

        print(f"\n{sep}\n")

    def print_comparison(self, diff: dict) -> None:
        """Print a formatted A/B model comparison."""
        if "error" in diff:
            print(f"[backtest] comparison error: {diff['error']}")
            return

        sep = "=" * 60
        print(f"\n{sep}")
        print("  MODEL A/B COMPARISON")
        print(sep)
        print(f"  Matches compared: {diff['n_matches']:,}")

        a = diff["model_a"]
        b = diff["model_b"]
        print(f"\n  {'Metric':<15} {'Model A':>10} {'Model B':>10} {'Diff':>10}")
        print("  " + "-" * 50)
        print(f"  {'RPS':<15} {a['rps']:10.5f} {b['rps']:10.5f} {diff['rps_diff']:+10.5f}")
        print(f"  {'Log-loss':<15} {a['log_loss']:10.5f} {b['log_loss']:10.5f} {diff['ll_diff']:+10.5f}")
        print(f"  {'Accuracy':<15} {a['accuracy']:10.4f} {b['accuracy']:10.4f} {diff['acc_diff']:+10.4f}")

        print("\n  Diebold-Mariano test (H0: equal predictive accuracy):")
        print(f"    Statistic: {diff['dm_statistic']:.4f}")
        print(f"    p-value:   {diff['dm_pvalue']:.6f}")
        sig = "YES" if diff["significant_at_005"] else "NO"
        print(f"    Significant at 5%:  {sig}")
        print(f"    Cohen's d: {diff['cohens_d']:.4f}")
        print(f"    Winner: Model {diff['winner']}")
        print(f"\n{sep}\n")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_matches(self) -> pd.DataFrame:
        """Load all finished matches with extras from the database."""
        from footy.models.council import _match_query

        where = (
            "m.status='FINISHED' "
            "AND m.home_goals IS NOT NULL "
            "AND m.away_goals IS NOT NULL"
        )
        df = self.con.execute(_match_query(where, include_match_id=True)).df()

        if not df.empty:
            df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)
            df["home_team"] = df["home_team"].map(canonical_team_name)
            df["away_team"] = df["away_team"].map(canonical_team_name)

        return df

    def _extract_market_probs(self, df: pd.DataFrame) -> np.ndarray:
        """Extract fair market implied probabilities from closing odds.

        Tries multiple odds sources in priority order:
            1. Pinnacle closing (sharpest)
            2. Bet365 closing
            3. Market average
            4. Bet365 opening (fallback)

        Overround is removed via the scoring module's remove_overround.
        """
        n = len(df)
        market = np.full((n, 3), 1.0 / 3)

        # Source priority: Pinnacle > B365 closing > Average > B365 opening
        odds_sources = [
            ("psh", "psd", "psa"),           # Pinnacle
            ("b365ch", "b365cd", "b365ca"),   # Bet365 closing
            ("avgh", "avgd", "avga"),         # Market average
            ("b365h", "b365d", "b365a"),      # Bet365 opening
        ]

        for h_col, d_col, a_col in odds_sources:
            if h_col not in df.columns:
                continue
            for i in range(n):
                if market[i].sum() > 0.99 and abs(market[i, 0] - 1.0 / 3) > 0.001:
                    continue  # already have odds for this match
                try:
                    h_odds = float(df.iloc[i][h_col])
                    d_odds = float(df.iloc[i][d_col])
                    a_odds = float(df.iloc[i][a_col])
                except (TypeError, ValueError):
                    continue
                if h_odds <= 1.0 or d_odds <= 1.0 or a_odds <= 1.0:
                    continue
                imp = [1.0 / h_odds, 1.0 / d_odds, 1.0 / a_odds]
                fair = remove_overround(imp)
                market[i] = fair

        return market

    # ------------------------------------------------------------------
    # Expert computation
    # ------------------------------------------------------------------

    def _run_experts(self, df: pd.DataFrame) -> list[ExpertResult]:
        """Run all experts on the full match DataFrame."""
        return [expert.compute(df) for expert in ALL_EXPERTS]

    # ------------------------------------------------------------------
    # Model training per window
    # ------------------------------------------------------------------

    def _train_and_predict(
        self,
        Xtr: np.ndarray,
        ytr: np.ndarray,
        Xte: np.ndarray,
        use_catboost: bool = True,
    ) -> np.ndarray:
        """Train model on training data and return test probabilities.

        Mirrors the v13 Oracle stack: CatBoost primary + HistGBM fallback,
        with learned stack weights and temperature scaling.

        Returns:
            Probability array of shape (n_test, 3).
        """
        available: list[tuple[str, np.ndarray]] = []

        # --- CatBoost primary ---
        if use_catboost:
            try:
                from catboost import CatBoostClassifier

                # Use a small internal holdout for early stopping
                split_idx = int(len(Xtr) * 0.88)
                Xtr_cat, Xval_cat = Xtr[:split_idx], Xtr[split_idx:]
                ytr_cat, yval_cat = ytr[:split_idx], ytr[split_idx:]

                cat = CatBoostClassifier(
                    loss_function="MultiClass",
                    eval_metric="MultiClass",
                    iterations=2000,
                    learning_rate=0.03,
                    depth=6,
                    l2_leaf_reg=3.0,
                    subsample=0.8,
                    colsample_bylevel=0.7,
                    min_data_in_leaf=30,
                    use_best_model=True,
                    early_stopping_rounds=50,
                    random_seed=42,
                    verbose=0,
                )
                cat.fit(Xtr_cat, ytr_cat, eval_set=(Xval_cat, yval_cat))
                P_cat = cat.predict_proba(Xte)
                available.append(("CAT", P_cat))
            except ImportError:
                pass
            except Exception as e:
                log.debug("CatBoost failed in backtest window: %s", e)

        # --- HistGBM fallback (always available) ---
        gbm = HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_depth=5,
            max_iter=2000,
            l2_regularization=3.0,
            min_samples_leaf=30,
            max_bins=255,
            max_leaf_nodes=63,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=50,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gbm.fit(Xtr, ytr)
        P_gbm = gbm.predict_proba(Xte)
        available.append(("GBM", P_gbm))

        # --- Stack via Nelder-Mead on training logloss ---
        if len(available) >= 2:
            from scipy.optimize import minimize as sp_minimize

            # Use last 12% of training data as validation for stack weights
            val_n = max(100, int(len(Xtr) * 0.12))
            Xval_stack = Xtr[-val_n:]
            yval_stack = ytr[-val_n:]

            # Get validation predictions from each model
            val_preds = []
            for name, _ in available:
                if name == "CAT":
                    try:
                        vp = cat.predict_proba(Xval_stack)
                    except Exception:
                        vp = gbm.predict_proba(Xval_stack)
                else:
                    vp = gbm.predict_proba(Xval_stack)
                val_preds.append(vp)

            def _stack_ll(w_raw):
                w = np.exp(w_raw) / np.exp(w_raw).sum()
                P_blend = sum(wi * Pi for wi, Pi in zip(w, val_preds))
                row_sums = P_blend.sum(axis=1, keepdims=True)
                P_blend = P_blend / np.maximum(row_sums, 1e-12)
                return -np.mean(
                    np.log(P_blend[np.arange(len(yval_stack)), yval_stack] + 1e-12)
                )

            w0 = np.zeros(len(available))
            res = sp_minimize(
                _stack_ll, w0, method="Nelder-Mead",
                options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-6},
            )
            opt_w = np.exp(res.x) / np.exp(res.x).sum()
            P = sum(
                wi * Pi for wi, (_, Pi) in zip(opt_w, available)
            )
        else:
            P = available[0][1]

        # Normalize
        row_sums = P.sum(axis=1, keepdims=True)
        P = P / np.maximum(row_sums, 1e-12)

        # Temperature scaling on the last 12% of training data
        try:
            from scipy.special import softmax as scipy_softmax

            val_n_t = max(100, int(len(Xtr) * 0.12))
            Xval_t = Xtr[-val_n_t:]
            yval_t = ytr[-val_n_t:]

            # Get blended validation predictions
            if len(available) >= 2:
                P_val = sum(
                    wi * Pi for wi, Pi in zip(opt_w, val_preds)
                )
                P_val = P_val / np.maximum(P_val.sum(axis=1, keepdims=True), 1e-12)
            else:
                P_val = gbm.predict_proba(Xval_t)

            logits_val = np.log(P_val + 1e-12)

            def _temp_nll(T_arr):
                T_val = float(T_arr[0])
                scaled = logits_val / max(T_val, 0.01)
                probs = scipy_softmax(scaled, axis=1)
                return -np.mean(
                    np.log(probs[np.arange(len(yval_t)), yval_t] + 1e-12)
                )

            from scipy.optimize import minimize as sp_min2
            res_t = sp_min2(
                _temp_nll, x0=[1.5], bounds=[(0.1, 10.0)], method="L-BFGS-B"
            )
            temperature = float(res_t.x[0])

            if temperature != 1.0:
                logits_test = np.log(P + 1e-12)
                P = scipy_softmax(logits_test / temperature, axis=1)
        except Exception:
            pass

        # Soft floor/ceiling to avoid degenerate probabilities
        P = np.clip(P, 0.02, 0.95)
        row_sums = P.sum(axis=1, keepdims=True)
        P = P / np.maximum(row_sums, 1e-12)

        return P

    def _extract_importance(
        self,
        Xtr: np.ndarray,
        ytr: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Train a quick HistGBM and extract feature importance."""
        try:
            gbm = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=4,
                max_iter=500,
                l2_regularization=3.0,
                min_samples_leaf=50,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=42,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gbm.fit(Xtr, ytr)
            if hasattr(gbm, "feature_importances_"):
                imp = gbm.feature_importances_
                return {
                    fname: float(imp[i])
                    for i, fname in enumerate(feature_names)
                    if i < len(imp)
                }
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_calibration(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> list[dict]:
        """Compute calibration data for reliability diagrams.

        Groups predictions by predicted confidence and compares the
        average predicted probability to the actual win rate within
        each bin.

        Returns:
            List of dicts with bin_lower, bin_upper, avg_confidence,
            avg_accuracy, count.
        """
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = (pred == outcomes).astype(float)

        bins_edges = np.linspace(0.0, 1.0, n_bins + 1)
        calibration: list[dict] = []

        for i in range(n_bins):
            lo, hi = bins_edges[i], bins_edges[i + 1]
            if i == n_bins - 1:
                mask = (conf >= lo) & (conf <= hi)
            else:
                mask = (conf >= lo) & (conf < hi)

            if mask.any():
                calibration.append({
                    "bin_lower": float(lo),
                    "bin_upper": float(hi),
                    "avg_confidence": float(conf[mask].mean()),
                    "avg_accuracy": float(correct[mask].mean()),
                    "count": int(mask.sum()),
                })

        return calibration

    def _compute_upset_metrics(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        market_probs: np.ndarray,
    ) -> dict:
        """Compute upset detection metrics.

        An upset is defined as: the market favourite lost.
        We check how well our model detected these upsets by
        looking at cases where we shifted probability away from
        the favourite compared to market odds.

        Returns:
            Dict with precision, recall, f1, n_upsets, n_detected.
        """
        n = len(outcomes)
        has_odds = market_probs.sum(axis=1) > 0.5

        # Market favourite = argmax of market probs
        # Upset = market favourite did NOT win
        mkt_fav = market_probs.argmax(axis=1)
        mkt_fav_conf = market_probs.max(axis=1)

        # Only consider matches where the favourite had > 45% probability
        # (otherwise it's not really a clear favourite)
        strong_fav = has_odds & (mkt_fav_conf > 0.45)

        if strong_fav.sum() == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "n_upsets": 0, "n_detected": 0}

        actual_upset = strong_fav & (mkt_fav != outcomes)
        n_upsets = int(actual_upset.sum())

        if n_upsets == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "n_upsets": 0, "n_detected": 0}

        # Our model "predicted" an upset when it assigned less probability
        # to the market favourite than the market did (by at least 5pp).
        our_fav_prob = np.array([
            probs[i, mkt_fav[i]] for i in range(n)
        ])
        mkt_fav_prob = np.array([
            market_probs[i, mkt_fav[i]] for i in range(n)
        ])
        predicted_upset = strong_fav & ((mkt_fav_prob - our_fav_prob) > 0.05)

        n_predicted = int(predicted_upset.sum())
        if n_predicted == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "n_upsets": n_upsets, "n_detected": 0}

        # True positives: we predicted upset AND it happened
        tp = int((predicted_upset & actual_upset).sum())

        precision = tp / max(n_predicted, 1)
        recall = tp / max(n_upsets, 1)
        f1 = (
            2 * precision * recall / max(precision + recall, 1e-12)
        )

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "n_upsets": n_upsets,
            "n_detected": tp,
            "n_predicted": n_predicted,
        }

    def _compute_by_competition(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        market_probs: np.ndarray,
        competitions: np.ndarray,
    ) -> dict:
        """Break down metrics by competition."""
        result: dict[str, dict] = {}
        unique_comps = np.unique(competitions)

        for comp in unique_comps:
            mask = competitions == comp
            n = int(mask.sum())
            if n < 5:
                continue

            p_c = probs[mask]
            y_c = outcomes[mask]
            m_c = market_probs[mask]

            rps_c = float(np.mean([
                ranked_probability_score(p_c[i], int(y_c[i]))
                for i in range(n)
            ]))
            ll_c = float(np.mean([
                single_log_loss(p_c[i], int(y_c[i]))
                for i in range(n)
            ]))
            acc_c = float(np.mean(p_c.argmax(axis=1) == y_c))

            # Market RPS for this competition
            has_odds = m_c.sum(axis=1) > 0.5
            if has_odds.sum() > 0:
                mkt_rps_c = float(np.mean([
                    ranked_probability_score(m_c[i], int(y_c[i]))
                    for i in range(n)
                    if has_odds[i]
                ]))
                vs_mkt = rps_c - mkt_rps_c
            else:
                mkt_rps_c = 0.0
                vs_mkt = 0.0

            result[str(comp)] = {
                "n": n,
                "rps": rps_c,
                "log_loss": ll_c,
                "accuracy": acc_c,
                "market_rps": mkt_rps_c,
                "vs_market_rps": vs_mkt,
            }

        return result

    def _compute_by_period(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        dates: np.ndarray,
    ) -> list[dict]:
        """Break down metrics by month."""
        df_tmp = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "outcome": outcomes,
        })
        df_tmp["period"] = df_tmp["date"].dt.to_period("M").astype(str)

        periods: list[dict] = []
        for period, group in df_tmp.groupby("period"):
            idx = group.index.to_numpy()
            n = len(idx)
            if n < 3:
                continue
            p_p = probs[idx]
            y_p = outcomes[idx]

            rps_p = float(np.mean([
                ranked_probability_score(p_p[i], int(y_p[i]))
                for i in range(n)
            ]))
            acc_p = float(np.mean(p_p.argmax(axis=1) == y_p))

            periods.append({
                "period": str(period),
                "n": n,
                "rps": rps_p,
                "accuracy": acc_p,
            })

        return periods

    def _compute_shap_importance_from_data(
        self,
        X_full: np.ndarray,
        y_full: np.ndarray,
        dates_full: np.ndarray,
        test_dates: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Compute SHAP feature importance if the shap library is installed.

        Trains a single HistGBM on data before the test period, then
        computes TreeExplainer SHAP values on a sample of test data.

        Returns:
            Dict of feature_name -> mean |SHAP value|, or empty dict
            if shap is not available.
        """
        try:
            import shap
        except ImportError:
            return {}

        try:
            # Use data before test period for training
            test_start = pd.Timestamp(np.min(test_dates))
            dates_pd = pd.to_datetime(dates_full)
            train_mask = (dates_pd < test_start).to_numpy()
            test_mask = np.isin(dates_full, test_dates)

            if train_mask.sum() < 500 or test_mask.sum() < 50:
                return {}

            Xtr = X_full[train_mask]
            ytr = y_full[train_mask]
            Xte = X_full[test_mask]

            # Subsample test data for speed (SHAP can be slow)
            max_shap = min(500, len(Xte))
            rng = np.random.RandomState(42)
            shap_idx = rng.choice(len(Xte), max_shap, replace=False)
            Xte_sample = Xte[shap_idx]

            gbm = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=4,
                max_iter=500,
                min_samples_leaf=50,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=42,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gbm.fit(Xtr, ytr)

            explainer = shap.TreeExplainer(gbm)
            shap_values = explainer.shap_values(Xte_sample)

            # shap_values is list of (n_samples, n_features) per class
            # Average absolute SHAP across classes
            if isinstance(shap_values, list):
                combined = np.mean(
                    [np.abs(sv) for sv in shap_values], axis=0
                )
            else:
                combined = np.abs(shap_values)

            mean_abs = combined.mean(axis=0)
            return {
                feature_names[i]: float(mean_abs[i])
                for i in range(min(len(feature_names), len(mean_abs)))
            }
        except Exception as e:
            log.debug("SHAP computation failed: %s", e)
            return {}

    def _average_importance(
        self,
        window_results: list[WindowResult],
        feature_names: list[str],
    ) -> dict[str, float]:
        """Average feature importance across all windows, weighted by test size."""
        if not window_results or not feature_names:
            return {}

        weights = np.array(
            [w.test_size for w in window_results], dtype=float
        )
        total_weight = weights.sum()
        if total_weight == 0:
            return {}

        avg: dict[str, float] = {}
        for fname in feature_names:
            val = 0.0
            for wr, w in zip(window_results, weights):
                val += wr.feature_importances.get(fname, 0.0) * w
            avg[fname] = val / total_weight

        # Sort descending
        return dict(sorted(avg.items(), key=lambda x: -x[1]))

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    @staticmethod
    def _diebold_mariano(
        d: np.ndarray,
        h: int = 1,
    ) -> tuple[float, float]:
        """Diebold-Mariano test for equal predictive accuracy.

        Tests H0: E[d_t] = 0  where d_t = L(e_A,t) - L(e_B,t)
        is the difference in loss between two forecasters.

        Uses the Harvey-Leybourne-Newbold (1997) small-sample correction.

        Args:
            d: Array of loss differentials (model A minus model B).
            h: Forecast horizon (1 for one-step-ahead).

        Returns:
            (test_statistic, two_sided_p_value)
        """
        n = len(d)
        if n < 10:
            return (0.0, 1.0)

        d_mean = np.mean(d)

        # Autocovariance estimation for Newey-West variance
        gamma_0 = np.var(d, ddof=1)
        gamma_sum = 0.0
        for k in range(1, h):
            if k >= n:
                break
            gamma_k = np.cov(d[k:], d[:-k], ddof=1)[0, 1] if n > k else 0.0
            gamma_sum += gamma_k

        var_d = (gamma_0 + 2 * gamma_sum) / n
        if var_d <= 0:
            var_d = gamma_0 / n

        dm_stat = d_mean / max(np.sqrt(var_d), 1e-12)

        # Harvey-Leybourne-Newbold small-sample correction
        correction = np.sqrt(
            (n + 1 - 2 * h + h * (h - 1) / n) / n
        )
        dm_corrected = dm_stat * correction

        # Two-sided p-value from t-distribution with n-1 df
        p_value = 2.0 * scipy_stats.t.sf(abs(dm_corrected), df=n - 1)

        return (float(dm_corrected), float(p_value))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_backtest(
    con,
    seasons: int = 3,
    verbose: bool = True,
    **kwargs,
) -> BacktestResult:
    """Convenience function to run a backtest.

    Args:
        con: DuckDB connection.
        seasons: Number of seasons to backtest.
        verbose: Print progress.
        **kwargs: Additional arguments passed to Backtester.run().

    Returns:
        BacktestResult with all metrics.
    """
    bt = Backtester(con)
    return bt.run(seasons=seasons, verbose=verbose, **kwargs)
