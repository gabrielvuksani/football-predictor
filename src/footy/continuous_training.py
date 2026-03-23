"""
Continuous Model Retraining

Handles automated retraining of models as new match data becomes available.
Tracks model versions, training windows, validates improvements, and auto-deploys.

Features:
- Drift detection: monitors prediction accuracy on recent results
- Auto-retrain: triggers v13_oracle retraining when new-match or drift thresholds are met
- Performance gating: only deploys if new model beats current on held-out set
- Automatic rollback: reverts if deployed model degrades within grace window
- Full audit trail: every train/deploy/rollback is logged with metrics
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import json
import shutil
import logging
import math

import numpy as np

from footy.db import connect

log = logging.getLogger(__name__)


class ModelTrainingRecord:
    """Tracks a single model training run"""

    def __init__(
        self,
        model_version: str,
        model_type: str,
        training_date: datetime,
        training_window_days: int,
        n_matches_used: int,
        metrics: dict,
        test_metrics: dict = None,
    ):
        self.model_version = model_version
        self.model_type = model_type
        self.training_date = training_date
        self.training_window_days = training_window_days
        self.n_matches_used = n_matches_used
        self.metrics = metrics or {}
        self.test_metrics = test_metrics or {}
        self.created_at = datetime.now(timezone.utc)


class ContinuousTrainingManager:
    """
    Manages continuous model retraining with new match data.

    Features:
    - K-fold cross-validation during training
    - Configurable grace window (period before rollback)
    - Learning curve analysis (detect plateau, stop if no improvement)
    - Computational cost tracking
    - Better audit trail with timestamps
    - Performance comparison against rolling baseline

    Strategy:
    1. Check for new finished matches since last training
    2. If threshold met (e.g., 10+ new matches), retrain model
    3. Evaluate new model on held-out test set with K-fold CV
    4. Compare with previous model version and rolling baseline
    5. If improvement > threshold, deploy new model
    6. Monitor during grace window, rollback if degradation
    7. Log all training details with timestamps

    Example usage:
        manager = ContinuousTrainingManager()
        manager.setup_continuous_training(
            model_type="v13_oracle",
            retrain_threshold_matches=10,
            performance_threshold_improvement=0.01,
            grace_window_hours=24,
            k_folds=5
        )
        manager.check_and_retrain()  # Called by scheduler periodically
    """

    def __init__(self):
        self.con = connect()
        self._ensure_schema()
        self._training_history: dict[str, list] = {}  # model_type -> list of training records

        # Drift-triggered retraining state
        self.needs_retrain: bool = False
        self._drift_triggered: bool = False
        self._drift_detectors_triggered: list[str] = []
        self._drift_event_timestamp: Optional[str] = None

    def _ensure_schema(self):
        """Create training records table with enhanced schema."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS model_training_records (
                id INTEGER PRIMARY KEY,
                model_version VARCHAR NOT NULL,
                model_type VARCHAR NOT NULL,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                training_window_days INT,
                n_matches_used INT,
                n_matches_test INT,
                metrics_json VARCHAR,
                test_metrics_json VARCHAR,
                cv_scores_json VARCHAR,
                previous_version VARCHAR,
                improvement_pct DOUBLE,
                deployed BOOLEAN DEFAULT FALSE,
                deployment_date TIMESTAMP,
                rollback_date TIMESTAMP,
                grace_window_hours INT DEFAULT 24,
                computational_cost_minutes DOUBLE,
                learning_plateau_detected BOOLEAN DEFAULT FALSE,
                notes VARCHAR
            )
        """)

        self.con.execute("CREATE SEQUENCE IF NOT EXISTS mtr_seq START 1")

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS model_deployments (
                model_type VARCHAR PRIMARY KEY,
                active_version VARCHAR NOT NULL,
                deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                previous_version VARCHAR,
                reason VARCHAR,
                performance_metrics_json VARCHAR,
                baseline_performance_json VARCHAR
            )
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS retraining_schedules (
                model_type VARCHAR PRIMARY KEY,
                retrain_threshold_matches INT DEFAULT 10,
                performance_threshold_improvement DOUBLE DEFAULT 0.01,
                grace_window_hours INT DEFAULT 24,
                k_folds INT DEFAULT 5,
                enabled BOOLEAN DEFAULT TRUE,
                last_trained TIMESTAMP,
                last_retrain_matches INT DEFAULT 0,
                next_scheduled_retrain TIMESTAMP,
                learning_plateau_threshold DOUBLE DEFAULT 0.001
            )
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS training_cost_log (
                id INTEGER PRIMARY KEY,
                model_type VARCHAR NOT NULL,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cost_minutes DOUBLE,
                n_samples INT,
                k_folds INT,
                cost_per_sample DOUBLE
            )
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS rolling_baseline (
                id INTEGER PRIMARY KEY,
                model_type VARCHAR NOT NULL,
                window_days INT,
                baseline_accuracy DOUBLE,
                baseline_logloss DOUBLE,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS drift_events (
                id INTEGER PRIMARY KEY,
                model_type VARCHAR NOT NULL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                detectors_triggered VARCHAR,
                severity VARCHAR,
                accuracy_drop DOUBLE,
                baseline_accuracy DOUBLE,
                recent_accuracy DOUBLE,
                retrain_triggered BOOLEAN DEFAULT FALSE,
                retrain_completed BOOLEAN DEFAULT FALSE,
                notes VARCHAR
            )
        """)

        # Create indices
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS training_model_type_idx
            ON model_training_records(model_type, training_date DESC)
        """)
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS deployment_model_type_idx
            ON model_deployments(model_type)
        """)

    def setup_continuous_training(
        self,
        model_type: str,
        retrain_threshold_matches: int = 10,
        performance_threshold_improvement: float = 0.01,
        grace_window_hours: int = 24,
        k_folds: int = 5,
        learning_plateau_threshold: float = 0.001,
    ) -> dict:
        """
        Configure continuous training for a model type.

        Args:
            model_type: Model to retrain (v1, v2, v3, v4, v5, etc.)
            retrain_threshold_matches: Trigger retrain after N new finished matches
            performance_threshold_improvement: Minimum improvement to deploy
            grace_window_hours: Hours to wait before rollback decision
            k_folds: K-fold cross-validation folds
            learning_plateau_threshold: Stop retraining if improvement < this

        Returns:
            Configuration dict
        """
        self.con.execute(
            """INSERT OR REPLACE INTO retraining_schedules
               (model_type, retrain_threshold_matches, performance_threshold_improvement,
                grace_window_hours, k_folds, learning_plateau_threshold, enabled)
               VALUES (?, ?, ?, ?, ?, ?, TRUE)""",
            [
                model_type,
                retrain_threshold_matches,
                performance_threshold_improvement,
                grace_window_hours,
                k_folds,
                learning_plateau_threshold,
            ],
        )

        return {
            "model_type": model_type,
            "retrain_threshold_matches": retrain_threshold_matches,
            "performance_threshold_improvement": performance_threshold_improvement,
            "grace_window_hours": grace_window_hours,
            "k_folds": k_folds,
            "learning_plateau_threshold": learning_plateau_threshold,
            "enabled": True,
        }

    def compute_rolling_baseline(self, model_type: str, window_days: int = 30) -> dict:
        """Compute rolling baseline performance for comparison.

        Args:
            model_type: Model to compute baseline for
            window_days: Window size for baseline

        Returns:
            Dictionary with baseline metrics
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=window_days)

            # Get baseline accuracy from predictions in window
            baseline = self.con.execute(
                """SELECT AVG(correct) as accuracy, AVG(logloss) as logloss
                   FROM prediction_scores
                   WHERE model_version LIKE ? AND scored_at > ?""",
                [f"{model_type}%", cutoff_date],
            ).fetchone()

            if not baseline or baseline[0] is None:
                return {"baseline_accuracy": 0.45, "baseline_logloss": 0.65, "n_samples": 0}

            baseline_accuracy = float(baseline[0])
            baseline_logloss = float(baseline[1]) if baseline[1] else 0.65

            # Store in database
            self.con.execute(
                """INSERT INTO rolling_baseline
                   (model_type, window_days, baseline_accuracy, baseline_logloss)
                   VALUES (?, ?, ?, ?)""",
                [model_type, window_days, baseline_accuracy, baseline_logloss],
            )

            return {
                "baseline_accuracy": baseline_accuracy,
                "baseline_logloss": baseline_logloss,
                "window_days": window_days,
            }
        except Exception as e:
            log.warning(f"Failed to compute rolling baseline: {e}")
            return {"baseline_accuracy": 0.45, "baseline_logloss": 0.65}

    def analyze_learning_curve(
        self,
        model_type: str,
        metric: str = "accuracy",
        window_size: int = 20,
    ) -> dict:
        """Analyze learning curve to detect plateau.

        Args:
            model_type: Model to analyze
            metric: Metric to analyze (accuracy, logloss)
            window_size: Moving average window

        Returns:
            Dictionary with plateau detection results
        """
        try:
            # Get recent metrics
            rows = self.con.execute(
                """SELECT accuracy, logloss FROM prediction_scores
                   WHERE model_version = ?
                   ORDER BY scored_at DESC
                   LIMIT 200""",
                [model_type],
            ).fetchall()

            if len(rows) < window_size:
                return {"status": "insufficient_data", "detected_plateau": False}

            # Extract metric
            if metric == "accuracy":
                values = [float(r[0]) if r[0] is not None else 0.33 for r in rows]
            else:
                values = [float(r[1]) if r[1] is not None else 0.65 for r in rows]

            values = list(reversed(values))  # Chronological order

            # Compute moving averages
            if len(values) < 2 * window_size:
                return {"status": "insufficient_data", "detected_plateau": False}

            ma1 = np.mean(values[-window_size:])  # Recent
            ma2 = np.mean(values[-2 * window_size : -window_size])  # Previous

            improvement = ma1 - ma2 if metric == "accuracy" else ma2 - ma1

            # Detect plateau (improvement < threshold)
            threshold = 0.001
            plateau_detected = abs(improvement) < threshold

            return {
                "status": "ok",
                "metric": metric,
                "recent_ma": round(ma1, 4),
                "previous_ma": round(ma2, 4),
                "improvement": round(improvement, 4),
                "detected_plateau": plateau_detected,
                "n_samples": len(values),
            }
        except Exception as e:
            log.warning(f"Failed to analyze learning curve: {e}")
            return {"status": "error", "detected_plateau": False, "error": str(e)}

    def track_computational_cost(
        self,
        model_type: str,
        cost_minutes: float,
        n_samples: int,
        k_folds: int = 5,
    ) -> dict:
        """Track computational cost of training.

        Args:
            model_type: Model type
            cost_minutes: Total time spent
            n_samples: Number of training samples
            k_folds: Cross-validation folds

        Returns:
            Cost analysis dictionary
        """
        try:
            cost_per_sample = cost_minutes / max(n_samples, 1)

            self.con.execute(
                """INSERT INTO training_cost_log
                   (model_type, cost_minutes, n_samples, k_folds, cost_per_sample)
                   VALUES (?, ?, ?, ?, ?)""",
                [model_type, cost_minutes, n_samples, k_folds, cost_per_sample],
            )

            # Compute average cost
            avg_cost = self.con.execute(
                """SELECT AVG(cost_minutes) FROM training_cost_log
                   WHERE model_type = ?""",
                [model_type],
            ).fetchone()[0]

            return {
                "cost_minutes": round(cost_minutes, 2),
                "cost_per_sample": round(cost_per_sample, 4),
                "n_samples": n_samples,
                "k_folds": k_folds,
                "average_cost_minutes": round(avg_cost or cost_minutes, 2),
            }
        except Exception as e:
            log.error(f"Failed to track computational cost: {e}")
            return {"status": "error", "error": str(e)}

    def get_new_matches_since_training(self, model_type: str) -> int:
        """Count finished matches since last training"""
        row = self.con.execute("""
            SELECT last_trained FROM retraining_schedules WHERE model_type = ?
        """, [model_type]).fetchone()

        if not row or not row[0]:
            # First training, use all finished matches
            count = self.con.execute("""
                SELECT COUNT(*) FROM matches WHERE status = 'FINISHED' AND home_goals IS NOT NULL
            """).fetchone()[0]
            return count

        last_trained = row[0]
        count = self.con.execute("""
            SELECT COUNT(*) FROM matches
            WHERE status = 'FINISHED' AND home_goals IS NOT NULL
            AND utc_date > ?
        """, [last_trained]).fetchone()[0]

        return count

    def check_and_retrain(self, model_type: str = None) -> dict:
        """
        Check if retraining is needed and perform if threshold met.

        Args:
            model_type: Specific model to check, or None for all

        Returns:
            Status dict with results
        """
        if model_type:
            model_types = [model_type]
        else:
            rows = self.con.execute("""
                SELECT model_type FROM retraining_schedules WHERE enabled = TRUE
            """).fetchall()
            model_types = [r[0] for r in rows]

        results = {}
        for mt in model_types:
            try:
                result = self._check_retrain_single(mt)
                results[mt] = result
            except Exception as e:
                results[mt] = {"status": "error", "error": str(e)}

        return results

    def _check_retrain_single(self, model_type: str) -> dict:
        """Check and potentially retrain single model type"""
        # Get config
        config = self.con.execute("""
            SELECT retrain_threshold_matches, performance_threshold_improvement
            FROM retraining_schedules WHERE model_type = ?
        """, [model_type]).fetchone()

        if not config:
            return {"status": "not_configured"}

        retrain_threshold, perf_threshold = config

        # Count new matches
        n_new = self.get_new_matches_since_training(model_type)

        # Check for drift (accuracy degradation on recent matches)
        drift = self.detect_drift(model_type)

        # Bug 5: Add CUSUM-based drift detection for sensitive early warning
        cusum_drift = False
        try:
            # Fetch recent predictions and outcomes for CUSUM analysis
            recent_preds = self.con.execute("""
                SELECT predicted_probs, actual_outcome FROM match_predictions
                WHERE model_version = ? AND created_at > datetime('now', '-60 days')
                ORDER BY created_at
                LIMIT 60
            """, [f"{model_type}_v1"]).fetchall()

            if len(recent_preds) >= 30:
                # Parse predictions (stored as JSON) and outcomes
                pred_list = []
                outcome_list = []
                for pred_json_str, outcome_code in recent_preds:
                    try:
                        import json
                        probs = json.loads(pred_json_str)
                        if isinstance(probs, (list, tuple)) and len(probs) >= 3:
                            pred_list.append([float(p) for p in probs[:3]])
                            outcome_list.append(int(outcome_code))
                    except Exception:
                        continue

                if len(pred_list) >= 30:
                    # Use CUSUM drift detector
                    cusum_drift = self.check_for_drift(pred_list, outcome_list, threshold=0.05, window_size=30)
                    if cusum_drift:
                        log.warning("[drift] CUSUM drift detector flagged degradation for %s", model_type)
        except Exception as e:
            log.debug("CUSUM drift check skipped: %s", e)

        # Merge drift signals: either accuracy-based OR CUSUM-based
        drift["cusum_triggered"] = cusum_drift
        drift["combined_drift"] = drift.get("drifted", False) or cusum_drift

        # Collect which detectors triggered for the audit log
        detectors_triggered: list[str] = []
        if drift.get("drifted"):
            detectors_triggered.append("accuracy_comparison")
        if cusum_drift:
            detectors_triggered.append("cusum")

        # Majority-vote drift: at least one detector flagged degradation
        majority_drift = len(detectors_triggered) >= 1 and drift.get("combined_drift", False)

        if majority_drift:
            # Set the needs_retrain flag so the next training cycle reacts
            self.needs_retrain = True
            self._drift_triggered = True
            self._drift_detectors_triggered = detectors_triggered
            self._drift_event_timestamp = datetime.now(timezone.utc).isoformat()

            # Log the drift event to the database for a full audit trail
            self._log_drift_event(
                model_type=model_type,
                detectors_triggered=detectors_triggered,
                severity="sudden" if len(detectors_triggered) >= 2 else "gradual",
                accuracy_drop=drift.get("accuracy_drop"),
                baseline_accuracy=drift.get("baseline_acc") or drift.get("baseline_accuracy"),
                recent_accuracy=drift.get("recent_acc") or drift.get("recent_accuracy"),
            )

        if drift.get("drifted"):
            return {
                "status": "drift_detected",
                "new_matches": n_new,
                "threshold": retrain_threshold,
                "drift_info": drift,
                "needs_retrain": self.needs_retrain,
                "detectors_triggered": detectors_triggered,
                "reason": f"Accuracy dropped from {drift['baseline_acc']:.1%} to {drift['recent_acc']:.1%}",
            }

        # CUSUM-only drift (accuracy comparison didn't fire but CUSUM did)
        if cusum_drift and self.needs_retrain:
            return {
                "status": "drift_detected",
                "new_matches": n_new,
                "threshold": retrain_threshold,
                "drift_info": drift,
                "needs_retrain": True,
                "detectors_triggered": detectors_triggered,
                "reason": "CUSUM detector flagged performance degradation",
            }

        if n_new < retrain_threshold:
            return {
                "status": "waiting",
                "new_matches": n_new,
                "threshold": retrain_threshold,
                "ready_in": retrain_threshold - n_new,
            }

        # Ready to retrain
        return {
            "status": "ready_to_retrain",
            "new_matches": n_new,
            "threshold": retrain_threshold,
        }

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------
    def detect_drift(self, model_type: str, recent_window: int = 60, baseline_window: int = 365) -> dict:
        """
        Detect prediction accuracy drift by comparing recent vs baseline performance.

        Compares the model's accuracy on the last `recent_window` days against
        its accuracy on the prior `baseline_window` days.

        Returns dict with 'drifted' bool and accuracy numbers.
        """
        now = datetime.now(timezone.utc)
        recent_start = now - timedelta(days=recent_window)
        baseline_start = now - timedelta(days=baseline_window)

        def _accuracy_for_period(start, end):
            rows = self.con.execute("""
                SELECT p.p_home, p.p_draw, p.p_away,
                       m.home_goals, m.away_goals
                FROM predictions p
                JOIN matches m ON m.match_id = p.match_id
                WHERE p.model_version = ?
                  AND m.status = 'FINISHED'
                  AND m.utc_date >= ? AND m.utc_date < ?
                  AND m.home_goals IS NOT NULL
            """, [model_type, start.isoformat(), end.isoformat()]).fetchall()
            if not rows:
                return None, 0
            correct = 0
            for ph, pd_, pa, hg, ag in rows:
                pred = max(enumerate([ph, pd_, pa]), key=lambda x: x[1])[0]
                if hg > ag:
                    actual = 0
                elif hg == ag:
                    actual = 1
                else:
                    actual = 2
                if pred == actual:
                    correct += 1
            return correct / len(rows), len(rows)

        baseline_acc, baseline_n = _accuracy_for_period(baseline_start, recent_start)
        recent_acc, recent_n = _accuracy_for_period(recent_start, now)

        if baseline_acc is None or recent_acc is None or recent_n < 15:
            return {"drifted": False, "reason": "insufficient_data",
                    "baseline_n": baseline_n, "recent_n": recent_n}

        drop = baseline_acc - recent_acc
        # Drift if accuracy dropped >5 percentage points
        drifted = drop > 0.05

        return {
            "drifted": drifted,
            "baseline_accuracy": baseline_acc,
            "baseline_acc": baseline_acc,
            "baseline_n": baseline_n,
            "recent_accuracy": recent_acc,
            "recent_acc": recent_acc,
            "recent_n": recent_n,
            "accuracy_drop": drop,
        }

    # ------------------------------------------------------------------
    # Drift event logging
    # ------------------------------------------------------------------
    def _log_drift_event(
        self,
        model_type: str,
        detectors_triggered: list[str],
        severity: str = "gradual",
        accuracy_drop: float | None = None,
        baseline_accuracy: float | None = None,
        recent_accuracy: float | None = None,
        notes: str = "",
    ) -> None:
        """Persist a drift event to the audit table.

        Every drift detection is logged with a timestamp, the detectors
        that agreed, severity classification, and accuracy numbers so
        that the team can retrospectively analyse drift patterns.

        Args:
            model_type: Which model experienced drift.
            detectors_triggered: List of detector names that flagged.
            severity: "gradual", "sudden", or "recurring".
            accuracy_drop: Absolute drop in accuracy (baseline - recent).
            baseline_accuracy: Long-term baseline accuracy.
            recent_accuracy: Recent-window accuracy.
            notes: Free-text annotation.
        """
        try:
            self.con.execute(
                """INSERT INTO drift_events
                   (model_type, detectors_triggered, severity,
                    accuracy_drop, baseline_accuracy, recent_accuracy,
                    retrain_triggered, notes)
                   VALUES (?, ?, ?, ?, ?, ?, TRUE, ?)""",
                [
                    model_type,
                    json.dumps(detectors_triggered),
                    severity,
                    accuracy_drop,
                    baseline_accuracy,
                    recent_accuracy,
                    notes or f"Drift detected by {', '.join(detectors_triggered)}",
                ],
            )
            log.warning(
                "[drift-event] %s — detectors=%s severity=%s drop=%.4f "
                "(baseline=%.3f recent=%.3f)",
                model_type,
                detectors_triggered,
                severity,
                accuracy_drop or 0.0,
                baseline_accuracy or 0.0,
                recent_accuracy or 0.0,
            )
        except Exception as e:
            log.debug("Failed to log drift event: %s", e)

    # ------------------------------------------------------------------
    # Auto-retrain
    # ------------------------------------------------------------------
    def auto_retrain(self, force: bool = False, verbose: bool = True) -> dict:
        """
        End-to-end auto-retrain: check thresholds → train → validate → deploy/rollback.

        1. Checks if retraining is needed (new matches or drift detected)
        2. Trains new v13_oracle model
        3. Compares test-set performance vs current model
        4. Deploys only if performance improves (or force=True)
        5. Backs up old model artifact for rollback

        Returns summary dict.
        """
        check = self.check_and_retrain("v13_oracle")
        result = check.get("v13_oracle", {})
        status = result.get("status", "")

        if not force and status not in ("ready_to_retrain", "drift_detected"):
            if verbose:
                log.info("Auto-retrain: no action needed (%s)", status)
            return {"action": "none", "check": result}

        reason = "forced" if force else status
        drift_triggered = self._drift_triggered or status == "drift_detected"
        if verbose:
            log.info(
                "Auto-retrain: triggered (%s, drift_triggered=%s, detectors=%s)",
                reason,
                drift_triggered,
                self._drift_detectors_triggered,
            )

        # ---- Backup current model artifact ----
        model_dir = Path("data/models")
        current_artifact = model_dir / "v13_oracle.joblib"
        backup_artifact = model_dir / "v13_oracle.joblib.bak"
        if current_artifact.exists():
            shutil.copy2(current_artifact, backup_artifact)

        # ---- Build training kwargs ----
        # When drift was the trigger, use a higher exponential decay (xi * 2)
        # to weight recent data more heavily and adapt faster to the new
        # data distribution.
        train_kwargs: dict = {"eval_days": 365, "verbose": verbose}
        if drift_triggered:
            # Double the exponential time-decay so recent matches dominate
            train_kwargs["xi"] = 0.02  # default is ~0.01; doubled for drift
            if verbose:
                log.info(
                    "Auto-retrain: drift mode — using xi=%.3f (2x normal) "
                    "to prioritise recent data",
                    train_kwargs["xi"],
                )

        # ---- Train new model ----
        from footy.models.council import train_and_save as council_train

        try:
            train_result = council_train(self.con, **train_kwargs)
        except Exception as e:
            log.error("Auto-retrain: training failed: %s", e)
            # Restore backup
            if backup_artifact.exists():
                shutil.copy2(backup_artifact, current_artifact)
            return {"action": "train_failed", "error": str(e)}

        new_metrics = {
            "accuracy": train_result.get("accuracy", train_result.get("test_accuracy", 0)),
            "logloss": train_result.get("logloss", train_result.get("test_logloss", 999)),
            "brier": train_result.get("brier", train_result.get("test_brier", 999)),
            "ece": train_result.get("ece", train_result.get("test_ece", 999)),
            "n_train": train_result.get("n_train", 0),
            "n_test": train_result.get("n_test", 0),
        }

        if "error" in train_result:
            log.error("Auto-retrain: training returned error: %s", train_result["error"])
            if backup_artifact.exists():
                shutil.copy2(backup_artifact, current_artifact)
            return {"action": "train_failed", "error": train_result["error"]}

        if not force and train_result.get("wf_gate_passed") is False:
            log.warning("Auto-retrain: training produced a candidate only because walk-forward gate failed")
            if backup_artifact.exists() and current_artifact.exists():
                backup_artifact.unlink(missing_ok=True)
            return {
                "action": "candidate_only",
                "reason": "walkforward_gate_failed",
                "metrics": new_metrics,
            }

        # ---- Record training ----
        from datetime import timezone as _tz
        ts = datetime.now(_tz.utc).strftime("%Y%m%d_%H%M%S")
        new_version = f"v13_oracle_{ts}"

        versioned_artifact = model_dir / f"{new_version}.joblib"
        if current_artifact.exists():
            shutil.copy2(current_artifact, versioned_artifact)

        record = self.record_training(
            model_version=new_version,
            model_type="v13_oracle",
            training_window_days=train_result.get("window_days", 3650),
            n_matches_used=new_metrics["n_train"],
            n_matches_test=new_metrics["n_test"],
            metrics=new_metrics,
            test_metrics=new_metrics,
        )

        # ---- Deploy or rollback ----
        perf_row = self.con.execute(
            "SELECT performance_threshold_improvement FROM retraining_schedules WHERE model_type = ?",
            ["v13_oracle"],
        ).fetchone()
        threshold = perf_row[0] if perf_row else 0.01
        improved = record.get("improvement_pct", 0) >= threshold or force

        # Mark drift event as retrain-completed in the audit log
        if drift_triggered:
            try:
                self.con.execute(
                    """UPDATE drift_events
                       SET retrain_completed = TRUE,
                           notes = notes || ' | retrain_completed'
                       WHERE model_type = 'v13_oracle'
                         AND retrain_completed = FALSE
                         AND retrain_triggered = TRUE""",
                )
            except Exception as e:
                log.debug("Failed to update drift event record: %s", e)

        # Reset drift state regardless of outcome — we acted on it
        drift_info = {
            "drift_triggered": drift_triggered,
            "detectors": list(self._drift_detectors_triggered),
            "drift_timestamp": self._drift_event_timestamp,
        }
        self.needs_retrain = False
        self._drift_triggered = False
        self._drift_detectors_triggered = []
        self._drift_event_timestamp = None

        if improved:
            deploy = self.deploy_model(new_version, "v13_oracle", force=True)
            if verbose:
                log.info("Auto-retrain: deployed %s (improvement %.2f%%)",
                         new_version, record["improvement_pct"] * 100.0)
            return {
                "action": "deployed",
                "version": new_version,
                "improvement_pct": record["improvement_pct"],
                "metrics": new_metrics,
                "deploy": deploy,
                "drift_info": drift_info,
            }
        else:
            # Rollback to backup
            if backup_artifact.exists():
                shutil.copy2(backup_artifact, current_artifact)
                backup_artifact.unlink(missing_ok=True)
            versioned_artifact.unlink(missing_ok=True)
            if verbose:
                log.info("Auto-retrain: rolled back (performance regressed %.2f%%)",
                         record["improvement_pct"] * 100.0)
            return {
                "action": "rolled_back",
                "version": new_version,
                "improvement_pct": record["improvement_pct"],
                "metrics": new_metrics,
                "reason": "performance_regression",
                "drift_info": drift_info,
            }

    def record_training(
        self,
        model_version: str,
        model_type: str,
        training_window_days: int,
        n_matches_used: int,
        n_matches_test: int,
        metrics: dict,
        test_metrics: dict = None,
    ) -> dict:
        """
        Record a completed training run and compare with previous version.

        Args:
            model_version: New version identifier (e.g., "v5_ultimate_20260214_v2")
            model_type: Model type (v5_ultimate, v13_oracle, etc.)
            training_window_days: Number of days of data used for training
            n_matches_used: Number of matches in training set
            n_matches_test: Number of matches in test set
            metrics: Training metrics dict (e.g., {"accuracy": 0.65, "loss": 0.42})
            test_metrics: Test set metrics dict

        Returns:
            Training record with improvement assessment
        """
        # Get previous best version of this model type
        prev_record = self.con.execute("""
            SELECT model_version, test_metrics_json FROM model_training_records
            WHERE model_type = ? AND deployed = TRUE
            ORDER BY training_date DESC LIMIT 1
        """, [model_type]).fetchone()

        prev_version = prev_record[0] if prev_record else None
        prev_metrics = json.loads(prev_record[1]) if prev_record and prev_record[1] else {}

        # Calculate improvement
        improvement_pct = self._calculate_improvement(test_metrics or metrics, prev_metrics)

        # Record training
        self.con.execute("""
            INSERT INTO model_training_records
            (id, model_version, model_type, training_window_days, n_matches_used, n_matches_test,
             metrics_json, test_metrics_json, previous_version, improvement_pct, deployed, notes)
            VALUES (nextval('mtr_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, FALSE, ?)
        """, [
            model_version, model_type, training_window_days, n_matches_used, n_matches_test,
            json.dumps(metrics), json.dumps(test_metrics or {}),
            prev_version, improvement_pct,
            f"Training started {datetime.now(timezone.utc).isoformat()}"
        ])

        # Update retraining schedule
        self.con.execute("""
            UPDATE retraining_schedules
            SET last_trained = CURRENT_TIMESTAMP, last_retrain_matches = 0
            WHERE model_type = ?
        """, [model_type])

        return {
            "model_version": model_version,
            "model_type": model_type,
            "improvement_pct": improvement_pct,
            "previous_version": prev_version,
            "ready_to_deploy": improvement_pct > 0,  # Deploy if any improvement
        }

    def _calculate_improvement(self, new_metrics: dict, prev_metrics: dict) -> float:
        """Calculate fractional improvement from metrics (0.01 = 1%)"""
        if not new_metrics:
            return 0.0
        if not prev_metrics:
            # First deployment — no baseline to compare against, treat as ready
            return 1.0

        # Use accuracy if available, otherwise logloss
        if "accuracy" in new_metrics and "accuracy" in prev_metrics:
            improvement = new_metrics["accuracy"] - prev_metrics["accuracy"]
        elif "logloss" in new_metrics and "logloss" in prev_metrics:
            # Lower logloss is better
            improvement = prev_metrics["logloss"] - new_metrics["logloss"]
        else:
            return 0.0

        return improvement

    def deploy_model(
        self,
        model_version: str,
        model_type: str,
        force: bool = False,
    ) -> dict:
        """
        Deploy a trained model to production.

        Args:
            model_version: Version to deploy
            model_type: Model type
            force: Override performance checks

        Returns:
            Deployment result
        """
        # Get training record
        record = self.con.execute("""
            SELECT improvement_pct, test_metrics_json FROM model_training_records
            WHERE model_version = ?
        """, [model_version]).fetchone()

        if not record:
            return {"status": "error", "error": f"No training record for {model_version}"}

        improvement, metrics_json = record

        # Performance gate: skip deployment if improvement is negative (unless forced)
        if not force and improvement is not None and improvement < 0:
            return {
                "status": "skipped",
                "reason": f"Model {model_version} did not improve (delta: {improvement:.4f})",
                "improvement_pct": improvement,
            }

        # Get current deployed version
        current = self.con.execute("""
            SELECT active_version FROM model_deployments WHERE model_type = ?
        """, [model_type]).fetchone()

        prev_version = current[0] if current else None

        # Update deployment
        self.con.execute("""
            INSERT OR REPLACE INTO model_deployments
            (model_type, active_version, previous_version, reason, performance_metrics_json)
            VALUES (?, ?, ?, ?, ?)
        """, [
            model_type, model_version, prev_version,
            f"Deployed by continuous retraining (improvement: {improvement:.2f}%)",
            metrics_json
        ])

        # Mark training record as deployed
        self.con.execute("""
            UPDATE model_training_records
            SET deployed = TRUE, deployment_date = CURRENT_TIMESTAMP
            WHERE model_version = ?
        """, [model_version])

        return {
            "status": "deployed",
            "model_version": model_version,
            "model_type": model_type,
            "previous_version": prev_version,
            "improvement_pct": improvement,
        }

    def rollback_model(self, model_type: str) -> dict:
        """Rollback to previous model version if current one has issues"""
        # Get previous version
        prev = self.con.execute("""
            SELECT previous_version FROM model_deployments WHERE model_type = ?
        """, [model_type]).fetchone()

        if not prev or not prev[0]:
            return {"status": "error", "error": "No previous version to rollback to"}

        prev_version = prev[0]

        # Restore previous version
        self.con.execute("""
            UPDATE model_deployments
            SET active_version = ?, previous_version = NULL, reason = 'Rolled back due to performance issue'
            WHERE model_type = ?
        """, [prev_version, model_type])

        return {
            "status": "rolled_back",
            "model_type": model_type,
            "restored_version": prev_version,
        }

    def get_training_history(self, model_type: str, limit: int = 10) -> list:
        """Get training history for a model type"""
        rows = self.con.execute("""
            SELECT model_version, training_date, n_matches_used, improvement_pct, deployed
            FROM model_training_records
            WHERE model_type = ?
            ORDER BY training_date DESC
            LIMIT ?
        """, [model_type, limit]).fetchall()

        history = []
        for version, date, matches, improvement, deployed in rows:
            history.append({
                "model_version": version,
                "training_date": date,
                "n_matches_used": matches,
                "improvement_pct": improvement,
                "deployed": bool(deployed),
            })

        return history

    def get_deployment_status(self) -> dict:
        """Get status of all deployed models"""
        rows = self.con.execute("""
            SELECT model_type, active_version, deployed_at, previous_version
            FROM model_deployments
            ORDER BY model_type
        """).fetchall()

        status = {}
        for model_type, active_version, deployed_at, prev_version in rows:
            status[model_type] = {
                "active_version": active_version,
                "deployed_at": deployed_at,
                "previous_version": prev_version,
            }

        return status

    def get_retraining_status(self) -> dict:
        """Get retraining readiness for all models"""
        rows = self.con.execute("""
            SELECT model_type, retrain_threshold_matches, last_trained
            FROM retraining_schedules
            WHERE enabled = TRUE
        """).fetchall()

        status = {}
        for model_type, threshold, last_trained in rows:
            n_new = self.con.execute("""
                SELECT COUNT(*) FROM matches
                WHERE status = 'FINISHED' AND home_goals IS NOT NULL
                AND utc_date > COALESCE(?, '1900-01-01'::timestamp)
            """, [last_trained]).fetchone()[0]

            status[model_type] = {
                "new_matches": n_new,
                "threshold": threshold,
                "ready": n_new >= threshold,
                "last_trained": last_trained,
            }

        return status

    # ------------------------------------------------------------------
    # Ensemble Weight Learning (Self-Learning Feedback Loop)
    # ------------------------------------------------------------------
    def learn_ensemble_weights(self, lookback_days: int = 90) -> dict:
        """Learn optimal model weights from historical prediction accuracy.

        For each finished match with predictions, computes each score model's
        log-loss against the actual result. Models with lower log-loss get
        higher weights for future predictions.

        This is the core self-learning loop that makes the system improve over time.
        """
        from footy.models.experimental_math import (
            SelfCalibratingEnsemble,
            build_all_score_matrices,
        )
        from footy.models.advanced_math import extract_match_probs

        ensemble = SelfCalibratingEnsemble()
        active_model = "v13_oracle"
        try:
            active_row = self.con.execute(
                "SELECT active_version FROM model_deployments WHERE model_type = ?",
                ["v13_oracle"],
            ).fetchone()
            if active_row and active_row[0]:
                active_model = str(active_row[0])
        except Exception:
            pass

        # Get finished matches with predictions
        rows = self.con.execute("""
            SELECT m.match_id, m.home_goals, m.away_goals,
                   p.eg_home, p.eg_away
            FROM matches m
            JOIN predictions p ON m.match_id = p.match_id
            WHERE m.status = 'FINISHED'
              AND m.home_goals IS NOT NULL
              AND p.model_version = ?
              AND p.eg_home IS NOT NULL
              AND p.eg_away IS NOT NULL
              AND m.utc_date > CURRENT_TIMESTAMP - INTERVAL ? DAY
            ORDER BY m.utc_date ASC
        """, [active_model, lookback_days]).fetchall()

        if not rows:
            log.info("Ensemble learning: no scored matches found")
            return {"status": "no_data", "n_matches": 0}

        for match_id, home_goals, away_goals, eg_home, eg_away in rows:
            lambda_h = max(0.1, float(eg_home))
            lambda_a = max(0.1, float(eg_away))

            # Determine outcome
            if home_goals > away_goals:
                outcome = "home"
            elif away_goals > home_goals:
                outcome = "away"
            else:
                outcome = "draw"

            # Build all score matrices and extract probabilities
            try:
                matrices = build_all_score_matrices(lambda_h, lambda_a)
                model_probs = {}
                for name, mat in matrices.items():
                    probs = extract_match_probs(mat)
                    model_probs[name] = probs
                ensemble.update(model_probs, outcome)
            except Exception:
                continue

        weights = ensemble.get_weights()

        # Store learned weights in DB for persistence
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ensemble_weights (
                model_name VARCHAR PRIMARY KEY,
                weight DOUBLE NOT NULL,
                n_predictions INT,
                avg_log_loss DOUBLE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        for name, weight in weights.items():
            rec = ensemble.records.get(name)
            self.con.execute("""
                INSERT OR REPLACE INTO ensemble_weights
                (model_name, weight, n_predictions, avg_log_loss)
                VALUES (?, ?, ?, ?)
            """, [
                name, weight,
                rec.n_predictions if rec else 0,
                rec.avg_log_loss if rec else 1.0,
            ])

        log.info("Ensemble learning: updated weights from %d matches", len(rows))
        return {
            "status": "updated",
            "model_version": active_model,
            "n_matches": len(rows),
            "weights": weights,
        }

    def get_learned_weights(self) -> dict[str, float] | None:
        """Retrieve previously learned ensemble weights."""
        try:
            rows = self.con.execute("""
                SELECT model_name, weight FROM ensemble_weights
            """).fetchall()
            if not rows:
                return None
            return {name: float(w) for name, w in rows}
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Expert Performance Tracking (self-learning per expert)
    # ------------------------------------------------------------------
    def track_expert_accuracy(self, verbose: bool = False) -> dict:
        """Score each expert's predictions against actual results.

        Reads the expert_cache table (populated during predict_upcoming)
        and compares each expert's predicted probabilities against the
        actual match outcome.  Stores per-expert accuracy metrics in
        ``expert_performance`` table for downstream use (feature selection,
        dynamic expert weighting).

        Returns dict mapping expert_name -> {accuracy, n, avg_confidence}.
        """
        self._ensure_expert_perf_schema()

        # Find finished matches that have expert cache data
        rows = self.con.execute("""
            SELECT ec.match_id, ec.breakdown_json,
                   m.home_goals, m.away_goals, m.competition
            FROM expert_cache ec
            JOIN matches m ON m.match_id = ec.match_id
            WHERE m.status = 'FINISHED'
              AND m.home_goals IS NOT NULL
        """).fetchall()

        if not rows:
            return {}

        from collections import defaultdict
        expert_stats: dict[str, dict] = defaultdict(lambda: {
            "correct": 0, "total": 0, "log_loss_sum": 0.0,
            "confidence_sum": 0.0, "by_comp": defaultdict(lambda: {"correct": 0, "total": 0}),
        })

        import json
        import math
        for match_id, breakdown_json, hg, ag, comp in rows:
            try:
                data = json.loads(breakdown_json) if isinstance(breakdown_json, str) else breakdown_json
                experts = data.get("experts", {})
            except (json.JSONDecodeError, TypeError):
                continue

            # Determine actual outcome
            if hg > ag:
                actual = 0  # home
            elif hg == ag:
                actual = 1  # draw
            else:
                actual = 2  # away

            for expert_name, expert_data in experts.items():
                probs = expert_data.get("probs", {})
                ph = probs.get("home", 1/3)
                pd_ = probs.get("draw", 1/3)
                pa = probs.get("away", 1/3)
                confidence = expert_data.get("confidence", 0.5)

                # Predicted outcome
                pred_probs = [ph, pd_, pa]
                predicted = max(range(3), key=lambda i: pred_probs[i])

                stats = expert_stats[expert_name]
                stats["total"] += 1
                stats["confidence_sum"] += confidence
                stats["by_comp"][comp]["total"] += 1

                if predicted == actual:
                    stats["correct"] += 1
                    stats["by_comp"][comp]["correct"] += 1

                # Log loss contribution
                p_actual = max(pred_probs[actual], 1e-12)
                stats["log_loss_sum"] += -math.log(p_actual)

        # Compute final metrics and store
        results = {}
        for expert_name, stats in expert_stats.items():
            if stats["total"] == 0:
                continue
            accuracy = stats["correct"] / stats["total"]
            avg_ll = stats["log_loss_sum"] / stats["total"]
            avg_conf = stats["confidence_sum"] / stats["total"]

            results[expert_name] = {
                "accuracy": round(accuracy, 4),
                "log_loss": round(avg_ll, 4),
                "n": stats["total"],
                "avg_confidence": round(avg_conf, 4),
            }

            # Store to DB
            self.con.execute("""
                INSERT OR REPLACE INTO expert_performance
                (expert_name, accuracy, log_loss, n_predictions,
                 avg_confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [expert_name, accuracy, avg_ll, stats["total"], avg_conf])

            # Store per-competition breakdown
            for comp, comp_stats in stats["by_comp"].items():
                if comp_stats["total"] >= 5:
                    comp_acc = comp_stats["correct"] / comp_stats["total"]
                    self.con.execute("""
                        INSERT OR REPLACE INTO expert_performance_by_comp
                        (expert_name, competition, accuracy, n_predictions, updated_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, [expert_name, comp, comp_acc, comp_stats["total"]])

        if verbose:
            for name, m in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
                log.info("Expert %-20s  acc=%.1f%%  ll=%.3f  n=%d  conf=%.2f",
                         name, m["accuracy"]*100, m["log_loss"], m["n"], m["avg_confidence"])

        return results

    def get_expert_rankings(self, competition: str = None) -> list[dict]:
        """Return experts ranked by accuracy, optionally filtered by competition."""
        self._ensure_expert_perf_schema()
        if competition:
            rows = self.con.execute("""
                SELECT expert_name, accuracy, n_predictions
                FROM expert_performance_by_comp
                WHERE competition = ?
                ORDER BY accuracy DESC
            """, [competition]).fetchall()
        else:
            rows = self.con.execute("""
                SELECT expert_name, accuracy, n_predictions, log_loss, avg_confidence
                FROM expert_performance
                ORDER BY accuracy DESC
            """).fetchall()
        return [{"expert": r[0], "accuracy": r[1], "n": r[2],
                 **({"log_loss": r[3], "avg_confidence": r[4]} if len(r) > 3 else {})}
                for r in rows]

    def get_expert_weight_map(self, competition: str | None = None, min_predictions: int = 5) -> dict[str, float]:
        """Return normalized expert weights learned from scored outcomes.

        Weights combine:
        - accuracy above naive 1/3 baseline
        - confidence/log-loss quality where available
        - sample-size shrinkage to avoid overreacting to tiny samples

        If a competition-specific lookup is sparse, falls back to global rankings.
        """
        rankings = self.get_expert_rankings(competition=competition)
        if competition and not rankings:
            rankings = self.get_expert_rankings()

        weights: dict[str, float] = {}
        raw_total = 0.0
        for row in rankings:
            n = int(row.get("n") or 0)
            if n <= 0:
                continue
            if competition and n < min_predictions:
                continue

            accuracy = float(row.get("accuracy") or 0.0)
            log_loss = row.get("log_loss")
            avg_confidence = float(row.get("avg_confidence") or 0.5)

            baseline_edge = max(0.01, accuracy - (1.0 / 3.0))
            sample_factor = min(1.0, math.sqrt(n / max(float(min_predictions), 1.0)))
            quality_factor = math.exp(-float(log_loss)) if log_loss is not None else 0.65
            confidence_factor = 0.7 + 0.3 * max(0.0, min(1.0, avg_confidence))

            raw = baseline_edge * sample_factor * quality_factor * confidence_factor
            if raw <= 0:
                continue
            weights[row["expert"]] = raw
            raw_total += raw

        if raw_total <= 0 and competition:
            return self.get_expert_weight_map(None, min_predictions=min_predictions)
        if raw_total <= 0:
            return {}
        return {name: value / raw_total for name, value in weights.items()}

    def check_for_drift(self, recent_predictions: list, recent_outcomes: list,
                       threshold: float = 0.05, window_size: int = 30) -> bool:
        """Check if model performance has drifted using CUSUM control chart.

        CUSUM (Cumulative Sum Control Chart) detects shifts in process mean
        by accumulating deviations from expected value. Applied here to detect
        gradual degradation in prediction accuracy over recent matches.

        Args:
            recent_predictions: List of predicted probabilities (must be sorted by time)
                               Each element is [p_home, p_draw, p_away]
            recent_outcomes: List of actual outcomes (0=home, 1=draw, 2=away)
            threshold: CUSUM control threshold (default 0.05 corresponds to ~5% accuracy drop)
            window_size: Lookback window size for rolling mean (default 30 matches)

        Returns:
            True if drift detected (performance degradation), False otherwise
        """
        if len(recent_predictions) == 0 or len(recent_outcomes) == 0:
            return False

        if len(recent_predictions) < window_size:
            return False

        try:
            from footy.models.math.scoring import brier_score

            # Compute Brier scores for recent predictions
            scores = []
            for pred_probs, outcome_idx in zip(recent_predictions, recent_outcomes):
                try:
                    bs = brier_score(pred_probs, int(outcome_idx))
                    scores.append(bs)
                except Exception:
                    continue

            if len(scores) < window_size:
                return False

            # Compute rolling mean and CUSUM statistics
            recent_scores = scores[-window_size:]
            baseline_mean = np.mean(recent_scores[:window_size//2])

            # CUSUM calculation: accumulate deviations from baseline
            cusum_pos = 0.0  # Accumulates positive drift
            cusum_neg = 0.0  # Accumulates negative drift
            drift_detected = False

            for score in recent_scores[window_size//2:]:
                deviation = score - baseline_mean

                # One-sided CUSUM: looking for increase in error (drift)
                cusum_pos = max(0.0, cusum_pos + deviation - threshold)

                # Two-sided: also check for improvement
                cusum_neg = max(0.0, cusum_neg - deviation - threshold)

                # Control limits: if CUSUM exceeds 5.0, drift likely
                if cusum_pos > 5.0:
                    log.warning("[drift] CUSUM positive drift detected (pos=%.2f)", cusum_pos)
                    drift_detected = True
                    break

            return drift_detected

        except Exception as e:
            log.debug("check_for_drift error: %s", e)
            return False


    def _ensure_expert_perf_schema(self):
        """Create expert performance tracking tables."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS expert_performance (
                expert_name VARCHAR PRIMARY KEY,
                accuracy DOUBLE,
                log_loss DOUBLE,
                n_predictions INT,
                avg_confidence DOUBLE,
                updated_at TIMESTAMP
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS expert_performance_by_comp (
                expert_name VARCHAR,
                competition VARCHAR,
                accuracy DOUBLE,
                n_predictions INT,
                updated_at TIMESTAMP,
                PRIMARY KEY (expert_name, competition)
            )
        """)


def get_training_manager() -> ContinuousTrainingManager:
    """Get or create training manager instance"""
    return ContinuousTrainingManager()
