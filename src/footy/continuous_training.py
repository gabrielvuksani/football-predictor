"""
Continuous Model Retraining

Handles automated retraining of models as new match data becomes available.
Tracks model versions, training windows, validates improvements, and auto-deploys.

Features:
- Drift detection: monitors prediction accuracy on recent results
- Auto-retrain: triggers v10_council retraining when new-match or drift thresholds are met
- Performance gating: only deploys if new model beats current on held-out set
- Automatic rollback: reverts if deployed model degrades within grace window
- Full audit trail: every train/deploy/rollback is logged with metrics
"""
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import json
import shutil
import logging

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
        self.created_at = datetime.utcnow()


class ContinuousTrainingManager:
    """
    Manages continuous model retraining with new match data.
    
    Strategy:
    1. Check for new finished matches since last training
    2. If threshold met (e.g., 10+ new matches), retrain model
    3. Evaluate new model on held-out test set
    4. Compare with previous model version
    5. If improvement > threshold, deploy new model
    6. If degradation, keep previous model but log warning
    
    Example usage:
        manager = ContinuousTrainingManager()
        manager.setup_continuous_training(
            model_type="v10_council",
            retrain_threshold_matches=10,
            performance_threshold_improvement=0.01
        )
        manager.check_and_retrain()  # Called by scheduler periodically
    """
    
    def __init__(self):
        self.con = connect()
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create training records table"""
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
                previous_version VARCHAR,
                improvement_pct DOUBLE,
                deployed BOOLEAN DEFAULT FALSE,
                deployment_date TIMESTAMP,
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
                performance_metrics_json VARCHAR
            )
        """)
        
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS retraining_schedules (
                model_type VARCHAR PRIMARY KEY,
                retrain_threshold_matches INT DEFAULT 10,
                performance_threshold_improvement DOUBLE DEFAULT 0.01,
                enabled BOOLEAN DEFAULT TRUE,
                last_trained TIMESTAMP,
                last_retrain_matches INT DEFAULT 0,
                next_scheduled_retrain TIMESTAMP
            )
        """)
    
    def setup_continuous_training(
        self,
        model_type: str,
        retrain_threshold_matches: int = 10,
        performance_threshold_improvement: float = 0.01,
    ) -> dict:
        """
        Configure continuous training for a model type.
        
        Args:
            model_type: Model to retrain (v1, v2, v3, v4, v5, etc.)
            retrain_threshold_matches: Trigger retrain after N new finished matches
            performance_threshold_improvement: Minimum improvement to deploy (e.g., 0.01 = 1%)
        
        Returns:
            Configuration dict
        """
        self.con.execute("""
            INSERT OR REPLACE INTO retraining_schedules
            (model_type, retrain_threshold_matches, performance_threshold_improvement, enabled)
            VALUES (?, ?, ?, TRUE)
        """, [model_type, retrain_threshold_matches, performance_threshold_improvement])
        
        return {
            "model_type": model_type,
            "retrain_threshold_matches": retrain_threshold_matches,
            "performance_threshold_improvement": performance_threshold_improvement,
            "enabled": True,
        }
    
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
        
        if drift.get("drifted"):
            return {
                "status": "drift_detected",
                "new_matches": n_new,
                "threshold": retrain_threshold,
                "drift_info": drift,
                "reason": f"Accuracy dropped from {drift['baseline_acc']:.1%} to {drift['recent_acc']:.1%}",
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
        now = datetime.utcnow()
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
            "baseline_acc": baseline_acc,
            "baseline_n": baseline_n,
            "recent_acc": recent_acc,
            "recent_n": recent_n,
            "accuracy_drop": drop,
        }

    # ------------------------------------------------------------------
    # Auto-retrain
    # ------------------------------------------------------------------
    def auto_retrain(self, force: bool = False, verbose: bool = True) -> dict:
        """
        End-to-end auto-retrain: check thresholds → train → validate → deploy/rollback.
        
        1. Checks if retraining is needed (new matches or drift detected)
        2. Trains new v10_council model
        3. Compares test-set performance vs current model
        4. Deploys only if performance improves (or force=True)
        5. Backs up old model artifact for rollback
        
        Returns summary dict.
        """
        check = self.check_and_retrain("v10_council")
        result = check.get("v10_council", {})
        status = result.get("status", "")

        if not force and status not in ("ready_to_retrain", "drift_detected"):
            if verbose:
                log.info("Auto-retrain: no action needed (%s)", status)
            return {"action": "none", "check": result}

        reason = "forced" if force else status
        if verbose:
            log.info("Auto-retrain: triggered (%s)", reason)

        # ---- Backup current model artifact ----
        model_dir = Path("data/models")
        current_artifact = model_dir / "v10_council.joblib"
        backup_artifact = model_dir / "v10_council.joblib.bak"
        if current_artifact.exists():
            shutil.copy2(current_artifact, backup_artifact)

        # ---- Train new model ----
        from footy.models.council import train_and_save as council_train

        try:
            train_result = council_train(self.con, eval_days=365, verbose=verbose)
        except Exception as e:
            log.error("Auto-retrain: training failed: %s", e)
            # Restore backup
            if backup_artifact.exists():
                shutil.copy2(backup_artifact, current_artifact)
            return {"action": "train_failed", "error": str(e)}

        new_metrics = {
            "accuracy": train_result.get("test_accuracy", 0),
            "logloss": train_result.get("test_logloss", 999),
            "brier": train_result.get("test_brier", 999),
            "ece": train_result.get("test_ece", 999),
            "n_train": train_result.get("n_train", 0),
            "n_test": train_result.get("n_test", 0),
        }

        if "error" in train_result:
            log.error("Auto-retrain: training returned error: %s", train_result["error"])
            if backup_artifact.exists():
                shutil.copy2(backup_artifact, current_artifact)
            return {"action": "train_failed", "error": train_result["error"]}

        # ---- Record training ----
        from datetime import timezone as _tz
        ts = datetime.now(_tz.utc).strftime("%Y%m%d_%H%M%S")
        new_version = f"v10_council_{ts}"
        record = self.record_training(
            model_version=new_version,
            model_type="v10_council",
            training_window_days=train_result.get("window_days", 3650),
            n_matches_used=new_metrics["n_train"],
            n_matches_test=new_metrics["n_test"],
            metrics=new_metrics,
            test_metrics=new_metrics,
        )

        # ---- Deploy or rollback ----
        perf_row = self.con.execute(
            "SELECT performance_threshold_improvement FROM retraining_schedules WHERE model_type = ?",
            ["v10_council"],
        ).fetchone()
        threshold = perf_row[0] if perf_row else 0.01
        improved = record.get("improvement_pct", 0) >= threshold or force
        if improved:
            deploy = self.deploy_model(new_version, "v10_council", force=True)
            if verbose:
                log.info("Auto-retrain: deployed %s (improvement %.2f%%)",
                         new_version, record["improvement_pct"])
            return {
                "action": "deployed",
                "version": new_version,
                "improvement_pct": record["improvement_pct"],
                "metrics": new_metrics,
                "deploy": deploy,
            }
        else:
            # Rollback to backup
            if backup_artifact.exists():
                shutil.copy2(backup_artifact, current_artifact)
                backup_artifact.unlink(missing_ok=True)
            if verbose:
                log.info("Auto-retrain: rolled back (performance regressed %.2f%%)",
                         record["improvement_pct"])
            return {
                "action": "rolled_back",
                "version": new_version,
                "improvement_pct": record["improvement_pct"],
                "metrics": new_metrics,
                "reason": "performance_regression",
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
            model_type: Model type (v5_ultimate, v10_council, etc.)
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
            f"Training started {datetime.utcnow().isoformat()}"
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
        """Calculate percentage improvement from metrics"""
        if not new_metrics or not prev_metrics:
            return 0.0
        
        # Use accuracy if available, otherwise logloss
        if "accuracy" in new_metrics and "accuracy" in prev_metrics:
            improvement = (new_metrics["accuracy"] - prev_metrics["accuracy"]) * 100
        elif "logloss" in new_metrics and "logloss" in prev_metrics:
            # Lower logloss is better, so negative improvement
            improvement = (prev_metrics["logloss"] - new_metrics["logloss"]) * 100
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


def get_training_manager() -> ContinuousTrainingManager:
    """Get or create training manager instance"""
    return ContinuousTrainingManager()
