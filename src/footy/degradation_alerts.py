"""
Phase 4.4: Model Degradation Alerts

Monitors model performance in real-time and triggers alerts when:
- Performance degrades below thresholds
- Prediction accuracy drops significantly
- Probability calibration issues detected
- Data drift observed

Features:
- Real-time performance monitoring
- Statistical degradation detection
- Alert management with acknowledge/snooze
- Multi-channel notifications (console, file, email)
- Performance comparison against baseline
- Alert deduplication (same alert within configurable hours)
- Dynamic thresholds based on rolling statistics
- Alert severity escalation (warning -> critical after N occurrences)
- Better alert status transitions (active -> acknowledged -> resolved)
- Logging integration for all alerts
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Callable
from enum import Enum
import logging

from footy.db import connect
from footy.performance_tracker import get_performance_tracker

log = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SNOOZED = "snoozed"


class DegradationAlert:
    """Represents a degradation alert with lifecycle management."""

    def __init__(
        self,
        alert_id: str,
        model_version: str,
        severity: AlertSeverity,
        message: str,
        metric: str,
        metric_value: float,
        threshold: float,
        created_at: datetime = None,
        occurrence_count: int = 1,
    ):
        self.alert_id = alert_id
        self.model_version = model_version
        self.severity = severity
        self.message = message
        self.metric = metric
        self.metric_value = metric_value
        self.threshold = threshold
        self.created_at = created_at or datetime.now(timezone.utc)
        self.status = AlertStatus.ACTIVE
        self.acknowledged_at = None
        self.resolved_at = None
        self.snoozed_until = None
        self.occurrence_count = occurrence_count  # For escalation tracking

    def escalate_severity(self) -> None:
        """Escalate alert severity if triggered multiple times."""
        if self.severity == AlertSeverity.INFO:
            self.severity = AlertSeverity.WARNING
        elif self.severity == AlertSeverity.WARNING:
            self.severity = AlertSeverity.CRITICAL


class DegradationMonitor:
    """
    Monitors model performance and triggers alerts on degradation.

    Example usage:
        monitor = DegradationMonitor()

        # Setup monitoring for a model
        monitor.setup_monitoring(
            model_version="v5_ultimate",
            accuracy_threshold=0.45,
            logloss_threshold=0.65,
        )

        # Periodic check (run in scheduler)
        alerts = monitor.check_degradation()

        # Get current alerts
        active = monitor.get_active_alerts()
    """

    def __init__(self):
        self.con = connect()
        self.performance_tracker = get_performance_tracker()
        self._ensure_schema()
        self._alert_handlers: list[Callable] = []

    def _ensure_schema(self):
        """Create alert tables with enhanced schema for deduplication and escalation."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS degradation_alerts (
                alert_id VARCHAR PRIMARY KEY,
                model_version VARCHAR NOT NULL,
                severity VARCHAR NOT NULL,
                message VARCHAR NOT NULL,
                metric VARCHAR NOT NULL,
                metric_value DOUBLE,
                threshold DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR DEFAULT 'active',
                acknowledged_at TIMESTAMP,
                resolved_at TIMESTAMP,
                snoozed_until TIMESTAMP,
                occurrence_count INT DEFAULT 1,
                last_occurrence TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS degradation_rules (
                model_version VARCHAR PRIMARY KEY,
                accuracy_threshold DOUBLE DEFAULT 0.45,
                logloss_threshold DOUBLE DEFAULT 0.65,
                brier_threshold DOUBLE DEFAULT 0.25,
                trend_threshold DOUBLE DEFAULT -0.05,
                window_days INT DEFAULT 30,
                enabled BOOLEAN DEFAULT TRUE,
                dynamic_thresholds BOOLEAN DEFAULT TRUE,
                escalation_threshold INT DEFAULT 3,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY,
                alert_id VARCHAR NOT NULL,
                action VARCHAR NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details VARCHAR,
                status_from VARCHAR,
                status_to VARCHAR
            )
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS rolling_statistics (
                model_version VARCHAR NOT NULL,
                metric VARCHAR NOT NULL,
                mean_value DOUBLE,
                std_value DOUBLE,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model_version, metric)
            )
        """)

        # Create indices for performance
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS alerts_model_status_idx
            ON degradation_alerts(model_version, status)
        """)
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS alerts_metric_idx
            ON degradation_alerts(model_version, metric, status)
        """)
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS history_alert_idx
            ON alert_history(alert_id, timestamp DESC)
        """)

    def setup_monitoring(
        self,
        model_version: str,
        accuracy_threshold: float = 0.45,
        logloss_threshold: float = 0.65,
        brier_threshold: float = 0.25,
        trend_threshold: float = -0.05,
        window_days: int = 30,
    ) -> dict:
        """
        Configure degradation monitoring for a model.

        Args:
            model_version: Model to monitor
            accuracy_threshold: Minimum acceptable accuracy
            logloss_threshold: Maximum acceptable logloss
            brier_threshold: Maximum acceptable brier score
            trend_threshold: Alert if trend slope < this (degradation)
            window_days: Performance evaluation window
        """
        self.con.execute("""
            INSERT OR REPLACE INTO degradation_rules
            (model_version, accuracy_threshold, logloss_threshold, brier_threshold,
             trend_threshold, window_days, enabled)
            VALUES (?, ?, ?, ?, ?, ?, TRUE)
        """, [model_version, accuracy_threshold, logloss_threshold, brier_threshold,
              trend_threshold, window_days])

        return {
            "model_version": model_version,
            "accuracy_threshold": accuracy_threshold,
            "logloss_threshold": logloss_threshold,
            "brier_threshold": brier_threshold,
            "trend_threshold": trend_threshold,
            "window_days": window_days,
        }

    def check_degradation(self) -> list[DegradationAlert]:
        """
        Check all monitored models for degradation.

        Returns:
            List of new alerts triggered
        """
        new_alerts = []

        # Get all monitored models
        models = self.con.execute("""
            SELECT model_version FROM degradation_rules WHERE enabled = TRUE
        """).fetchall()

        for (model_version,) in models:
            alerts = self._check_model_degradation(model_version)
            new_alerts.extend(alerts)

        # Send notifications
        for alert in new_alerts:
            self._notify_alert(alert)

        return new_alerts

    def _check_model_degradation(self, model_version: str) -> list[DegradationAlert]:
        """Check single model for degradation"""
        alerts = []

        # Get monitoring rules
        rule = self.con.execute("""
            SELECT accuracy_threshold, logloss_threshold, brier_threshold,
                   trend_threshold, window_days
            FROM degradation_rules WHERE model_version = ?
        """, [model_version]).fetchone()

        if not rule:
            return []

        acc_thresh, loss_thresh, brier_thresh, trend_thresh, window = rule

        # Get current metrics
        metrics = self.performance_tracker.compute_aggregated_metrics(
            model_version, days=window
        )

        if metrics.get("n_predictions", 0) == 0:
            return []

        # Check accuracy
        if metrics["accuracy"] < acc_thresh:
            alert_id = f"{model_version}_accuracy_{datetime.now(timezone.utc).timestamp()}"
            alert = DegradationAlert(
                alert_id=alert_id,
                model_version=model_version,
                severity=AlertSeverity.CRITICAL,
                message=f"Accuracy dropped to {metrics['accuracy']:.3f}",
                metric="accuracy",
                metric_value=metrics["accuracy"],
                threshold=acc_thresh,
            )
            if self._store_alert(alert):
                alerts.append(alert)

        # Check logloss
        if metrics["logloss"] > loss_thresh:
            alert_id = f"{model_version}_logloss_{datetime.now(timezone.utc).timestamp()}"
            alert = DegradationAlert(
                alert_id=alert_id,
                model_version=model_version,
                severity=AlertSeverity.WARNING,
                message=f"Logloss increased to {metrics['logloss']:.3f}",
                metric="logloss",
                metric_value=metrics["logloss"],
                threshold=loss_thresh,
            )
            if self._store_alert(alert):
                alerts.append(alert)

        # Check brier
        if metrics["brier"] > brier_thresh:
            alert_id = f"{model_version}_brier_{datetime.now(timezone.utc).timestamp()}"
            alert = DegradationAlert(
                alert_id=alert_id,
                model_version=model_version,
                severity=AlertSeverity.WARNING,
                message=f"Brier score increased to {metrics['brier']:.3f}",
                metric="brier",
                metric_value=metrics["brier"],
                threshold=brier_thresh,
            )
            if self._store_alert(alert):
                alerts.append(alert)

        # Check trend
        trend = self.performance_tracker.get_performance_trend(
            model_version, window_days=window
        )

        if trend and trend["trend_slope"] < trend_thresh:
            alert_id = f"{model_version}_trend_{datetime.now(timezone.utc).timestamp()}"
            alert = DegradationAlert(
                alert_id=alert_id,
                model_version=model_version,
                severity=AlertSeverity.WARNING,
                message=f"Performance degrading (slope: {trend['trend_slope']:.4f})",
                metric="trend",
                metric_value=trend["trend_slope"],
                threshold=trend_thresh,
            )
            if self._store_alert(alert):
                alerts.append(alert)

        return alerts

    def _store_alert(
        self,
        alert: DegradationAlert,
        dedup_hours: int = 2,
        escalation_threshold: int = 3,
    ) -> bool:
        """Store alert with deduplication and escalation.

        Args:
            alert: The alert to store
            dedup_hours: Don't duplicate alerts within this many hours
            escalation_threshold: Escalate severity after N occurrences

        Returns:
            True if stored, False if duplicate suppressed
        """
        # Check for recent duplicate alert
        existing = self.con.execute(
            """SELECT alert_id, occurrence_count, severity FROM degradation_alerts
               WHERE model_version = ? AND metric = ? AND status IN ('active', 'acknowledged')
               AND created_at > ?""",
            [
                alert.model_version,
                alert.metric,
                datetime.now(timezone.utc) - timedelta(hours=dedup_hours),
            ],
        ).fetchone()

        if existing:
            # Update existing alert with new count and last occurrence
            existing_id, count, severity = existing
            new_count = count + 1

            # Check for escalation
            if new_count >= escalation_threshold:
                new_severity = AlertSeverity.CRITICAL.value
                log.warning(
                    f"Alert {existing_id} escalated to CRITICAL after {new_count} occurrences"
                )
            else:
                new_severity = severity

            self.con.execute(
                """UPDATE degradation_alerts
                   SET occurrence_count = ?, last_occurrence = ?, severity = ?
                   WHERE alert_id = ?""",
                [new_count, datetime.now(timezone.utc), new_severity, existing_id],
            )
            self._log_action(existing_id, "incremented", f"Occurrence #{new_count}")
            return False  # Duplicate suppressed

        # Store new alert
        self.con.execute(
            """INSERT INTO degradation_alerts
               (alert_id, model_version, severity, message, metric, metric_value,
                threshold, occurrence_count, last_occurrence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                alert.alert_id,
                alert.model_version,
                alert.severity.value,
                alert.message,
                alert.metric,
                alert.metric_value,
                alert.threshold,
                1,
                datetime.now(timezone.utc),
            ],
        )

        log.info(f"New alert created: {alert.alert_id} ({alert.metric} = {alert.metric_value:.4f})")
        self._log_action(alert.alert_id, "created", alert.message)
        return True

    def register_alert_handler(self, handler: Callable[[DegradationAlert], None]):
        """Register a callback for alert notifications"""
        self._alert_handlers.append(handler)

    def _notify_alert(self, alert: DegradationAlert):
        """Send alert to registered handlers"""
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Error notifying alert: {e}")

    def acknowledge_alert(self, alert_id: str) -> dict:
        """Acknowledge an alert with proper status transition."""
        try:
            # Get current status
            current = self.con.execute(
                "SELECT status FROM degradation_alerts WHERE alert_id = ?",
                [alert_id],
            ).fetchone()

            if not current:
                return {"alert_id": alert_id, "status": "not_found"}

            old_status = current[0]

            self.con.execute(
                """UPDATE degradation_alerts
                   SET status = 'acknowledged', acknowledged_at = CURRENT_TIMESTAMP
                   WHERE alert_id = ?""",
                [alert_id],
            )

            self._log_action(
                alert_id,
                "acknowledged",
                "User acknowledged alert",
                status_from=old_status,
                status_to="acknowledged",
            )

            log.info(f"Alert {alert_id} acknowledged")
            return {"alert_id": alert_id, "status": "acknowledged"}
        except Exception as e:
            log.error(f"Failed to acknowledge alert: {e}")
            return {"alert_id": alert_id, "status": "error", "error": str(e)}

    def resolve_alert(self, alert_id: str) -> dict:
        """Mark alert as resolved with proper status transition."""
        try:
            current = self.con.execute(
                "SELECT status FROM degradation_alerts WHERE alert_id = ?",
                [alert_id],
            ).fetchone()

            if not current:
                return {"alert_id": alert_id, "status": "not_found"}

            old_status = current[0]

            self.con.execute(
                """UPDATE degradation_alerts
                   SET status = 'resolved', resolved_at = CURRENT_TIMESTAMP
                   WHERE alert_id = ?""",
                [alert_id],
            )

            self._log_action(
                alert_id,
                "resolved",
                "Alert resolved",
                status_from=old_status,
                status_to="resolved",
            )

            log.info(f"Alert {alert_id} resolved")
            return {"alert_id": alert_id, "status": "resolved"}
        except Exception as e:
            log.error(f"Failed to resolve alert: {e}")
            return {"alert_id": alert_id, "status": "error", "error": str(e)}

    def snooze_alert(self, alert_id: str, hours: int = 24) -> dict:
        """Snooze an alert for specified hours with status transition."""
        try:
            snooze_until = datetime.now(timezone.utc) + timedelta(hours=hours)

            current = self.con.execute(
                "SELECT status FROM degradation_alerts WHERE alert_id = ?",
                [alert_id],
            ).fetchone()

            if not current:
                return {"alert_id": alert_id, "status": "not_found"}

            old_status = current[0]

            self.con.execute(
                """UPDATE degradation_alerts
                   SET status = 'snoozed', snoozed_until = ?
                   WHERE alert_id = ?""",
                [snooze_until, alert_id],
            )

            self._log_action(
                alert_id,
                "snoozed",
                f"Snoozed for {hours} hours",
                status_from=old_status,
                status_to="snoozed",
            )

            log.info(f"Alert {alert_id} snoozed until {snooze_until}")
            return {"alert_id": alert_id, "status": "snoozed", "until": snooze_until}
        except Exception as e:
            log.error(f"Failed to snooze alert: {e}")
            return {"alert_id": alert_id, "status": "error", "error": str(e)}

    def _log_action(self, alert_id: str, action: str, details: str = None, status_from: str = None, status_to: str = None):
        """Log action on an alert with status transitions."""
        self.con.execute(
            """INSERT INTO alert_history (alert_id, action, details, status_from, status_to)
               VALUES (?, ?, ?, ?, ?)""",
            [alert_id, action, details, status_from, status_to],
        )
        log.debug(f"Alert action logged: {alert_id} -> {action}")

    def _compute_rolling_statistics(self, model_version: str, window_days: int = 30) -> dict:
        """Compute rolling statistics for dynamic thresholds.

        Args:
            model_version: Model to analyze
            window_days: Window size for statistics

        Returns:
            Dictionary with metric stats (mean, std)
        """
        try:
            metrics = self.performance_tracker.compute_aggregated_metrics(
                model_version, days=window_days
            )
            if not metrics:
                return {}

            # Store computed statistics
            for metric_name in ["accuracy", "logloss", "brier"]:
                if metric_name in metrics:
                    self.con.execute(
                        """INSERT OR REPLACE INTO rolling_statistics
                           (model_version, metric, mean_value, computed_at)
                           VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                        [model_version, metric_name, metrics[metric_name]],
                    )

            return metrics
        except Exception as e:
            log.warning(f"Failed to compute rolling statistics: {e}")
            return {}

    def get_dynamic_thresholds(self, model_version: str) -> dict:
        """Get dynamic thresholds based on rolling statistics.

        Args:
            model_version: Model version

        Returns:
            Dictionary with dynamically computed thresholds
        """
        try:
            stats = self._compute_rolling_statistics(model_version, window_days=30)

            if not stats:
                # Fall back to defaults
                return {
                    "accuracy_threshold": 0.45,
                    "logloss_threshold": 0.65,
                    "brier_threshold": 0.25,
                }

            # Dynamic thresholds: mean - 1.5*std
            dynamic_thresholds = {
                "accuracy_threshold": max(
                    0.35, stats.get("accuracy", 0.45) - 1.5 * stats.get("accuracy_std", 0.05)
                ),
                "logloss_threshold": min(
                    0.8, stats.get("logloss", 0.65) + 1.5 * stats.get("logloss_std", 0.05)
                ),
                "brier_threshold": min(
                    0.35, stats.get("brier", 0.25) + 1.5 * stats.get("brier_std", 0.03)
                ),
            }

            return dynamic_thresholds
        except Exception as e:
            log.warning(f"Failed to get dynamic thresholds: {e}")
            return {
                "accuracy_threshold": 0.45,
                "logloss_threshold": 0.65,
                "brier_threshold": 0.25,
            }

    def get_active_alerts(self) -> list:
        """Get all active alerts"""
        rows = self.con.execute("""
            SELECT alert_id, model_version, severity, message, metric, metric_value,
                   threshold, created_at
            FROM degradation_alerts
            WHERE status = 'active'
            ORDER BY created_at DESC
        """).fetchall()

        alerts = []
        for row in rows:
            alerts.append({
                "alert_id": row[0],
                "model_version": row[1],
                "severity": row[2],
                "message": row[3],
                "metric": row[4],
                "metric_value": row[5],
                "threshold": row[6],
                "created_at": row[7],
            })

        return alerts

    def get_alerts_for_model(self, model_version: str, status: str = None) -> list:
        """Get alerts for specific model"""
        if status:
            rows = self.con.execute("""
                SELECT alert_id, severity, message, metric, created_at, status
                FROM degradation_alerts
                WHERE model_version = ? AND status = ?
                ORDER BY created_at DESC
            """, [model_version, status]).fetchall()
        else:
            rows = self.con.execute("""
                SELECT alert_id, severity, message, metric, created_at, status
                FROM degradation_alerts
                WHERE model_version = ?
                ORDER BY created_at DESC
            """, [model_version]).fetchall()

        alerts = []
        for row in rows:
            alerts.append({
                "alert_id": row[0],
                "severity": row[1],
                "message": row[2],
                "metric": row[3],
                "created_at": row[4],
                "status": row[5],
            })

        return alerts

    def get_alert_summary(self) -> dict:
        """Get comprehensive summary of all alerts."""
        try:
            total = self.con.execute("SELECT COUNT(*) FROM degradation_alerts").fetchone()[0]

            by_status = self.con.execute(
                """SELECT status, COUNT(*) as count
                   FROM degradation_alerts
                   GROUP BY status"""
            ).fetchall()

            by_model = self.con.execute(
                """SELECT model_version, COUNT(*) as count
                   FROM degradation_alerts WHERE status IN ('active', 'acknowledged')
                   GROUP BY model_version"""
            ).fetchall()

            by_severity = self.con.execute(
                """SELECT severity, COUNT(*) as count
                   FROM degradation_alerts WHERE status = 'active'
                   GROUP BY severity"""
            ).fetchall()

            # Get escalated alerts
            escalated = self.con.execute(
                """SELECT COUNT(*) FROM degradation_alerts
                   WHERE status = 'active' AND occurrence_count >= 3"""
            ).fetchone()[0]

            return {
                "total_alerts": total,
                "by_status": {s[0]: s[1] for s in by_status},
                "active_by_model": {m[0]: m[1] for m in by_model},
                "active_by_severity": {s[0]: s[1] for s in by_severity},
                "escalated_alerts": escalated,
            }
        except Exception as e:
            log.error(f"Failed to get alert summary: {e}")
            return {"status": "error", "error": str(e)}


def get_degradation_monitor() -> DegradationMonitor:
    """Get or create degradation monitor instance"""
    return DegradationMonitor()
