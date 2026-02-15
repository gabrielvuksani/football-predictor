"""
Phase 4.3: Performance Tracking System

Tracks model performance metrics over time, provides aggregated statistics,
and enables comparison across models, time periods, and competitions.

Features:
- Per-model aggregated metrics (accuracy, logloss, brier)
- Rolling and cumulative performance tracking
- Competition-specific and league-specific metrics
- Time series tracking for degradation detection
- Comparative analysis across models
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import math

from footy.db import connect


class PerformanceTracker:
    """
    Tracks and aggregates model prediction performance metrics.
    
    Metrics computed:
    - Accuracy: % of correct predictions
    - Log Loss: measures probability calibration
    - Brier Score: mean squared error of probabilities
    - Calibration Slope: how well probabilities match outcomes
    - AUC: discriminative ability for 2-class problems
    
    Example usage:
        tracker = PerformanceTracker()
        
        # Get daily performance
        daily = tracker.get_daily_performance(model_version="v5_ultimate", days=30)
        
        # Compare models
        comparison = tracker.compare_models(["v1", "v4", "v5"], days=365)
        
        # Track degradation
        trend = tracker.get_performance_trend("v5_ultimate", window_days=30)
        degradation = trend["trend_slope"] if trend else 0
    """
    
    def __init__(self):
        self.con = connect()
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create performance tracking tables"""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY,
                model_version VARCHAR NOT NULL,
                metric_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                window_days INT DEFAULT 1,
                n_predictions INT DEFAULT 0,
                n_correct INT DEFAULT 0,
                accuracy DOUBLE DEFAULT 0.0,
                logloss DOUBLE DEFAULT 0.0,
                brier DOUBLE DEFAULT 0.0,
                calibration_slope DOUBLE DEFAULT 1.0,
                competition VARCHAR,
                home_accuracy DOUBLE,
                draw_accuracy DOUBLE,
                away_accuracy DOUBLE
            )
        """)
        
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS performance_thresholds (
                model_version VARCHAR PRIMARY KEY,
                min_accuracy DOUBLE DEFAULT 0.45,
                max_logloss DOUBLE DEFAULT 0.65,
                max_brier DOUBLE DEFAULT 0.25,
                alert_threshold DOUBLE DEFAULT -0.05,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS performance_metrics_idx 
            ON performance_metrics(model_version, metric_date DESC)
        """)
    
    def compute_aggregated_metrics(
        self,
        model_version: str,
        days: int = 365,
        window_start: Optional[datetime] = None,
    ) -> dict:
        """
        Compute aggregated metrics for a model over a time window.
        
        Args:
            model_version: Model version to analyze
            days: Time window in days (default: 365)
            window_start: Start date (default: today - days)
        
        Returns:
            Aggregated metrics dict
        """
        if window_start is None:
            window_start = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get all scored predictions for this model in the window
        rows = self.con.execute("""
            SELECT ps.outcome, p.p_home, p.p_draw, p.p_away, m.competition
            FROM prediction_scores ps
            JOIN predictions p ON ps.match_id = p.match_id AND ps.model_version = p.model_version
            JOIN matches m ON ps.match_id = m.match_id
            WHERE ps.model_version = ? AND ps.scored_at >= ?
            ORDER BY ps.scored_at
        """, [model_version, window_start]).fetchall()
        
        if not rows:
            return {
                "model_version": model_version,
                "n_predictions": 0,
                "window_days": days,
                "status": "no_data",
            }
        
        outcomes = []
        probabilities = []
        competitions = {}
        
        n_correct = 0
        total_logloss = 0.0
        total_brier = 0.0
        
        # Outcome counts
        outcome_counts = {0: 0, 1: 0, 2: 0}  # Home, Draw, Away
        outcome_correct_counts = {0: 0, 1: 0, 2: 0}
        
        for outcome, p_home, p_draw, p_away, competition in rows:
            probs = [p_home, p_draw, p_away]
            outcome_prob = probs[outcome]
            
            outcomes.append(outcome)
            probabilities.append((outcome, probs))
            
            if outcome_prob > 0.5:
                n_correct += 1
            
            outcome_counts[outcome] += 1
            
            # Log Loss
            logloss = -math.log(max(outcome_prob, 1e-15))
            total_logloss += logloss
            
            # Brier Score
            brier = sum((p - (1.0 if i == outcome else 0.0)) ** 2 for i, p in enumerate(probs)) / 3
            total_brier += brier
            
            # Track by outcome for accuracy breakdown
            predicted_outcome = max(range(3), key=lambda i: probs[i])
            if predicted_outcome == outcome:
                outcome_correct_counts[outcome] += 1
            
            # Track by competition
            if competition not in competitions:
                competitions[competition] = {"total": 0, "correct": 0}
            competitions[competition]["total"] += 1
            
            if outcome_prob > 0.5:
                competitions[competition]["correct"] += 1
        
        n = len(rows)
        accuracy = n_correct / n if n > 0 else 0.0
        avg_logloss = total_logloss / n if n > 0 else 0.0
        avg_brier = total_brier / n if n > 0 else 0.0
        
        # Outcome-specific accuracy
        home_accuracy = outcome_correct_counts[0] / (outcome_counts[0] + 1e-10)
        draw_accuracy = outcome_correct_counts[1] / (outcome_counts[1] + 1e-10)
        away_accuracy = outcome_correct_counts[2] / (outcome_counts[2] + 1e-10)
        
        # Competition breakdown
        comp_metrics = {}
        for comp, stats in competitions.items():
            comp_metrics[comp] = {
                "accuracy": stats["correct"] / stats["total"],
                "n_predictions": stats["total"],
            }
        
        return {
            "model_version": model_version,
            "n_predictions": n,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "logloss": avg_logloss,
            "brier": avg_brier,
            "window_days": days,
            "window_start": window_start,
            "home_accuracy": home_accuracy,
            "draw_accuracy": draw_accuracy,
            "away_accuracy": away_accuracy,
            "by_competition": comp_metrics,
        }
    
    def get_daily_performance(self, model_version: str, days: int = 30) -> list:
        """Get daily performance breakdown"""
        window_start = datetime.now(timezone.utc) - timedelta(days=days)
        
        rows = self.con.execute("""
            SELECT DATE(ps.scored_at) as score_date, COUNT(*) as n_pred, 
                   SUM(CASE WHEN ps.correct THEN 1 ELSE 0 END) as n_correct,
                   AVG(ps.logloss) as avg_logloss, AVG(ps.brier) as avg_brier
            FROM prediction_scores ps
            WHERE ps.model_version = ? AND ps.scored_at >= ?
            GROUP BY DATE(ps.scored_at)
            ORDER BY score_date DESC
        """, [model_version, window_start]).fetchall()
        
        daily = []
        for score_date, n_pred, n_correct, logloss, brier in rows:
            accuracy = (n_correct or 0) / n_pred if n_pred else 0.0
            daily.append({
                "date": score_date,
                "n_predictions": n_pred,
                "accuracy": accuracy,
                "logloss": logloss or 0.0,
                "brier": brier or 0.0,
            })
        
        return daily
    
    def get_performance_trend(
        self,
        model_version: str,
        window_days: int = 30,
        lookback_days: int = 180,
    ) -> Optional[dict]:
        """
        Calculate performance trend to detect degradation.
        
        Returns:
            Trend analysis with slope (negative = degrading)
        """
        daily = self.get_daily_performance(model_version, days=lookback_days)
        
        if len(daily) < 2:
            return None
        
        # Use recent window for trend slope
        recent = [d for d in daily if
                  (datetime.now(timezone.utc) - timedelta(days=window_days)) <= d['date']]
        
        if len(recent) < 2:
            recent = daily[:window_days]
        
        if len(recent) < 2:
            return None
        
        # Calculate trend using linear regression
        accuracies = [d["accuracy"] for d in recent]
        n = len(accuracies)
        
        # Simple slope calculation
        x_sum = sum(range(n))
        y_sum = sum(accuracies)
        xy_sum = sum(i * accuracies[i] for i in range(n))
        x2_sum = sum(i**2 for i in range(n))
        
        numerator = n * xy_sum - x_sum * y_sum
        denominator = n * x2_sum - x_sum ** 2
        
        slope = numerator / denominator if denominator != 0 else 0.0
        intercept = (y_sum - slope * x_sum) / n
        
        # R-squared
        y_mean = y_sum / n
        ss_tot = sum((y - y_mean) ** 2 for y in accuracies)
        ss_res = sum((y - (intercept + slope * i)) ** 2 for i, y in enumerate(accuracies))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            "model_version": model_version,
            "window_days": window_days,
            "n_data_points": n,
            "current_accuracy": accuracies[-1] if accuracies else 0.0,
            "avg_accuracy": sum(accuracies) / len(accuracies),
            "trend_slope": slope,
            "degrading": slope < -0.001,  # Threshold for degradation
            "r_squared": r_squared,
        }
    
    def compare_models(
        self,
        model_versions: list,
        days: int = 365,
    ) -> dict:
        """
        Compare performance across multiple models.
        
        Args:
            model_versions: List of model versions to compare
            days: Time window for comparison
        
        Returns:
            Comparative metrics dict
        """
        comparison = []
        
        for model_version in model_versions:
            metrics = self.compute_aggregated_metrics(model_version, days=days)
            comparison.append(metrics)
        
        # Sort by accuracy
        comparison.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
        
        return {
            "models": comparison,
            "window_days": days,
            "best_model": comparison[0]["model_version"] if comparison else None,
            "best_accuracy": comparison[0].get("accuracy", 0) if comparison else 0,
        }
    
    def set_performance_thresholds(
        self,
        model_version: str,
        min_accuracy: float = 0.45,
        max_logloss: float = 0.65,
        max_brier: float = 0.25,
        alert_threshold: float = -0.05,
    ) -> dict:
        """
        Set performance thresholds for monitoring and alerting.
        
        Args:
            model_version: Model to configure
            min_accuracy: Minimum acceptable accuracy
            max_logloss: Maximum acceptable logloss
            max_brier: Maximum acceptable brier score
            alert_threshold: Performance drop to trigger alert (e.g., -0.05 = 5% drop)
        """
        self.con.execute("""
            INSERT OR REPLACE INTO performance_thresholds
            (model_version, min_accuracy, max_logloss, max_brier, alert_threshold)
            VALUES (?, ?, ?, ?, ?)
        """, [model_version, min_accuracy, max_logloss, max_brier, alert_threshold])
        
        return {
            "model_version": model_version,
            "min_accuracy": min_accuracy,
            "max_logloss": max_logloss,
            "max_brier": max_brier,
            "alert_threshold": alert_threshold,
        }
    
    def check_performance_health(self, model_version: str, days: int = 30) -> dict:
        """
        Check if model performance is within acceptable thresholds.
        
        Returns:
            Health status with alerts if thresholds violated
        """
        # Get thresholds
        threshold_row = self.con.execute("""
            SELECT min_accuracy, max_logloss, max_brier, alert_threshold
            FROM performance_thresholds WHERE model_version = ?
        """, [model_version]).fetchone()
        
        if not threshold_row:
            return {"status": "not_configured"}
        
        min_acc, max_loss, max_brier, alert_thresh = threshold_row
        
        # Get current metrics
        metrics = self.compute_aggregated_metrics(model_version, days=days)
        
        if metrics.get("n_predictions", 0) == 0:
            return {"status": "no_data"}
        
        # Check thresholds
        alerts = []
        
        if metrics["accuracy"] < min_acc:
            alerts.append(f"Accuracy below threshold ({metrics['accuracy']:.3f} < {min_acc:.3f})")
        
        if metrics["logloss"] > max_loss:
            alerts.append(f"Logloss above threshold ({metrics['logloss']:.3f} > {max_loss:.3f})")
        
        if metrics["brier"] > max_brier:
            alerts.append(f"Brier above threshold ({metrics['brier']:.3f} > {max_brier:.3f})")
        
        # Check trend
        trend = self.get_performance_trend(model_version, window_days=days)
        if trend and trend["degrading"] and trend["trend_slope"] < alert_thresh:
            alerts.append(f"Performance degrading (slope: {trend['trend_slope']:.4f})")
        
        return {
            "status": "healthy" if not alerts else "degraded",
            "model_version": model_version,
            "metrics": metrics,
            "alerts": alerts,
            "window_days": days,
        }
    
    def get_model_rankings(self, days: int = 365) -> list:
        """Get all models ranked by accuracy"""
        rows = self.con.execute("""
            SELECT DISTINCT ps.model_version
            FROM prediction_scores ps
            WHERE ps.scored_at >= ?
        """, [datetime.now(timezone.utc) - timedelta(days=days)]).fetchall()
        
        models = [r[0] for r in rows]
        comparison = self.compare_models(models, days=days)
        
        return comparison["models"]
    
    def get_summary(self) -> dict:
        """Get overall performance summary for all models"""
        rows = self.con.execute("""
            SELECT DISTINCT model_version FROM predictions
        """).fetchall()
        
        models = [r[0] for r in rows]
        
        summary = {}
        for model_version in models:
            metrics = self.compute_aggregated_metrics(model_version, days=180)
            if metrics.get("n_predictions", 0) > 0:
                summary[model_version] = {
                    "accuracy": metrics.get("accuracy", 0),
                    "logloss": metrics.get("logloss", 0),
                    "brier": metrics.get("brier", 0),
                    "n_predictions": metrics.get("n_predictions", 0),
                }
        
        return summary


def get_performance_tracker() -> PerformanceTracker:
    """Get or create performance tracker instance"""
    return PerformanceTracker()
