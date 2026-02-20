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
from typing import Optional

from footy.db import connect
from footy.utils import score_prediction


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
            WHERE ps.model_version = ? AND m.utc_date >= ?
            ORDER BY m.utc_date
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
            s = score_prediction(probs, outcome)
            
            outcomes.append(outcome)
            probabilities.append((outcome, probs))
            
            if s["correct"]:
                n_correct += 1
            
            outcome_counts[outcome] += 1
            
            # Log Loss
            total_logloss += s["logloss"]
            
            # Brier Score
            total_brier += s["brier"]
            
            # Track by outcome for accuracy breakdown
            if s["correct"]:
                outcome_correct_counts[outcome] += 1
            
            # Track by competition
            if competition not in competitions:
                competitions[competition] = {"total": 0, "correct": 0}
            competitions[competition]["total"] += 1
            
            if predicted_outcome == outcome:
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
            SELECT DATE(m.utc_date) as match_date, COUNT(*) as n_pred, 
                   SUM(CASE WHEN ps.correct THEN 1 ELSE 0 END) as n_correct,
                   AVG(ps.logloss) as avg_logloss, AVG(ps.brier) as avg_brier
            FROM prediction_scores ps
            JOIN matches m ON ps.match_id = m.match_id
            WHERE ps.model_version = ? AND m.utc_date >= ?
            GROUP BY DATE(m.utc_date)
            ORDER BY match_date DESC
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
            JOIN matches m2 ON ps.match_id = m2.match_id
            WHERE m2.utc_date >= ?
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


# ===================================================================
# SELF-IMPROVEMENT: Error Analysis & Feedback Loop
# ===================================================================

def analyze_prediction_errors(model_version: str = "v10_council",
                               days: int = 180) -> dict:
    """
    Deep error analysis — identifies systematic weaknesses in predictions.

    Returns a dict with:
    - error_patterns: where the model consistently gets it wrong
    - calibration_analysis: reliability curve data
    - btts_accuracy / ou25_accuracy: market prediction stats
    - goal_mae: expected goal accuracy
    - recommendations: actionable retraining suggestions
    """
    con = connect()
    window = datetime.now(timezone.utc) - timedelta(days=days)

    # ---- 1. Outcome confusion matrix ----
    rows = con.execute("""
        SELECT ps.outcome, p.p_home, p.p_draw, p.p_away,
               m.competition, m.home_team, m.away_team,
               m.home_goals, m.away_goals,
               ps.goals_mae, ps.btts_correct, ps.ou25_correct,
               ps.score_correct, ps.p_btts, ps.p_o25,
               ps.eg_home, ps.eg_away
        FROM prediction_scores ps
        JOIN predictions p ON ps.match_id = p.match_id AND ps.model_version = p.model_version
        JOIN matches m ON ps.match_id = m.match_id
        WHERE ps.model_version = ? AND m.utc_date >= ?
        ORDER BY m.utc_date
    """, [model_version, window]).fetchall()

    if not rows:
        return {"status": "no_data", "n_scored": 0}

    # Confusion matrix
    confusion = [[0]*3 for _ in range(3)]  # confusion[predicted][actual]
    comp_errors = {}
    confidence_buckets = {i: {"n": 0, "correct": 0} for i in range(10)}  # 0-10%, 10-20%, etc.

    total_goals_mae = 0.0
    n_goals = 0
    n_btts = 0
    n_btts_correct = 0
    n_ou25 = 0
    n_ou25_correct = 0
    n_score = 0
    n_score_correct = 0
    overconfident_wrong = []  # Cases where model was very confident but wrong
    draw_bias_data = {"predicted_draws": 0, "actual_draws": 0, "draw_correct": 0}

    for (outcome, ph, pd_, pa, comp, ht, at, hg, ag,
         g_mae, btts_ok, ou_ok, sc_ok, p_btts, p_o25, eg_h, eg_a) in rows:
        probs = [float(ph), float(pd_), float(pa)]
        predicted = max(range(3), key=lambda i: probs[i])
        max_prob = probs[predicted]

        confusion[predicted][outcome] += 1

        # Calibration buckets
        bucket = min(9, int(max_prob * 10))
        confidence_buckets[bucket]["n"] += 1
        if predicted == outcome:
            confidence_buckets[bucket]["correct"] += 1

        # Competition-level errors
        if comp not in comp_errors:
            comp_errors[comp] = {"n": 0, "correct": 0, "draw_missed": 0, "home_bias": 0}
        comp_errors[comp]["n"] += 1
        if predicted == outcome:
            comp_errors[comp]["correct"] += 1
        if outcome == 1 and predicted != 1:
            comp_errors[comp]["draw_missed"] += 1
        if predicted == 0 and outcome != 0:
            comp_errors[comp]["home_bias"] += 1

        # Draw analysis
        if predicted == 1:
            draw_bias_data["predicted_draws"] += 1
        if outcome == 1:
            draw_bias_data["actual_draws"] += 1
            if predicted == 1:
                draw_bias_data["draw_correct"] += 1

        # Overconfident wrong
        if max_prob >= 0.65 and predicted != outcome:
            overconfident_wrong.append({
                "match": f"{ht} vs {at}", "comp": comp,
                "predicted": ["Home", "Draw", "Away"][predicted],
                "actual": ["Home", "Draw", "Away"][outcome],
                "confidence": max_prob,
            })

        # Goals MAE
        if g_mae is not None:
            total_goals_mae += g_mae
            n_goals += 1
        # BTTS
        if btts_ok is not None:
            n_btts += 1
            n_btts_correct += int(btts_ok)
        # O/U 2.5
        if ou_ok is not None:
            n_ou25 += 1
            n_ou25_correct += int(ou_ok)
        # Score
        if sc_ok is not None:
            n_score += 1
            n_score_correct += int(sc_ok)

    n = len(rows)
    # ---- 2. Calibration curve ----
    calibration = []
    for bucket_i in range(10):
        b = confidence_buckets[bucket_i]
        if b["n"] > 0:
            calibration.append({
                "bin_center": (bucket_i + 0.5) / 10,
                "predicted_prob": (bucket_i + 0.5) / 10,
                "actual_freq": b["correct"] / b["n"],
                "count": b["n"],
            })

    # ---- 3. Per-competition analysis ----
    comp_analysis = {}
    for comp, data in comp_errors.items():
        acc = data["correct"] / data["n"] if data["n"] > 0 else 0
        comp_analysis[comp] = {
            "accuracy": acc,
            "n": data["n"],
            "draw_miss_rate": data["draw_missed"] / data["n"] if data["n"] > 0 else 0,
            "home_bias_rate": data["home_bias"] / data["n"] if data["n"] > 0 else 0,
        }

    # ---- 4. Generate recommendations ----
    recommendations = []
    overall_accuracy = sum(confusion[i][i] for i in range(3)) / n if n > 0 else 0

    # Draw detection
    draw_rate = draw_bias_data["actual_draws"] / n if n > 0 else 0
    draw_pred_rate = draw_bias_data["predicted_draws"] / n if n > 0 else 0
    if draw_rate > 0.25 and draw_pred_rate < draw_rate * 0.5:
        recommendations.append({
            "type": "draw_underprediction",
            "severity": "high",
            "message": f"Model predicts draws {draw_pred_rate:.0%} of the time but actual draw rate is {draw_rate:.0%}. Consider adding draw-specialist features.",
        })

    # Home bias
    home_pred_rate = sum(confusion[0]) / n if n > 0 else 0
    actual_home_rate = (confusion[0][0] + confusion[1][0] + confusion[2][0]) / n if n > 0 else 0
    if home_pred_rate > actual_home_rate * 1.2:
        recommendations.append({
            "type": "home_bias",
            "severity": "medium",
            "message": f"Model over-predicts home wins ({home_pred_rate:.0%} predicted vs {actual_home_rate:.0%} actual).",
        })

    # Overconfidence
    if len(overconfident_wrong) > n * 0.1:
        recommendations.append({
            "type": "overconfidence",
            "severity": "high",
            "message": f"{len(overconfident_wrong)} predictions had >=65% confidence but were wrong ({len(overconfident_wrong)/n:.0%} of total).",
        })

    # Per-competition weakness
    for comp, data in comp_analysis.items():
        if data["n"] >= 10 and data["accuracy"] < overall_accuracy - 0.1:
            recommendations.append({
                "type": "weak_competition",
                "severity": "medium",
                "message": f"Accuracy in {comp} is {data['accuracy']:.0%} vs {overall_accuracy:.0%} overall. Consider competition-specific features.",
                "competition": comp,
            })

    # Goals accuracy
    if n_goals > 0:
        avg_mae = total_goals_mae / n_goals
        if avg_mae > 1.0:
            recommendations.append({
                "type": "goal_accuracy",
                "severity": "medium",
                "message": f"Expected goals MAE is {avg_mae:.2f} — consider improving Poisson/DC models.",
            })

    return {
        "status": "ok",
        "model_version": model_version,
        "n_scored": n,
        "window_days": days,
        "overall_accuracy": overall_accuracy,
        "confusion_matrix": {
            "labels": ["Home", "Draw", "Away"],
            "matrix": confusion,
        },
        "calibration": calibration,
        "by_competition": comp_analysis,
        "draw_analysis": draw_bias_data,
        "goals_mae": total_goals_mae / n_goals if n_goals else None,
        "btts_accuracy": n_btts_correct / n_btts if n_btts else None,
        "ou25_accuracy": n_ou25_correct / n_ou25 if n_ou25 else None,
        "score_accuracy": n_score_correct / n_score if n_score else None,
        "n_btts": n_btts,
        "n_ou25": n_ou25,
        "n_score": n_score,
        "overconfident_wrong": overconfident_wrong[:10],  # top 10 worst
        "recommendations": recommendations,
    }


def generate_improvement_report(model_version: str = "v10_council",
                                 days: int = 180) -> str:
    """
    Generate a human-readable self-improvement report.

    Used by CLI `footy improvement-report` and by the auto-retrain system
    to decide whether retraining should focus on specific weaknesses.
    """
    analysis = analyze_prediction_errors(model_version, days)

    if analysis["status"] == "no_data":
        return "No scored predictions to analyze."

    lines = [
        f"=== Self-Improvement Report: {model_version} ===",
        f"Window: last {days} days | Scored: {analysis['n_scored']}",
        f"Overall Accuracy: {analysis['overall_accuracy']:.1%}",
        "",
        "--- Confusion Matrix ---",
        "            Actual: Home   Draw   Away",
    ]

    labels = ["Home", "Draw", "Away"]
    for i, label in enumerate(labels):
        row = analysis["confusion_matrix"]["matrix"][i]
        lines.append(f"  Predicted {label:5s}: {row[0]:5d}  {row[1]:5d}  {row[2]:5d}")

    # Market predictions
    lines.append("")
    lines.append("--- Market Predictions ---")
    if analysis["btts_accuracy"] is not None:
        lines.append(f"  BTTS:       {analysis['btts_accuracy']:.1%} ({analysis['n_btts']} scored)")
    if analysis["ou25_accuracy"] is not None:
        lines.append(f"  Over 2.5:   {analysis['ou25_accuracy']:.1%} ({analysis['n_ou25']} scored)")
    if analysis["score_accuracy"] is not None:
        lines.append(f"  Exact Score: {analysis['score_accuracy']:.1%} ({analysis['n_score']} scored)")
    if analysis["goals_mae"] is not None:
        lines.append(f"  Goals MAE:  {analysis['goals_mae']:.3f}")

    # Calibration
    lines.append("")
    lines.append("--- Calibration ---")
    for cal in analysis["calibration"]:
        bar_len = int(cal["actual_freq"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"  {cal['predicted_prob']:.0%}: {bar} {cal['actual_freq']:.0%} (n={cal['count']})")

    # Per-competition
    lines.append("")
    lines.append("--- By Competition ---")
    for comp, data in sorted(analysis["by_competition"].items(), key=lambda x: x[1]["accuracy"]):
        lines.append(f"  {comp:5s}: {data['accuracy']:.1%} (n={data['n']}, draw_miss={data['draw_miss_rate']:.0%})")

    # Recommendations
    if analysis["recommendations"]:
        lines.append("")
        lines.append("--- Recommendations ---")
        for rec in analysis["recommendations"]:
            icon = "🔴" if rec["severity"] == "high" else "🟡"
            lines.append(f"  {icon} [{rec['type']}] {rec['message']}")

    return "\n".join(lines)
