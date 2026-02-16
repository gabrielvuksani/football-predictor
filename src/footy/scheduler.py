"""
Phase 4: Training Scheduler

Manages automated scheduling of data ingestion, model training, and predictions.
Uses APScheduler for job scheduling with persistent storage in DuckDB.

Scheduled Jobs:
- ingest: Fetch upcoming fixtures and finished match results
- train_base: Train Elo + Poisson models
- train_council: Train council model (v8) with 6 experts + meta-learner
- predict: Generate predictions for upcoming matches
- score: Score finished predictions and update metrics

Legacy job types (still supported):
- train_meta: Train meta-stacker (v2)
- train_v3: Train GBDT form model
- train_v4: Alias for train_council
- train_v5: Train ultimate ensemble
"""
from __future__ import annotations
import json
import logging
import time
from datetime import datetime
from typing import Optional
from enum import Enum
import math

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.job import Job

from footy.db import connect
from footy import pipeline
from footy.models.council import train_and_save as council_train_and_save, predict_upcoming as council_predict_upcoming

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a scheduled job"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DISABLED = "disabled"


class ScheduledJob:
    """Represents a scheduled job with metadata and execution tracking"""
    
    def __init__(
        self,
        job_id: str,
        job_type: str,
        cron_schedule: str,
        enabled: bool = True,
        params: Optional[dict] = None,
    ):
        self.job_id = job_id
        self.job_type = job_type
        self.cron_schedule = cron_schedule
        self.enabled = enabled
        self.params = params or {}
        self.created_at = datetime.utcnow()
        self.last_run_at = None
        self.next_run_at = None
        self.last_status = JobStatus.PENDING
        self.last_error = None
        self.total_runs = 0
        self.failed_runs = 0


class TrainingScheduler:
    """
    Manages automated training pipeline using APScheduler.
    
    Example usage:
        scheduler = TrainingScheduler()
        scheduler.add_job(
            job_id="daily_ingest",
            job_type="ingest",
            cron_schedule="0 2 * * *",  # 2 AM daily
            params={"days_back": 30, "days_forward": 7}
        )
        scheduler.start()
        scheduler.list_jobs()
    """
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.con = connect()
        self._ensure_schema()
        self._load_jobs_from_db()
    
    def _ensure_schema(self):
        """Create scheduled_jobs and job_runs tables if they don't exist"""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS scheduled_jobs (
                job_id VARCHAR PRIMARY KEY,
                job_type VARCHAR NOT NULL,
                cron_schedule VARCHAR NOT NULL,
                enabled BOOLEAN DEFAULT TRUE,
                params VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS job_runs (
                run_id INTEGER PRIMARY KEY,
                job_id VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                duration_seconds DOUBLE,
                error_message VARCHAR,
                result_json VARCHAR,
                FOREIGN KEY (job_id) REFERENCES scheduled_jobs(job_id)
            )
        """)
        
        self.con.execute("""
            CREATE SEQUENCE IF NOT EXISTS job_runs_seq START 1
        """)
        
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS job_runs_job_id_idx ON job_runs(job_id, started_at DESC)
        """)
    
    def _load_jobs_from_db(self):
        """Load scheduled jobs from database and add to scheduler"""
        rows = self.con.execute("""
            SELECT job_id, job_type, cron_schedule, enabled, params
            FROM scheduled_jobs
            WHERE enabled = TRUE
            ORDER BY created_at
        """).fetchall()
        
        for job_id, job_type, cron_schedule, enabled, params_json in rows:
            params = json.loads(params_json) if params_json else {}
            self._schedule_job_internal(job_id, job_type, cron_schedule, params)
    
    def add_job(
        self,
        job_id: str,
        job_type: str,
        cron_schedule: str,
        params: Optional[dict] = None,
    ) -> dict:
        """
        Add a new scheduled job.
        
        Args:
            job_id: Unique identifier (e.g., "daily_ingest")
            job_type: Job type (ingest, train_base, train_council, predict, score)
            cron_schedule: Cron expression (e.g., "0 2 * * *" for 2 AM daily)
            params: Optional parameters for the job
        
        Returns:
            Job configuration dict
        """
        valid_types = ["ingest", "train_base", "train_council", "predict", "score"]
        if job_type not in valid_types:
            raise ValueError(f"Invalid job_type: {job_type}")
        
        params = params or {}
        params_json = json.dumps(params)
        
        # Store in database
        self.con.execute("""
            INSERT OR REPLACE INTO scheduled_jobs
            (job_id, job_type, cron_schedule, enabled, params, updated_at)
            VALUES (?, ?, ?, TRUE, ?, CURRENT_TIMESTAMP)
        """, [job_id, job_type, cron_schedule, params_json])
        
        # Schedule in APScheduler
        self._schedule_job_internal(job_id, job_type, cron_schedule, params)
        
        return {
            "job_id": job_id,
            "job_type": job_type,
            "cron_schedule": cron_schedule,
            "enabled": True,
            "params": params,
        }
    
    def _schedule_job_internal(self, job_id: str, job_type: str, cron_schedule: str, params: dict):
        """Internal method to schedule a job in APScheduler"""
        # Remove existing job if present
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
        
        # Get the job handler function
        job_func = self._get_job_func(job_type)
        
        # Add job to scheduler
        self.scheduler.add_job(
            self._run_job_wrapper,
            trigger=CronTrigger.from_crontab(cron_schedule),
            id=job_id,
            args=[job_id, job_func, params],
            replace_existing=True,
            misfire_grace_time=600,  # Allow 10 mins grace before missing a fire
        )
    
    def _get_job_func(self, job_type: str):
        """Get the function to run for a job type"""
        job_funcs = {
            "ingest": self._job_ingest,
            "train_base": self._job_train_base,
            "train_council": self._job_train_council,
            "predict": self._job_predict,
            "score": self._job_score,
        }
        return job_funcs.get(job_type, self._job_ingest)
    
    def _run_job_wrapper(self, job_id: str, job_func, params: dict):
        """Wrapper that executes a job and logs the result"""
        started_at = datetime.utcnow()
        status = JobStatus.RUNNING
        error_message = None
        result = None
        
        try:
            result = job_func(**params)
            status = JobStatus.SUCCESS
        except Exception as e:
            status = JobStatus.FAILED
            error_message = str(e)
            log.error("Job %s failed: %s", job_id, error_message)
        finally:
            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()
            
            # Log to database
            result_json = json.dumps(result) if result else None
            run_id = self.con.execute("SELECT nextval('job_runs_seq')").fetchone()[0]
            self.con.execute("""
                INSERT INTO job_runs
                (run_id, job_id, status, started_at, completed_at, duration_seconds, error_message, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [run_id, job_id, status.value, started_at, completed_at, duration, error_message, result_json])
    
    # Job handler methods
    def _job_ingest(self, days_back: int = 30, days_forward: int = 7) -> dict:
        """Ingest fixture data"""
        con = connect()
        n = pipeline.ingest(days_back=days_back, days_forward=days_forward, verbose=False)
        return {"rows_ingested": n}
    
    def _job_train_base(self) -> dict:
        """Train base Elo + Poisson models"""
        con = connect()
        n_elo = pipeline.update_elo_from_finished(verbose=False)
        state = pipeline.refit_poisson(verbose=False)
        return {"elo_updates": n_elo, "poisson_teams": len(state.get("teams", []))}
    
    def _job_train_council(self, eval_days: int = 365) -> dict:
        """Train council model (v8)."""
        con = connect()
        result = council_train_and_save(con, eval_days=eval_days, verbose=False)
        return result

    def _job_predict(self, lookahead_days: int = 7) -> dict:
        """Generate predictions for upcoming matches using the council model."""
        con = connect()
        n = council_predict_upcoming(con, lookahead_days=lookahead_days, verbose=False)
        return {"council_predictions": n}
    
    def _job_score(self) -> dict:
        """Score finished predictions"""
        con = connect()
        
        # Find finished matches with predictions
        rows = con.execute("""
            SELECT m.match_id, m.home_goals, m.away_goals, p.model_version, p.p_home, p.p_draw, p.p_away
            FROM matches m
            JOIN predictions p ON m.match_id = p.match_id
            WHERE m.status = 'FINISHED'
            AND m.home_goals IS NOT NULL
            AND m.away_goals IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM prediction_scores ps
                WHERE ps.match_id = m.match_id AND ps.model_version = p.model_version
            )
            ORDER BY m.utc_date DESC
            LIMIT 100
        """).fetchall()
        
        scored_count = 0
        for match_id, home_goals, away_goals, model_version, p_home, p_draw, p_away in rows:
            # Determine outcome
            if home_goals > away_goals:
                outcome = 0  # Home win
            elif home_goals < away_goals:
                outcome = 2  # Away win
            else:
                outcome = 1  # Draw
            
            # Calculate metrics
            probs = [p_home, p_draw, p_away]
            outcome_prob = probs[outcome]
            
            # Brier score (mean squared error)
            brier = sum((p - (1.0 if i == outcome else 0.0)) ** 2 for i, p in enumerate(probs)) / 3
            
            # Log loss (negative log likelihood)
            logloss = -math.log(max(outcome_prob, 1e-15))
            
            # Store score
            predicted_outcome = max(range(3), key=lambda i: probs[i])
            con.execute("""
                INSERT INTO prediction_scores
                (match_id, model_version, outcome, logloss, brier, correct)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [match_id, model_version, outcome, logloss, brier, predicted_outcome == outcome])
            
            scored_count += 1
        
        return {"scored_predictions": scored_count}
    
    def remove_job(self, job_id: str) -> dict:
        """Remove a scheduled job"""
        # Remove from APScheduler
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
        
        # Remove from database
        self.con.execute("DELETE FROM scheduled_jobs WHERE job_id = ?", [job_id])
        
        return {"removed": job_id}
    
    def disable_job(self, job_id: str) -> dict:
        """Disable a scheduled job (keeps config, stops execution)"""
        self.con.execute(
            "UPDATE scheduled_jobs SET enabled = FALSE, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?",
            [job_id]
        )
        
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
        
        return {"disabled": job_id}
    
    def enable_job(self, job_id: str) -> dict:
        """Re-enable a disabled scheduled job"""
        row = self.con.execute(
            "SELECT job_type, cron_schedule, params FROM scheduled_jobs WHERE job_id = ?",
            [job_id]
        ).fetchone()
        
        if not row:
            raise ValueError(f"Job {job_id} not found")
        
        job_type, cron_schedule, params_json = row
        params = json.loads(params_json) if params_json else {}
        
        self.con.execute(
            "UPDATE scheduled_jobs SET enabled = TRUE, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?",
            [job_id]
        )
        
        self._schedule_job_internal(job_id, job_type, cron_schedule, params)
        
        return {"enabled": job_id}
    
    def list_jobs(self) -> list:
        """List all scheduled jobs with their status"""
        rows = self.con.execute("""
            SELECT job_id, job_type, cron_schedule, enabled, created_at
            FROM scheduled_jobs
            ORDER BY created_at DESC
        """).fetchall()
        
        jobs = []
        for job_id, job_type, cron_schedule, enabled, created_at in rows:
            # Get last run
            last_run = self.con.execute("""
                SELECT status, started_at, duration_seconds, error_message
                FROM job_runs
                WHERE job_id = ?
                ORDER BY started_at DESC
                LIMIT 1
            """, [job_id]).fetchone()
            
            last_status = last_run[0] if last_run else None
            last_run_at = last_run[1] if last_run else None
            last_duration = last_run[2] if last_run else None
            last_error = last_run[3] if last_run else None
            
            # Get APScheduler next run time
            ap_job = self.scheduler.get_job(job_id)
            next_run_at = ap_job.next_run_time if ap_job and hasattr(ap_job, 'next_run_time') else None
            
            jobs.append({
                "job_id": job_id,
                "job_type": job_type,
                "cron_schedule": cron_schedule,
                "enabled": enabled,
                "created_at": created_at,
                "last_status": last_status,
                "last_run_at": last_run_at,
                "last_duration_seconds": last_duration,
                "last_error": last_error,
                "next_run_at": next_run_at,
            })
        
        return jobs
    
    def get_job_history(self, job_id: str, limit: int = 10) -> list:
        """Get execution history for a job"""
        rows = self.con.execute("""
            SELECT started_at, status, duration_seconds, error_message, result_json
            FROM job_runs
            WHERE job_id = ?
            ORDER BY started_at DESC
            LIMIT ?
        """, [job_id, limit]).fetchall()
        
        history = []
        for started_at, status, duration, error, result_json in rows:
            result = json.loads(result_json) if result_json else None
            history.append({
                "started_at": started_at,
                "status": status,
                "duration_seconds": duration,
                "error": error,
                "result": result,
            })
        
        return history
    
    def start(self):
        """Start the scheduler in background"""
        if not self.scheduler.running:
            self.scheduler.start()
    
    def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
    
    def get_stats(self) -> dict:
        """Get scheduler statistics"""
        rows = self.con.execute("""
            SELECT sj.job_type, COUNT(*) as total_runs, 
                   SUM(CASE WHEN jr.status = 'SUCCESS' THEN 1 ELSE 0 END) as successful,
                   SUM(CASE WHEN jr.status = 'FAILED' THEN 1 ELSE 0 END) as failed,
                   AVG(CASE WHEN jr.duration_seconds IS NOT NULL THEN jr.duration_seconds ELSE 0 END) as avg_duration_seconds
            FROM job_runs jr
            JOIN scheduled_jobs sj ON jr.job_id = sj.job_id
            GROUP BY sj.job_type
            ORDER BY total_runs DESC
        """).fetchall()
        
        stats = {}
        for job_type, total, successful, failed, avg_duration in rows:
            stats[job_type] = {
                "total_runs": total,
                "successful": successful or 0,
                "failed": failed or 0,
                "avg_duration_seconds": avg_duration or 0,
            }
        
        return {
            "total_jobs": len(self.list_jobs()),
            "active_jobs": len(self.scheduler.get_jobs()) if self.scheduler.running else 0,
            "scheduler_running": self.scheduler.running,
            "stats_by_type": stats,
        }


# Global scheduler instance
_scheduler_instance = None


def get_scheduler() -> TrainingScheduler:
    """Get or create the global scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = TrainingScheduler()
    return _scheduler_instance
