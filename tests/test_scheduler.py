"""Tests for the scheduler module.

Tests job management, execution logging, and lifecycle operations
using an in-memory DuckDB database.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from footy.scheduler import TrainingScheduler, JobStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    """Create a scheduler backed by a temp DuckDB file so schema is fresh."""
    import duckdb

    db_path = str(tmp_path / "test_sched.duckdb")
    con = duckdb.connect(db_path)

    # Patch connect() so the scheduler uses our temp DB
    monkeypatch.setattr("footy.scheduler.connect", lambda: con)
    s = TrainingScheduler()
    yield s
    try:
        s.stop()
    except Exception:
        pass
    con.close()


# ---------------------------------------------------------------------------
# Schema & Init
# ---------------------------------------------------------------------------

class TestSchedulerInit:
    def test_schema_created(self, scheduler):
        tables = [r[0] for r in scheduler.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]
        assert "scheduled_jobs" in tables
        assert "job_runs" in tables

    def test_no_jobs_initially(self, scheduler):
        assert scheduler.list_jobs() == []

    def test_stats_empty(self, scheduler):
        stats = scheduler.get_stats()
        assert stats["total_jobs"] == 0
        assert stats["stats_by_type"] == {}


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------

class TestJobCRUD:
    def test_add_job(self, scheduler):
        result = scheduler.add_job("test_ingest", "ingest", "0 2 * * *")
        assert result["job_id"] == "test_ingest"
        assert result["job_type"] == "ingest"
        assert result["enabled"] is True

    def test_add_job_with_params(self, scheduler):
        result = scheduler.add_job(
            "custom_ingest", "ingest", "0 3 * * *",
            params={"days_back": 60}
        )
        assert result["params"]["days_back"] == 60

    def test_add_invalid_type_raises(self, scheduler):
        with pytest.raises(ValueError, match="Invalid job_type"):
            scheduler.add_job("bad", "nonexistent_type", "0 0 * * *")

    def test_list_after_add(self, scheduler):
        scheduler.add_job("j1", "ingest", "0 1 * * *")
        scheduler.add_job("j2", "predict", "0 2 * * *")
        jobs = scheduler.list_jobs()
        assert len(jobs) == 2
        ids = {j["job_id"] for j in jobs}
        assert ids == {"j1", "j2"}

    def test_remove_job(self, scheduler):
        scheduler.add_job("rm_me", "score", "0 5 * * *")
        assert len(scheduler.list_jobs()) == 1
        scheduler.remove_job("rm_me")
        assert len(scheduler.list_jobs()) == 0

    def test_disable_job(self, scheduler):
        scheduler.add_job("dis", "ingest", "0 0 * * *")
        scheduler.disable_job("dis")
        jobs = scheduler.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["enabled"] is False

    def test_enable_job(self, scheduler):
        scheduler.add_job("enab", "ingest", "0 0 * * *")
        scheduler.disable_job("enab")
        scheduler.enable_job("enab")
        jobs = scheduler.list_jobs()
        assert jobs[0]["enabled"] is True

    def test_enable_nonexistent_raises(self, scheduler):
        with pytest.raises(ValueError, match="not found"):
            scheduler.enable_job("ghost")


# ---------------------------------------------------------------------------
# Job Types
# ---------------------------------------------------------------------------

class TestJobTypes:
    """Verify all 7 job type handlers resolve without import errors."""

    @pytest.mark.parametrize("jt", [
        "ingest", "train_base", "train_council",
        "predict", "score", "retrain", "full_refresh",
    ])
    def test_job_func_resolves(self, scheduler, jt):
        func = scheduler._get_job_func(jt)
        assert callable(func)


# ---------------------------------------------------------------------------
# Job Execution Wrapper
# ---------------------------------------------------------------------------

class TestJobExecution:
    def test_successful_run_logged(self, scheduler):
        scheduler.add_job("ok_job", "ingest", "0 0 * * *")

        # Mock the actual ingest so it returns without doing real I/O
        with patch.object(scheduler, "_job_ingest", return_value={"rows_ingested": 42}):
            scheduler._run_job_wrapper("ok_job", scheduler._job_ingest, {})

        history = scheduler.get_job_history("ok_job")
        assert len(history) == 1
        assert history[0]["status"] == "success"
        assert history[0]["result"]["rows_ingested"] == 42
        assert history[0]["duration_seconds"] >= 0

    def test_failed_run_logged(self, scheduler):
        scheduler.add_job("fail_job", "ingest", "0 0 * * *")

        def _boom(**kw):
            raise RuntimeError("boom")

        scheduler._run_job_wrapper("fail_job", _boom, {})

        history = scheduler.get_job_history("fail_job")
        assert len(history) == 1
        assert history[0]["status"] == "failed"
        assert "boom" in history[0]["error"]

    def test_multiple_runs_history(self, scheduler):
        scheduler.add_job("multi", "score", "0 0 * * *")

        with patch.object(scheduler, "_job_score", return_value={"scored_predictions": 5}):
            for _ in range(3):
                scheduler._run_job_wrapper("multi", scheduler._job_score, {})

        history = scheduler.get_job_history("multi", limit=10)
        assert len(history) == 3


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestSchedulerStats:
    def test_stats_after_runs(self, scheduler):
        scheduler.add_job("st_job", "ingest", "0 0 * * *")

        with patch.object(scheduler, "_job_ingest", return_value={"rows_ingested": 1}):
            scheduler._run_job_wrapper("st_job", scheduler._job_ingest, {})

        stats = scheduler.get_stats()
        assert stats["total_jobs"] == 1
        assert "ingest" in stats["stats_by_type"]
        assert stats["stats_by_type"]["ingest"]["total_runs"] == 1
        assert stats["stats_by_type"]["ingest"]["successful"] == 1


# ---------------------------------------------------------------------------
# Start / Stop
# ---------------------------------------------------------------------------

class TestStartStop:
    def test_start_stop(self, scheduler):
        scheduler.add_job("ss", "ingest", "0 0 * * *")
        scheduler.start()
        assert scheduler.scheduler.running
        scheduler.stop()
        assert not scheduler.scheduler.running

    def test_double_stop_safe(self, scheduler):
        # Should not raise
        scheduler.stop()
        scheduler.stop()
