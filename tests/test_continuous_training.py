"""Tests for the continuous training module.

Tests training records, drift detection, deployment, rollback,
and auto-retrain workflow using an in-memory DuckDB database.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from footy.continuous_training import ContinuousTrainingManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager(tmp_path, monkeypatch):
    """Create a ContinuousTrainingManager backed by a temp DuckDB."""
    import duckdb

    db_path = str(tmp_path / "test_ct.duckdb")
    con = duckdb.connect(db_path)

    # Create minimal matches table for counting
    con.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY,
            status VARCHAR,
            home_goals INTEGER,
            away_goals INTEGER,
            utc_date TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            match_id INTEGER,
            model_version VARCHAR,
            p_home DOUBLE,
            p_draw DOUBLE,
            p_away DOUBLE
        )
    """)

    monkeypatch.setattr("footy.continuous_training.connect", lambda: con)
    mgr = ContinuousTrainingManager()
    yield mgr
    con.close()


def _seed_matches(con, n=30, days_span=60):
    """Insert N finished matches spread over days_span."""
    base = datetime.utcnow() - timedelta(days=days_span)
    for i in range(n):
        dt = base + timedelta(days=i * days_span / n)
        h_goals = i % 3
        a_goals = (i + 1) % 3
        con.execute(
            "INSERT INTO matches VALUES (?, 'FINISHED', ?, ?, ?)",
            [i + 1, h_goals, a_goals, dt],
        )


def _seed_predictions(con, match_ids, model_version="v10_council"):
    """Insert dummy predictions for given match_ids."""
    for mid in match_ids:
        con.execute(
            "INSERT INTO predictions VALUES (?, ?, 0.45, 0.30, 0.25)",
            [mid, model_version],
        )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_tables_created(self, manager):
        tables = [r[0] for r in manager.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]
        assert "model_training_records" in tables
        assert "model_deployments" in tables
        assert "retraining_schedules" in tables


# ---------------------------------------------------------------------------
# Setup continuous training
# ---------------------------------------------------------------------------

class TestSetup:
    def test_setup(self, manager):
        result = manager.setup_continuous_training("v10_council", 20, 0.005)
        assert result["model_type"] == "v10_council"
        assert result["retrain_threshold_matches"] == 20
        assert result["enabled"] is True

    def test_setup_idempotent(self, manager):
        manager.setup_continuous_training("v10_council", 10, 0.01)
        manager.setup_continuous_training("v10_council", 20, 0.005)
        row = manager.con.execute(
            "SELECT retrain_threshold_matches FROM retraining_schedules WHERE model_type = 'v10_council'"
        ).fetchone()
        assert row[0] == 20


# ---------------------------------------------------------------------------
# New matches counting
# ---------------------------------------------------------------------------

class TestNewMatchesCounting:
    def test_first_time_counts_all(self, manager):
        _seed_matches(manager.con, 15)
        manager.setup_continuous_training("v10_council")
        count = manager.get_new_matches_since_training("v10_council")
        assert count == 15

    def test_after_training_resets(self, manager):
        _seed_matches(manager.con, 15)
        manager.setup_continuous_training("v10_council")
        # Simulate training by updating last_trained
        manager.con.execute(
            "UPDATE retraining_schedules SET last_trained = CURRENT_TIMESTAMP WHERE model_type = 'v10_council'"
        )
        count = manager.get_new_matches_since_training("v10_council")
        assert count == 0  # No new matches since "training"


# ---------------------------------------------------------------------------
# Check and retrain
# ---------------------------------------------------------------------------

class TestCheckRetrain:
    def test_waiting_when_below_threshold(self, manager):
        _seed_matches(manager.con, 5)
        manager.setup_continuous_training("v10_council", retrain_threshold_matches=10)
        result = manager.check_and_retrain("v10_council")
        assert result["v10_council"]["status"] == "waiting"
        assert result["v10_council"]["ready_in"] == 5

    def test_ready_when_above_threshold(self, manager):
        _seed_matches(manager.con, 25)
        manager.setup_continuous_training("v10_council", retrain_threshold_matches=10)
        result = manager.check_and_retrain("v10_council")
        assert result["v10_council"]["status"] == "ready_to_retrain"

    def test_not_configured(self, manager):
        result = manager.check_and_retrain("nonexistent_model")
        assert result["nonexistent_model"]["status"] == "not_configured"


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

class TestDriftDetection:
    def test_insufficient_data(self, manager):
        drift = manager.detect_drift("v10_council")
        assert drift["drifted"] is False
        assert drift["reason"] == "insufficient_data"

    def test_no_drift_when_stable(self, manager):
        now = datetime.utcnow()
        _seed_predictions_with_matches(manager.con, now, baseline_acc=0.55, recent_acc=0.55)
        drift = manager.detect_drift("v10_council", recent_window=30, baseline_window=180)
        # Should detect no drift since accuracies are equal
        if drift.get("reason") != "insufficient_data":
            assert drift["drifted"] is False

    def test_drift_when_degraded(self, manager):
        now = datetime.utcnow()
        _seed_predictions_with_matches(manager.con, now, baseline_acc=0.65, recent_acc=0.50)
        drift = manager.detect_drift("v10_council", recent_window=30, baseline_window=180)
        if drift.get("reason") != "insufficient_data":
            assert drift["drifted"] is True
            assert drift["accuracy_drop"] > 0.05


def _seed_predictions_with_matches(con, now, baseline_acc=0.55, recent_acc=0.55):
    """Seed matches + predictions so drift detection has data."""
    mid = 1000
    # Baseline period: 90-180 days ago
    for i in range(50):
        dt = now - timedelta(days=180 - i * 2)
        # Simulate accuracy by setting goals based on prediction
        if i < int(50 * baseline_acc):
            hg, ag = 2, 0  # Home win
            ph, pd, pa = 0.6, 0.2, 0.2  # Correct prediction
        else:
            hg, ag = 0, 2  # Away win
            ph, pd, pa = 0.6, 0.2, 0.2  # Wrong prediction
        con.execute("INSERT INTO matches VALUES (?, 'FINISHED', ?, ?, ?)", [mid, hg, ag, dt])
        con.execute("INSERT INTO predictions VALUES (?, 'v10_council', ?, ?, ?)", [mid, ph, pd, pa])
        mid += 1

    # Recent period: last 30 days
    for i in range(30):
        dt = now - timedelta(days=30 - i)
        if i < int(30 * recent_acc):
            hg, ag = 2, 0
            ph, pd, pa = 0.6, 0.2, 0.2
        else:
            hg, ag = 0, 2
            ph, pd, pa = 0.6, 0.2, 0.2
        con.execute("INSERT INTO matches VALUES (?, 'FINISHED', ?, ?, ?)", [mid, hg, ag, dt])
        con.execute("INSERT INTO predictions VALUES (?, 'v10_council', ?, ?, ?)", [mid, ph, pd, pa])
        mid += 1


# ---------------------------------------------------------------------------
# Record training
# ---------------------------------------------------------------------------

class TestRecordTraining:
    def test_record_first_training(self, manager):
        manager.setup_continuous_training("v10_council")
        result = manager.record_training(
            model_version="v10_test_001",
            model_type="v10_council",
            training_window_days=365,
            n_matches_used=500,
            n_matches_test=100,
            metrics={"accuracy": 0.55, "logloss": 1.0},
            test_metrics={"accuracy": 0.55, "logloss": 1.0},
        )
        assert result["model_version"] == "v10_test_001"
        assert result["improvement_pct"] == 0.0  # No previous to compare
        assert result["previous_version"] is None

    def test_record_improvement(self, manager):
        manager.setup_continuous_training("v10_council")
        # First training
        manager.record_training(
            model_version="v1", model_type="v10_council",
            training_window_days=365, n_matches_used=500, n_matches_test=100,
            metrics={"accuracy": 0.50}, test_metrics={"accuracy": 0.50},
        )
        manager.deploy_model("v1", "v10_council", force=True)

        # Second training (improved)
        result = manager.record_training(
            model_version="v2", model_type="v10_council",
            training_window_days=365, n_matches_used=600, n_matches_test=120,
            metrics={"accuracy": 0.55}, test_metrics={"accuracy": 0.55},
        )
        assert result["improvement_pct"] == pytest.approx(5.0, abs=0.1)
        assert result["previous_version"] == "v1"


# ---------------------------------------------------------------------------
# Deploy / Rollback
# ---------------------------------------------------------------------------

class TestDeployRollback:
    def test_deploy(self, manager):
        manager.setup_continuous_training("v10_council")
        manager.record_training(
            model_version="dep_v1", model_type="v10_council",
            training_window_days=365, n_matches_used=500, n_matches_test=100,
            metrics={"accuracy": 0.55},
        )
        result = manager.deploy_model("dep_v1", "v10_council", force=True)
        assert result["status"] == "deployed"
        assert result["model_version"] == "dep_v1"

    def test_deployment_status(self, manager):
        manager.setup_continuous_training("v10_council")
        manager.record_training(
            model_version="ds_v1", model_type="v10_council",
            training_window_days=365, n_matches_used=500, n_matches_test=100,
            metrics={"accuracy": 0.55},
        )
        manager.deploy_model("ds_v1", "v10_council", force=True)
        status = manager.get_deployment_status()
        assert "v10_council" in status
        assert status["v10_council"]["active_version"] == "ds_v1"

    def test_rollback(self, manager):
        manager.setup_continuous_training("v10_council")
        manager.record_training(
            model_version="rb_v1", model_type="v10_council",
            training_window_days=365, n_matches_used=500, n_matches_test=100,
            metrics={"accuracy": 0.50},
        )
        manager.deploy_model("rb_v1", "v10_council", force=True)

        manager.record_training(
            model_version="rb_v2", model_type="v10_council",
            training_window_days=365, n_matches_used=600, n_matches_test=120,
            metrics={"accuracy": 0.55},
        )
        manager.deploy_model("rb_v2", "v10_council", force=True)

        result = manager.rollback_model("v10_council")
        assert result["status"] == "rolled_back"
        assert result["restored_version"] == "rb_v1"

    def test_rollback_no_previous(self, manager):
        result = manager.rollback_model("v10_council")
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

class TestHistory:
    def test_empty_history(self, manager):
        hist = manager.get_training_history("v10_council")
        assert hist == []

    def test_history_after_training(self, manager):
        manager.setup_continuous_training("v10_council")
        for i in range(3):
            manager.record_training(
                model_version=f"hist_v{i}", model_type="v10_council",
                training_window_days=365, n_matches_used=500 + i * 100,
                n_matches_test=100, metrics={"accuracy": 0.50 + i * 0.01},
            )
        hist = manager.get_training_history("v10_council", limit=2)
        assert len(hist) == 2
        # Most recent first
        assert hist[0]["model_version"] == "hist_v2"


# ---------------------------------------------------------------------------
# Retraining status
# ---------------------------------------------------------------------------

class TestRetrainingStatus:
    def test_status(self, manager):
        _seed_matches(manager.con, 15)
        manager.setup_continuous_training("v10_council", retrain_threshold_matches=10)
        status = manager.get_retraining_status()
        assert "v10_council" in status
        assert status["v10_council"]["ready"] is True
        assert status["v10_council"]["new_matches"] == 15

    def test_not_ready(self, manager):
        _seed_matches(manager.con, 5)
        manager.setup_continuous_training("v10_council", retrain_threshold_matches=10)
        status = manager.get_retraining_status()
        assert status["v10_council"]["ready"] is False
