from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest


def _create_export_test_db(path: Path) -> None:
    db = duckdb.connect(str(path))
    db.execute("""
        CREATE TABLE matches (
            match_id BIGINT PRIMARY KEY,
            provider VARCHAR,
            competition VARCHAR,
            season INTEGER,
            utc_date TIMESTAMP,
            status VARCHAR,
            home_team VARCHAR,
            away_team VARCHAR,
            home_goals INTEGER,
            away_goals INTEGER,
            raw_json VARCHAR
        )
    """)
    db.execute("""
        CREATE TABLE predictions (
            match_id BIGINT,
            model_version VARCHAR,
            p_home DOUBLE,
            p_draw DOUBLE,
            p_away DOUBLE,
            eg_home DOUBLE,
            eg_away DOUBLE,
            notes VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (match_id, model_version)
        )
    """)
    db.execute("""
        CREATE TABLE match_extras (
            match_id BIGINT PRIMARY KEY,
            competition VARCHAR,
            season_code VARCHAR,
            b365h DOUBLE,
            b365d DOUBLE,
            b365a DOUBLE
        )
    """)
    db.execute("""
        CREATE TABLE elo_state (
            team VARCHAR PRIMARY KEY,
            rating DOUBLE
        )
    """)
    db.execute("""
        CREATE TABLE prediction_scores (
            match_id BIGINT,
            model_version VARCHAR,
            scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            outcome INTEGER,
            logloss DOUBLE,
            brier DOUBLE,
            correct BOOLEAN,
            goals_mae DOUBLE,
            eg_home DOUBLE,
            eg_away DOUBLE,
            btts_correct BOOLEAN,
            ou25_correct BOOLEAN,
            score_correct BOOLEAN,
            p_btts DOUBLE,
            p_o25 DOUBLE,
            predicted_score_h INT,
            predicted_score_a INT,
            PRIMARY KEY (match_id, model_version)
        )
    """)
    db.execute("""
        CREATE TABLE metrics (
            model_version VARCHAR PRIMARY KEY,
            n_matches INTEGER,
            logloss DOUBLE,
            brier DOUBLE,
            accuracy DOUBLE
        )
    """)
    db.execute("""
        CREATE TABLE expert_cache (
            match_id BIGINT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            breakdown_json VARCHAR
        )
    """)
    db.execute("""
        CREATE TABLE h2h_stats (
            team_a VARCHAR,
            team_b VARCHAR,
            team_a_canonical VARCHAR,
            team_b_canonical VARCHAR,
            total_matches INTEGER,
            team_a_wins INTEGER,
            team_b_wins INTEGER,
            draws INTEGER,
            team_a_goals_for INTEGER,
            team_a_goals_against INTEGER,
            team_b_goals_for INTEGER,
            team_b_goals_against INTEGER,
            team_a_avg_goals_for DOUBLE,
            team_a_avg_goals_against DOUBLE,
            team_b_avg_goals_for DOUBLE,
            team_b_avg_goals_against DOUBLE,
            last_updated TIMESTAMP,
            PRIMARY KEY (team_a, team_b)
        )
    """)
    db.execute("""
        CREATE TABLE model_deployments (
            model_type VARCHAR PRIMARY KEY,
            active_version VARCHAR NOT NULL,
            deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            previous_version VARCHAR,
            reason VARCHAR,
            performance_metrics_json VARCHAR
        )
    """)
    db.execute("""
        CREATE TABLE retraining_schedules (
            model_type VARCHAR PRIMARY KEY,
            retrain_threshold_matches INT DEFAULT 10,
            performance_threshold_improvement DOUBLE DEFAULT 0.01,
            enabled BOOLEAN DEFAULT TRUE,
            last_trained TIMESTAMP,
            last_retrain_matches INT DEFAULT 0,
            next_scheduled_retrain TIMESTAMP
        )
    """)
    db.execute("""
        CREATE TABLE model_training_records (
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

    db.execute("""
        INSERT INTO matches VALUES
        (1001, 'test', 'PL', 2025, '2026-03-22 15:00:00', 'TIMED', 'Arsenal', 'Chelsea', NULL, NULL, NULL),
        (1002, 'test', 'PL', 2025, '2026-03-18 19:45:00', 'FINISHED', 'Arsenal', 'Chelsea', 2, 1, NULL)
    """)
    db.execute("""
        INSERT INTO predictions VALUES
        (1001, 'v13_oracle', 0.57, 0.24, 0.19, 1.7, 1.0,
         '{"btts":0.58,"o25":0.62,"predicted_score":[2,1],"lambda_home":1.7,"lambda_away":1.0}', CURRENT_TIMESTAMP),
        (1002, 'v13_oracle', 0.54, 0.26, 0.20, 1.6, 0.9,
         '{"btts":0.51,"o25":0.56,"predicted_score":[2,1],"lambda_home":1.6,"lambda_away":0.9}', CURRENT_TIMESTAMP)
    """)
    db.execute("""
        INSERT INTO match_extras VALUES
        (1001, 'PL', '2526', 1.95, 3.60, 4.20)
    """)
    db.execute("""
        INSERT INTO elo_state VALUES ('Arsenal', 1650.0), ('Chelsea', 1580.0)
    """)
    db.execute("""
        INSERT INTO prediction_scores VALUES
        (1002, 'v13_oracle', CURRENT_TIMESTAMP, 0, 0.616, 0.31, TRUE, 0.4, 1.6, 0.9, TRUE, TRUE, TRUE, 0.51, 0.56, 2, 1)
    """)
    db.execute("""
        INSERT INTO metrics VALUES ('v13_oracle', 1, 0.616, 0.31, 1.0)
    """)
    db.execute("""
        INSERT INTO expert_cache VALUES
        (1001, CURRENT_TIMESTAMP,
         '{"match_id":1001,"home_team":"Arsenal","away_team":"Chelsea","competition":"PL","experts":{"elo":{"probs":{"home":0.61,"draw":0.22,"away":0.17},"confidence":0.72}}}')
    """)
    db.execute("""
        INSERT INTO h2h_stats VALUES
        ('Arsenal', 'Chelsea', 'Arsenal', 'Chelsea', 5, 3, 1, 1, 8, 5, 5, 8, 1.6, 1.0, 1.0, 1.6, CURRENT_TIMESTAMP)
    """)
    db.execute("""
        INSERT INTO model_deployments VALUES
        ('v13_oracle', 'v13_oracle_20260321', CURRENT_TIMESTAMP, NULL, 'test', '{"accuracy":1.0}')
    """)
    db.execute("""
        INSERT INTO retraining_schedules VALUES
        ('v13_oracle', 10, 0.01, TRUE, CURRENT_TIMESTAMP, 1, NULL)
    """)
    db.execute("""
        INSERT INTO model_training_records VALUES
        (1, 'v13_oracle_20260321', 'v13_oracle', CURRENT_TIMESTAMP, 3650, 1200, 240,
         '{"accuracy":1.0,"logloss":0.616}', '{"accuracy":1.0,"logloss":0.616}', NULL, 0.025, TRUE, CURRENT_TIMESTAMP, 'test run')
    """)
    db.close()


def _fake_detect_drift(self: Any, model_type: str) -> dict[str, float | bool]:
    return {"drifted": False, "recent_accuracy": 1.0, "baseline_accuracy": 0.8}


def _fake_get_expert_rankings(self: Any, competition: str | None = None) -> list[dict[str, float | int | str]]:
    return [{"expert": "elo", "accuracy": 0.7, "n": 8}]


def _fake_get_new_matches_since_training(self: Any, model_type: str) -> int:
    return 1


def _fake_round_summary(competition_code: str = "PL") -> dict[str, Any]:
    return {
        "competition": competition_code,
        "matches": 1,
        "summary": f"{competition_code} round summary",
        "headline_pick": "Arsenal vs Chelsea",
        "predictions": [{"match_id": 1001, "home": "Arsenal", "away": "Chelsea", "date": "2026-03-22", "pred": "Home", "probs": "57%/24%/19%"}],
    }


def _fake_post_match_review(days_back: int = 7, competition_code: str | None = None) -> dict[str, Any]:
    return {
        "matches_reviewed": 1,
        "correct": 1,
        "accuracy": 1.0,
        "review": "Strong performance.",
        "misses": [],
    }


def _fake_btts_ou() -> dict[str, list[dict[str, Any]]]:
    entry: dict[str, Any] = {"match_id": 1001, "home_team": "Arsenal", "away_team": "Chelsea", "competition": "PL", "date": "2026-03-22", "btts_prob": 0.58, "o25_prob": 0.62}
    return {"btts_likely": [entry], "btts_unlikely": [], "over25": [entry], "under25": []}


def _fake_accumulators(min_prob: float = 0.55, max_legs: int = 5) -> list[dict[str, Any]]:
    return [{"type": "🛡️ Safe", "legs": [{"match_id": 1001, "pick": "Home", "prob": 0.57, "odds": 1.95}], "combined_prob": 0.57, "combined_odds": 1.95}]


def _fake_form_table(competition_code: str, last_n: int = 6) -> list[dict[str, Any]]:
    return [{"team": "Arsenal", "played": 1, "w": 1, "d": 0, "l": 0, "ppg": 3.0, "gf": 2, "ga": 1, "gd": 1, "btts_pct": 100, "o25_pct": 100}]


def _fake_accuracy(days_back: int = 30) -> dict[str, Any]:
    return {"total": 1, "accuracy": 1.0, "by_competition": [{"competition": "PL", "n": 1, "accuracy": 1.0}], "by_confidence": {"high": {"n": 1, "accuracy": 1.0}}, "brier": 0.31}


def test_pages_export_writes_static_tab_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "export.duckdb"
    out_dir = tmp_path / "site"
    _create_export_test_db(db_path)

    monkeypatch.setenv("DB_PATH", str(db_path))

    from footy.config import settings
    settings.cache_clear()

    import footy.continuous_training as continuous_training
    from footy.continuous_training import ContinuousTrainingManager
    monkeypatch.setattr(continuous_training, "connect", lambda: duckdb.connect(str(db_path)))
    monkeypatch.setattr(ContinuousTrainingManager, "detect_drift", _fake_detect_drift)
    monkeypatch.setattr(ContinuousTrainingManager, "get_expert_rankings", _fake_get_expert_rankings)
    monkeypatch.setattr(ContinuousTrainingManager, "get_new_matches_since_training", _fake_get_new_matches_since_training)

    import footy.llm.insights as insights
    monkeypatch.setattr(insights, "btts_ou_insights", _fake_btts_ou)
    monkeypatch.setattr(insights, "build_accumulators", _fake_accumulators)
    monkeypatch.setattr(insights, "league_form_table", _fake_form_table)
    monkeypatch.setattr(insights, "prediction_accuracy_stats", _fake_accuracy)
    monkeypatch.setattr(insights, "league_round_summary", _fake_round_summary)
    monkeypatch.setattr(insights, "post_match_review", _fake_post_match_review)

    from footy.cli.pages_cmds import export
    export(out=str(out_dir), days=14)

    expected_files = [
        out_dir / "api" / "btts-ou.json",
        out_dir / "api" / "accumulators.json",
        out_dir / "api" / "accuracy.json",
        out_dir / "api" / "post-match-review.json",
        out_dir / "api" / "training" / "status.json",
        out_dir / "api" / "form-table" / "PL.json",
        out_dir / "api" / "round-preview" / "PL.json",
    ]
    for file_path in expected_files:
        assert file_path.exists(), f"Missing exported file: {file_path}"

    training = json.loads((out_dir / "api" / "training" / "status.json").read_text())
    assert "v15_architect" in training["active_version"] or "v13_oracle" in training["active_version"]
    assert "expert_rankings" in training
    assert "drift" in training

    accuracy = json.loads((out_dir / "api" / "accuracy.json").read_text())
    assert accuracy["total"] >= 1

    preview = json.loads((out_dir / "api" / "round-preview" / "PL.json").read_text())
    assert preview["competition"] == "PL"
    assert preview["matches"] == 1
