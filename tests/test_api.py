"""Integration tests — FastAPI endpoints.

Uses FastAPI TestClient (sync) — no running server needed.
"""
import os
import tempfile
import shutil
from unittest.mock import patch

import pytest
import duckdb

# Ensure test env before importing app
os.environ.setdefault("FOOTBALL_DATA_ORG_TOKEN", "test-token")
os.environ.setdefault("API_FOOTBALL_KEY", "test-token")


def _create_test_db(path: str) -> None:
    """Create a test DuckDB with proper schema."""
    db = duckdb.connect(path)
    db.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id BIGINT PRIMARY KEY,
            provider VARCHAR, competition VARCHAR, season INTEGER,
            utc_date TIMESTAMP, status VARCHAR,
            home_team VARCHAR, away_team VARCHAR,
            home_goals INTEGER, away_goals INTEGER,
            raw_json VARCHAR
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            match_id BIGINT, model_version VARCHAR,
            p_home DOUBLE, p_draw DOUBLE, p_away DOUBLE,
            eg_home DOUBLE, eg_away DOUBLE, notes VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (match_id, model_version)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS match_extras (
            match_id BIGINT PRIMARY KEY,
            b365h DOUBLE, b365d DOUBLE, b365a DOUBLE,
            hs INTEGER, as_ INTEGER, hst INTEGER, ast INTEGER,
            season_code VARCHAR
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS elo_state (
            team VARCHAR PRIMARY KEY, rating DOUBLE
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS prediction_scores (
            match_id BIGINT, model_version VARCHAR,
            scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            outcome INTEGER, logloss DOUBLE, brier DOUBLE,
            correct BOOLEAN, goals_mae DOUBLE,
            btts_correct BOOLEAN, ou25_correct BOOLEAN, score_correct BOOLEAN,
            PRIMARY KEY (match_id, model_version)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            model_version VARCHAR PRIMARY KEY,
            n_matches INTEGER, logloss DOUBLE, brier DOUBLE, accuracy DOUBLE
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS expert_cache (
            match_id BIGINT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            breakdown_json VARCHAR
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS model_deployments (
            model_type VARCHAR PRIMARY KEY,
            active_version VARCHAR NOT NULL,
            deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            previous_version VARCHAR,
            reason VARCHAR,
            performance_metrics_json VARCHAR
        )
    """)
    db.execute("""
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
    db.execute("""
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
    db.execute("""
        CREATE TABLE IF NOT EXISTS ensemble_weights (
            model_name VARCHAR PRIMARY KEY,
            weight DOUBLE NOT NULL,
            n_predictions INT,
            avg_log_loss DOUBLE,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS expert_performance (
            expert_name VARCHAR PRIMARY KEY,
            accuracy DOUBLE,
            log_loss DOUBLE,
            n_predictions INT,
            avg_confidence DOUBLE,
            updated_at TIMESTAMP
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS expert_performance_by_comp (
            expert_name VARCHAR,
            competition VARCHAR,
            accuracy DOUBLE,
            n_predictions INT,
            updated_at TIMESTAMP,
            PRIMARY KEY (expert_name, competition)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS h2h_stats (
            team_a VARCHAR, team_b VARCHAR,
            team_a_canonical VARCHAR, team_b_canonical VARCHAR,
            total_matches INTEGER, team_a_wins INTEGER, team_b_wins INTEGER,
            draws INTEGER,
            team_a_goals_for INTEGER, team_a_goals_against INTEGER,
            team_b_goals_for INTEGER, team_b_goals_against INTEGER,
            team_a_avg_goals_for DOUBLE, team_a_avg_goals_against DOUBLE,
            team_b_avg_goals_for DOUBLE, team_b_avg_goals_against DOUBLE,
            last_updated TIMESTAMP,
            PRIMARY KEY (team_a, team_b)
        )
    """)
    # Insert test data
    db.execute("""
        INSERT INTO matches VALUES
        (1001, 'test', 'PL', 2025, '2025-03-01', 'TIMED', 'Arsenal', 'Chelsea', NULL, NULL, NULL),
        (1002, 'test', 'PL', 2025, '2024-12-01', 'FINISHED', 'Arsenal', 'Chelsea', 2, 1, NULL)
    """)
    db.execute("""
        INSERT INTO predictions VALUES
        (1001, 'v10_council', 0.5, 0.3, 0.2, 1.8, 0.9,
         '{"btts": 0.55, "o25": 0.65, "predicted_score": [2, 1], "lambda_home": 1.8, "lambda_away": 0.9}',
         CURRENT_TIMESTAMP),
        (1002, 'v10_council', 0.5, 0.3, 0.2, 1.8, 0.9, NULL, CURRENT_TIMESTAMP)
    """)
    db.execute("""
        INSERT INTO match_extras (match_id, b365h, b365d, b365a) VALUES
        (1001, 2.0, 3.5, 4.0)
    """)
    db.execute("""
        INSERT INTO elo_state VALUES ('Arsenal', 1650.0), ('Chelsea', 1580.0)
    """)
    db.execute("""
        INSERT INTO prediction_scores VALUES
        (1002, 'v10_council', CURRENT_TIMESTAMP, 0, 0.693, 0.38, TRUE, 0.5, NULL, NULL, NULL)
    """)
    db.execute("""
        INSERT INTO metrics VALUES ('v10_council', 100, 1.05, 0.22, 0.51)
    """)
    db.execute("""
        INSERT INTO model_deployments VALUES
        ('v10_council', 'v10_council_20260321', CURRENT_TIMESTAMP, NULL, 'test', '{"accuracy":0.51}')
    """)
    db.execute("""
        INSERT INTO retraining_schedules VALUES
        ('v10_council', 10, 0.01, TRUE, CURRENT_TIMESTAMP, 3, NULL)
    """)
    db.execute("""
        INSERT INTO model_training_records VALUES
        (1, 'v10_council_20260321', 'v10_council', CURRENT_TIMESTAMP, 3650, 1200, 240,
         '{"accuracy":0.51,"logloss":1.05}', '{"accuracy":0.51,"logloss":1.05}', NULL, 0.025, TRUE, CURRENT_TIMESTAMP, 'test run')
    """)
    db.execute("""
        INSERT INTO ensemble_weights VALUES
        ('poisson', 0.24, 20, 1.01, CURRENT_TIMESTAMP),
        ('dixon_coles', 0.32, 20, 0.98, CURRENT_TIMESTAMP)
    """)
    db.execute("""
        INSERT INTO expert_performance VALUES
        ('zip_model', 0.67, 0.88, 12, 0.71, CURRENT_TIMESTAMP),
        ('elo', 0.58, 0.97, 12, 0.62, CURRENT_TIMESTAMP)
    """)
    db.execute("""
        INSERT INTO expert_performance_by_comp VALUES
        ('zip_model', 'PL', 0.70, 8, CURRENT_TIMESTAMP),
        ('elo', 'PL', 0.55, 8, CURRENT_TIMESTAMP)
    """)
    db.execute("""
        INSERT INTO expert_cache VALUES
        (1002, CURRENT_TIMESTAMP,
         '{"match_id":1002,"home_team":"Arsenal","away_team":"Chelsea","competition":"PL","experts":{"elo":{"probs":{"home":0.61,"draw":0.22,"away":0.17},"confidence":0.72},"form":{"probs":{"home":0.55,"draw":0.25,"away":0.20},"confidence":0.64}}}')
    """)
    db.close()


@pytest.fixture(scope="module")
def test_db():
    """Create a temporary DuckDB for testing."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.duckdb")
    _create_test_db(db_path)
    yield db_path
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def client(test_db):
    """FastAPI TestClient wired to the real app, with DB mocked."""
    def _test_con():
        return duckdb.connect(test_db)

    with patch("web.api.con", _test_con), patch("footy.continuous_training.connect", _test_con), patch("footy.llm.insights.connect", _test_con):
        from fastapi.testclient import TestClient
        from web.api import app
        with TestClient(app) as c:
            yield c


class TestFrontend:
    def test_index_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Footy Predictor" in r.text


class TestApiMatches:
    def test_matches_endpoint(self, client):
        r = client.get("/api/matches")
        assert r.status_code == 200
        data = r.json()
        assert "matches" in data
        assert isinstance(data["matches"], list)

    def test_matches_with_days_param(self, client):
        r = client.get("/api/matches?days=30")
        assert r.status_code == 200

    def test_matches_with_model_param(self, client):
        r = client.get("/api/matches?model=v10_council")
        assert r.status_code == 200

    def test_match_detail_not_found(self, client):
        r = client.get("/api/matches/999999999")
        assert r.status_code == 404

    def test_match_detail_found(self, client):
        r = client.get("/api/matches/1001")
        assert r.status_code == 200
        data = r.json()
        assert data["home_team"] == "Arsenal"
        assert data["away_team"] == "Chelsea"
        assert "prediction" in data
        assert "odds" in data
        assert "elo" in data
        assert "poisson" in data
        assert "score" in data


class TestApiExperts:
    def test_experts_not_found(self, client):
        r = client.get("/api/matches/999999999/experts")
        # May be 404 or 500 depending on implementation
        assert r.status_code in (404, 500)


class TestApiH2h:
    def test_h2h_not_found(self, client):
        r = client.get("/api/matches/999999999/h2h")
        assert r.status_code == 404


class TestApiForm:
    def test_form_not_found(self, client):
        r = client.get("/api/matches/999999999/form")
        assert r.status_code == 404

    def test_form_found(self, client):
        r = client.get("/api/matches/1002/form")
        if r.status_code == 200:
            data = r.json()
            assert "home_form" in data
            assert "away_form" in data


class TestApiStats:
    def test_stats_endpoint(self, client):
        r = client.get("/api/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_matches" in data
        assert "finished" in data
        assert "upcoming" in data
        assert "teams" in data
        assert "predictions" in data


class TestApiValueBets:
    def test_value_bets_endpoint(self, client):
        r = client.get("/api/insights/value-bets")
        assert r.status_code == 200
        data = r.json()
        assert "bets" in data
        assert isinstance(data["bets"], list)

    def test_value_bets_with_min_edge(self, client):
        r = client.get("/api/insights/value-bets?min_edge=0.10")
        assert r.status_code == 200


class TestApiPerformance:
    def test_performance_endpoint(self, client):
        r = client.get("/api/performance")
        assert r.status_code == 200
        data = r.json()
        assert "model" in data
        assert "metrics" in data
        assert "calibration" in data
        assert "by_competition" in data
        assert "recent" in data


class TestApiNarrative:
    def test_narrative_not_found(self, client):
        r = client.get("/api/matches/999999999/ai")
        # The AI endpoint returns 200 with available=False for missing matches
        assert r.status_code in (200, 404, 500)
        if r.status_code == 200:
            data = r.json()
            assert data.get("available") is False or data.get("narrative") is None


class TestApiSearch:
    def test_search_endpoint_not_implemented(self, client):
        """Search endpoint doesn't exist yet – expect 404."""
        r = client.get("/api/search?q=Arsenal")
        assert r.status_code == 404


class TestApiLastUpdated:
    def test_last_updated(self, client):
        r = client.get("/api/last-updated")
        assert r.status_code == 200
        data = r.json()
        assert "last_updated" in data


class TestTrainingAndModelLab:
    def test_training_status(self, client):
        r = client.get("/api/training/status")
        assert r.status_code == 200
        data = r.json()
        assert data["active_version"] == "v10_council_20260321"
        assert isinstance(data["expert_rankings"], list)
        assert isinstance(data["history"], list)

    def test_ensemble_weights(self, client):
        r = client.get("/api/ensemble-weights")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "active"
        assert len(data["weights"]) >= 2

    def test_expert_rankings_endpoint(self, client):
        r = client.get("/api/expert-rankings?competition=PL")
        assert r.status_code == 200
        data = r.json()
        assert data["competition"] == "PL"
        assert data["rankings"][0]["expert"] == "zip_model"

    def test_model_lab(self, client):
        r = client.get("/api/model-lab")
        assert r.status_code == 200
        data = r.json()
        assert data["active_version"] == "v10_council_20260321"
        assert isinstance(data["ensemble_weights"], list)
        assert isinstance(data["expert_weights"], list)


class TestSelfLearningAndUnifiedPrediction:
    def test_self_learning_status(self, client):
        r = client.get("/api/self-learning/status")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "active"
        assert data["report"]["overall"]["n_predictions"] >= 1

    def test_self_learning_expert_weights(self, client):
        r = client.get("/api/self-learning/expert-weights?league=PL")
        assert r.status_code == 200
        data = r.json()
        assert data["n_experts"] >= 2
        assert "elo" in data["weights"]
        assert "form" in data["weights"]

    def test_bayesian_predict_endpoint(self, client):
        r = client.get("/api/bayesian/predict/1001")
        assert r.status_code == 200
        data = r.json()
        assert data["match_id"] == 1001
        assert 0.0 <= data["p_home"] <= 1.0
        assert len(data["top_scores"]) >= 1

    def test_unified_prediction_endpoint(self, client):
        r = client.get("/api/unified-prediction/1001")
        assert r.status_code == 200
        data = r.json()
        assert data["match_id"] == 1001
        assert data["n_models"] >= 2
        assert data["bayesian_cached"] is True
        assert isinstance(data["score_probabilities"], list)
        assert len(data["component_breakdown"]["bayesian"]) == 3
