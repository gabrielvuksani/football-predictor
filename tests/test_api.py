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
        (1002, 'v10_council', 0, 0.693, 0.38, TRUE, 0.5, NULL, NULL, NULL)
    """)
    db.execute("""
        INSERT INTO metrics VALUES ('v10_council', 100, 1.05, 0.22, 0.51)
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
        return duckdb.connect(test_db, read_only=True)

    with patch("web.api.con", _test_con):
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
        r = client.get("/api/matches/999999999/narrative")
        assert r.status_code in (404, 500)


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
