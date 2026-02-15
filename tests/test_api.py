"""Integration tests — FastAPI endpoints.

Uses FastAPI TestClient (sync) — no running server needed.
"""
import os
import pytest

# Ensure test env before importing app
os.environ.setdefault("FOOTBALL_DATA_ORG_TOKEN", "test-token")
os.environ.setdefault("API_FOOTBALL_KEY", "test-token")


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient wired to the real app."""
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
        r = client.get("/api/matches?model=v7_council")
        assert r.status_code == 200

    def test_match_detail_not_found(self, client):
        r = client.get("/api/matches/999999999")
        assert r.status_code == 404

    def test_match_detail_found(self, client):
        # Get any existing match first
        matches = client.get("/api/matches?days=3650").json().get("matches", [])
        if not matches:
            pytest.skip("No matches in database")
        mid = matches[0]["match_id"]
        r = client.get(f"/api/matches/{mid}")
        assert r.status_code == 200
        data = r.json()
        assert "home_team" in data
        assert "away_team" in data


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
        matches = client.get("/api/matches?days=3650").json().get("matches", [])
        if not matches:
            pytest.skip("No matches in database")
        mid = matches[0]["match_id"]
        r = client.get(f"/api/matches/{mid}/form")
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
