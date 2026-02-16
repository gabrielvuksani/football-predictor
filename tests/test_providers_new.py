"""Tests for new API providers — The Odds API, FPL, TheSportsDB.

These tests don't make real HTTP calls. They test:
    - Module imports & function signatures
    - Response parsing logic with mock data
    - Graceful degradation (missing API keys, API errors)
    - DB ingestion helpers
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import duckdb
import pytest


# ── The Odds API ──────────────────────────────────────────────────────────────

class TestTheOddsApiImport:
    def test_import_module(self):
        from footy.providers import the_odds_api
        assert hasattr(the_odds_api, "fetch_odds")
        assert hasattr(the_odds_api, "fetch_all_odds")
        assert hasattr(the_odds_api, "parse_odds_for_match")
        assert hasattr(the_odds_api, "ingest_odds_to_db")

    def test_sport_keys(self):
        from footy.providers.the_odds_api import SPORT_KEYS
        assert "PL" in SPORT_KEYS
        assert "PD" in SPORT_KEYS
        assert "SA" in SPORT_KEYS
        assert "BL1" in SPORT_KEYS

    def test_unknown_competition_returns_empty(self):
        from footy.providers.the_odds_api import fetch_odds
        # Without API key, should return empty list gracefully
        with patch.dict("os.environ", {"THE_ODDS_API_KEY": ""}):
            result = fetch_odds("NONEXISTENT")
            assert result == []


class TestTheOddsApiParsing:
    """Test parse_odds_for_match with synthetic event data."""

    def _make_event(self):
        return {
            "id": "abc123",
            "sport_key": "soccer_epl",
            "commence_time": "2025-02-20T15:00:00Z",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "_competition": "PL",
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 2.10},
                                {"name": "Draw", "price": 3.40},
                                {"name": "Chelsea", "price": 3.60},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 2.5, "price": 1.85},
                                {"name": "Under", "point": 2.5, "price": 2.00},
                            ],
                        },
                    ],
                },
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 2.20},
                                {"name": "Draw", "price": 3.30},
                                {"name": "Chelsea", "price": 3.50},
                            ],
                        },
                    ],
                },
            ],
        }

    def test_parse_h2h_odds(self):
        from footy.providers.the_odds_api import parse_odds_for_match
        event = self._make_event()
        parsed = parse_odds_for_match(event)

        assert parsed["competition"] == "PL"
        assert parsed["n_bookmakers"] == 2
        # Best odds = max across bookmakers
        assert parsed["best_h"] == 2.20
        assert parsed["best_d"] == 3.40
        assert parsed["best_a"] == 3.60
        # Average
        assert abs(parsed["avg_h"] - 2.15) < 0.01
        assert abs(parsed["avg_d"] - 3.35) < 0.01

    def test_parse_totals(self):
        from footy.providers.the_odds_api import parse_odds_for_match
        event = self._make_event()
        parsed = parse_odds_for_match(event)
        assert parsed["totals"]["over_25"] == 1.85
        assert parsed["totals"]["under_25"] == 2.00

    def test_parse_empty_bookmakers(self):
        from footy.providers.the_odds_api import parse_odds_for_match
        event = {"home_team": "X", "away_team": "Y", "bookmakers": []}
        parsed = parse_odds_for_match(event)
        assert parsed["best_h"] == 0
        assert parsed["n_bookmakers"] == 0

    def test_parse_odds_details(self):
        from footy.providers.the_odds_api import parse_odds_for_match
        event = self._make_event()
        parsed = parse_odds_for_match(event)
        assert len(parsed["odds_details"]) == 2
        assert parsed["odds_details"][0]["bookmaker"] == "Bet365"


# ── FPL API ───────────────────────────────────────────────────────────────────

class TestFPLImport:
    def test_import_module(self):
        from footy.providers import fpl
        assert hasattr(fpl, "fetch_bootstrap")
        assert hasattr(fpl, "get_team_injury_report")
        assert hasattr(fpl, "get_all_teams_availability")
        assert hasattr(fpl, "get_fixture_difficulty")

    def test_ensure_team_map_callable(self):
        from footy.providers.fpl import _ensure_team_map
        assert callable(_ensure_team_map)


class TestFPLParsing:
    def test_injury_report_structure(self):
        """Test that get_team_injury_report returns proper structure when data available."""
        from footy.providers.fpl import get_team_injury_report
        # Without real API, this will likely return empty but shouldn't crash
        with patch("footy.providers.fpl._get", return_value=None):
            result = get_team_injury_report("Arsenal")
            # Should handle None response gracefully
            assert isinstance(result, dict)

    @patch("footy.providers.fpl.fetch_fixtures")
    @patch("footy.providers.fpl.fetch_bootstrap")
    @patch("footy.providers.fpl._FPL_TEAM_MAP", {1: "Arsenal", 2: "Chelsea"})
    def test_fixture_difficulty_with_mock(self, mock_bootstrap, mock_fixtures):
        """Test fixture difficulty parsing with mocked API response."""
        from footy.providers.fpl import get_fixture_difficulty

        mock_bootstrap.return_value = {
            "events": [{"id": 20, "is_current": True}],
            "teams": [
                {"id": 1, "name": "Arsenal"},
                {"id": 2, "name": "Chelsea"},
            ],
        }
        mock_fixtures.return_value = [
            {
                "team_h": 1, "team_a": 2,
                "team_h_difficulty": 3, "team_a_difficulty": 4,
                "event": 21, "finished": False,
                "kickoff_time": "2025-02-20T15:00:00Z",
            },
        ]
        result = get_fixture_difficulty()
        assert isinstance(result, dict)


# ── TheSportsDB ──────────────────────────────────────────────────────────────

class TestTheSportsDBImport:
    def test_import_module(self):
        from footy.providers import thesportsdb
        assert hasattr(thesportsdb, "search_team")
        assert hasattr(thesportsdb, "get_venue_info")
        assert hasattr(thesportsdb, "get_last_events")
        assert hasattr(thesportsdb, "get_next_events")
        assert hasattr(thesportsdb, "enrich_matches_with_venue")

    def test_safe_int(self):
        from footy.providers.thesportsdb import _safe_int
        assert _safe_int("42") == 42
        assert _safe_int(None) is None
        assert _safe_int("") is None
        assert _safe_int("abc") is None

    def test_api_key_default(self):
        from footy.providers.thesportsdb import _api_key
        # Default key is "3" (free dev key)
        with patch.dict("os.environ", {"THESPORTSDB_KEY": ""}):
            assert _api_key() == "3"


class TestTheSportsDBParsing:
    @patch("footy.providers.thesportsdb._get")
    def test_search_team_parsing(self, mock_get):
        """Test search_team correctly parses API response."""
        from footy.providers.thesportsdb import search_team, _team_cache

        # Clear cache to force the mock path
        _team_cache.clear()

        mock_get.return_value = {
            "teams": [
                {
                    "idTeam": "133604",
                    "strTeam": "Arsenal",
                    "strTeamShort": "ARS",
                    "strStadium": "Emirates Stadium",
                    "intStadiumCapacity": "60704",
                    "strCountry": "England",
                    "intFormedYear": "1886",
                    "strLeague": "English Premier League",
                    "strBadge": "https://example.com/badge.png",
                    "strDescriptionEN": "Arsenal Football Club...",
                }
            ]
        }
        result = search_team("Arsenal")
        assert result is not None
        assert result["name"] == "Arsenal"
        assert result["stadium"] == "Emirates Stadium"
        assert result["capacity"] == 60704

    @patch("footy.providers.thesportsdb._get")
    def test_search_team_not_found(self, mock_get):
        from footy.providers.thesportsdb import search_team
        mock_get.return_value = {"teams": None}
        result = search_team("ZZZNonexistent")
        assert result is None

    @patch("footy.providers.thesportsdb._get")
    def test_get_venue_info(self, mock_get):
        from footy.providers.thesportsdb import get_venue_info

        mock_get.return_value = {
            "teams": [
                {
                    "strTeam": "Arsenal",
                    "strStadium": "Emirates Stadium",
                    "intStadiumCapacity": "60704",
                    "strStadiumLocation": "Holloway, London",
                }
            ]
        }
        result = get_venue_info("Arsenal")
        assert result is not None
        assert "stadium" in result
        assert result["stadium"] == "Emirates Stadium"
        assert result["capacity"] == 60704

    @patch("footy.providers.thesportsdb._get")
    def test_get_last_events(self, mock_get):
        from footy.providers.thesportsdb import get_last_events, _team_cache

        # Pre-populate team cache so search_team doesn't make its own _get call
        _team_cache["arsenal"] = {
            "team_id": "133604", "name": "Arsenal",
            "stadium": "Emirates Stadium", "capacity": 60704,
        }

        mock_get.return_value = {
            "results": [
                {
                    "idEvent": "1234",
                    "strEvent": "Arsenal vs Chelsea",
                    "strHomeTeam": "Arsenal",
                    "strAwayTeam": "Chelsea",
                    "intHomeScore": "2",
                    "intAwayScore": "1",
                    "dateEvent": "2025-02-15",
                    "strVenue": "Emirates Stadium",
                    "strLeague": "English Premier League",
                }
            ]
        }
        result = get_last_events("Arsenal", n=5)
        assert len(result) == 1
        assert result[0]["home_team"] == "Arsenal"
        assert result[0]["home_score"] == 2
