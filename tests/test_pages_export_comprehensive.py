"""Comprehensive tests for GitHub Pages export functionality."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from footy.cli.pages_cmds import (
    _dump,
    _export_match_detail,
    _build_value_bets,
    _build_league_table,
    _build_stats,
    _build_performance,
)


class TestDumpFunction:
    """Tests for _dump helper function."""

    def test_dump_creates_parent_directories(self):
        """Test that _dump creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "level1" / "level2" / "file.json"
            data = {"key": "value"}

            _dump(path, data)

            assert path.exists()
            assert path.parent.exists()

    def test_dump_writes_valid_json(self):
        """Test that _dump writes valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"home": "Arsenal", "away": "Chelsea", "score": [2, 1]}

            _dump(path, data)

            content = json.loads(path.read_text())
            assert content["home"] == "Arsenal"
            assert content["score"] == [2, 1]

    def test_dump_handles_dict_and_list(self):
        """Test that _dump handles both dict and list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dict_path = Path(tmpdir) / "dict.json"
            list_path = Path(tmpdir) / "list.json"

            _dump(dict_path, {"key": "value"})
            _dump(list_path, [1, 2, 3])

            dict_content = json.loads(dict_path.read_text())
            list_content = json.loads(list_path.read_text())

            assert isinstance(dict_content, dict)
            assert isinstance(list_content, list)

    def test_dump_serializes_datetime(self):
        """Test that _dump handles datetime objects."""
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "datetime.json"
            data = {"timestamp": datetime(2025, 1, 1, 12, 0, 0)}

            _dump(path, data)

            content = json.loads(path.read_text())
            assert "timestamp" in content

    def test_dump_overwrites_existing_file(self):
        """Test that _dump overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"

            _dump(path, {"version": 1})
            _dump(path, {"version": 2})

            content = json.loads(path.read_text())
            assert content["version"] == 2


class TestBuildValueBets:
    """Tests for value bet extraction."""

    def test_build_value_bets_empty_list(self):
        """Test value bets with empty match list."""
        bets = _build_value_bets([])
        assert isinstance(bets, list)
        assert len(bets) == 0

    def test_build_value_bets_no_odds(self):
        """Test that matches without odds are skipped."""
        matches = [
            {
                "match_id": 1,
                "home_team": "A",
                "away_team": "B",
                "p_home": 0.6,
                "p_draw": 0.2,
                "p_away": 0.2,
                # No odds
            }
        ]

        bets = _build_value_bets(matches)
        assert len(bets) == 0

    def test_build_value_bets_finds_positive_edge(self):
        """Test that positive edge bets are identified."""
        matches = [
            {
                "match_id": 1,
                "home_team": "A",
                "away_team": "B",
                "competition": "PL",
                "utc_date": "2025-03-01",
                "p_home": 0.7,
                "p_draw": 0.15,
                "p_away": 0.15,
                "b365h": 1.5,  # Implied prob ~0.67 vs model 0.7 = positive edge
                "b365d": 6.0,
                "b365a": 6.0,
                "confidence": 0.7,
                "model_agreement": 0.8,
            }
        ]

        bets = _build_value_bets(matches, min_edge=0.01)

        assert len(bets) > 0
        assert bets[0]["outcome"] == "Home"

    def test_build_value_bets_requires_min_edge(self):
        """Test that min_edge parameter affects results."""
        matches = [
            {
                "match_id": 1,
                "home_team": "A",
                "away_team": "B",
                "competition": "PL",
                "utc_date": "2025-03-01",
                "p_home": 0.55,
                "p_draw": 0.25,
                "p_away": 0.20,
                "b365h": 1.9,  # Implied ~0.526, edge ~0.024
                "b365d": 4.0,
                "b365a": 4.0,
                "confidence": 0.5,
                "model_agreement": 0.5,
            }
        ]

        bets_strict = _build_value_bets(matches, min_edge=0.10)
        bets_loose = _build_value_bets(matches, min_edge=0.001)

        # Stricter edge requirement should result in fewer or equal bets
        assert len(bets_strict) <= len(bets_loose)

    def test_build_value_bets_structure(self):
        """Test that value bet structure is correct."""
        matches = [
            {
                "match_id": 123,
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "competition": "PL",
                "utc_date": "2025-03-15",
                "p_home": 0.65,
                "p_draw": 0.2,
                "p_away": 0.15,
                "b365h": 1.6,
                "b365d": 4.0,
                "b365a": 6.0,
                "confidence": 0.75,
                "model_agreement": 0.8,
            }
        ]

        bets = _build_value_bets(matches, min_edge=0.01)

        if len(bets) > 0:
            bet = bets[0]
            assert "match_id" in bet
            assert "home_team" in bet
            assert "away_team" in bet
            assert "outcome" in bet
            assert "model_prob" in bet
            assert "odds" in bet
            assert "edge" in bet
            assert "kelly_pct" in bet
            assert "risk_rating" in bet

    def test_build_value_bets_kelly_sizing(self):
        """Test that Kelly sizing is included."""
        matches = [
            {
                "match_id": 1,
                "home_team": "A",
                "away_team": "B",
                "competition": "PL",
                "utc_date": "2025-03-01",
                "p_home": 0.7,
                "p_draw": 0.15,
                "p_away": 0.15,
                "b365h": 1.5,
                "b365d": 6.0,
                "b365a": 6.0,
                "confidence": 0.7,
                "model_agreement": 0.8,
            }
        ]

        bets = _build_value_bets(matches, use_kelly=True, min_edge=0.01)

        if len(bets) > 0:
            assert bets[0]["kelly_pct"] >= 0
            assert bets[0]["kelly_fraction"] >= 0

    def test_build_value_bets_sorted_by_edge(self):
        """Test that bets are sorted by edge descending."""
        matches = [
            {
                "match_id": 1,
                "home_team": "A",
                "away_team": "B",
                "competition": "PL",
                "utc_date": "2025-03-01",
                "p_home": 0.8,
                "p_draw": 0.1,
                "p_away": 0.1,
                "b365h": 1.3,  # Large edge
                "b365d": 10.0,
                "b365a": 10.0,
                "confidence": 0.8,
                "model_agreement": 0.9,
            },
            {
                "match_id": 2,
                "home_team": "C",
                "away_team": "D",
                "competition": "PL",
                "utc_date": "2025-03-02",
                "p_home": 0.55,
                "p_draw": 0.25,
                "p_away": 0.2,
                "b365h": 1.8,  # Smaller edge
                "b365d": 4.0,
                "b365a": 4.0,
                "confidence": 0.55,
                "model_agreement": 0.6,
            }
        ]

        bets = _build_value_bets(matches, min_edge=0.01)

        if len(bets) >= 2:
            assert bets[0]["edge"] >= bets[1]["edge"]


class TestBuildLeagueTable:
    """Tests for league table building."""

    def test_build_league_table_empty_matches(self, seeded_con):
        """Test league table with no matches."""
        seeded_con.execute("DELETE FROM matches WHERE status='FINISHED'")

        standings = _build_league_table(seeded_con, "PL")
        assert isinstance(standings, list)
        assert len(standings) == 0

    def test_build_league_table_structure(self, seeded_con):
        """Test that league table has correct structure."""
        standings = _build_league_table(seeded_con, "PL")

        if len(standings) > 0:
            team = standings[0]
            assert "team" in team
            assert "p" in team  # Played
            assert "w" in team  # Won
            assert "d" in team  # Drawn
            assert "l" in team  # Lost
            assert "gf" in team  # Goals for
            assert "ga" in team  # Goals against
            assert "pts" in team  # Points
            assert "pos" in team  # Position
            assert "gd" in team  # Goal difference

    def test_build_league_table_points_calculation(self, seeded_con):
        """Test that points are calculated correctly."""
        standings = _build_league_table(seeded_con, "PL")

        for team in standings:
            expected_pts = team["w"] * 3 + team["d"] * 1
            assert team["pts"] == expected_pts

    def test_build_league_table_sorted_by_points(self, seeded_con):
        """Test that standings are sorted by points."""
        standings = _build_league_table(seeded_con, "PL")

        if len(standings) > 1:
            for i in range(len(standings) - 1):
                assert standings[i]["pts"] >= standings[i + 1]["pts"]

    def test_build_league_table_positions_sequential(self, seeded_con):
        """Test that positions are sequential."""
        standings = _build_league_table(seeded_con, "PL")

        for i, team in enumerate(standings):
            assert team["pos"] == i + 1


class TestBuildStats:
    """Tests for statistics building."""

    def test_build_stats_structure(self, seeded_con):
        """Test that stats have correct structure."""
        stats = _build_stats(seeded_con)

        assert isinstance(stats, dict)
        assert "total_matches" in stats
        assert "finished" in stats
        assert "upcoming" in stats
        assert "predictions" in stats
        assert "leagues" in stats
        assert "teams" in stats

    def test_build_stats_values_sensible(self, seeded_con):
        """Test that stats values are sensible."""
        stats = _build_stats(seeded_con)

        assert stats["total_matches"] >= 0
        assert stats["finished"] >= 0
        assert stats["upcoming"] >= 0
        assert stats["leagues"] >= 0
        assert stats["teams"] >= 0
        assert stats["finished"] <= stats["total_matches"]

    def test_build_stats_counts_matches(self, seeded_con):
        """Test that stats count matches correctly."""
        stats = _build_stats(seeded_con)

        # Should have many matches from seeded_con
        assert stats["total_matches"] > 50


class TestBuildPerformance:
    """Tests for performance metrics building."""

    def test_build_performance_structure(self, seeded_con):
        """Test that performance has correct structure."""
        perf = _build_performance(seeded_con)

        assert isinstance(perf, dict)
        assert "model" in perf
        assert "metrics" in perf
        assert "recent" in perf
        assert "by_competition" in perf
        assert "calibration" in perf

    def test_build_performance_recent_predictions(self, seeded_con):
        """Test that recent predictions are included."""
        perf = _build_performance(seeded_con)

        assert isinstance(perf["recent"], list)
        # May be empty if no scored predictions

    def test_build_performance_by_competition(self, seeded_con):
        """Test competition breakdown."""
        perf = _build_performance(seeded_con)

        assert isinstance(perf["by_competition"], list)
        for comp in perf["by_competition"]:
            assert "competition" in comp
            assert "n" in comp
            assert "accuracy" in comp

    def test_build_performance_calibration_bins(self, seeded_con):
        """Test that calibration has 10 bins."""
        perf = _build_performance(seeded_con)

        assert len(perf["calibration"]) == 10

    def test_build_performance_calibration_structure(self, seeded_con):
        """Test calibration bin structure."""
        perf = _build_performance(seeded_con)

        for bin_data in perf["calibration"]:
            assert "bucket" in bin_data
            assert "avg_predicted" in bin_data
            assert "avg_actual" in bin_data
            assert "count" in bin_data


class TestExportMatchDetail:
    """Tests for match detail export."""

    def test_export_match_detail_creates_file(self, seeded_con):
        """Test that match detail creates JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api_dir = Path(tmpdir) / "api"
            api_dir.mkdir()

            # Get a match ID from seeded database
            row = seeded_con.execute(
                "SELECT match_id FROM matches LIMIT 1"
            ).fetchone()

            if row:
                _export_match_detail(seeded_con, api_dir, row[0])

                detail_file = api_dir / "matches" / f"{row[0]}.json"
                assert detail_file.exists()

    def test_export_match_detail_structure(self, seeded_con):
        """Test match detail JSON structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api_dir = Path(tmpdir) / "api"
            api_dir.mkdir()

            row = seeded_con.execute(
                "SELECT match_id FROM matches LIMIT 1"
            ).fetchone()

            if row:
                _export_match_detail(seeded_con, api_dir, row[0])

                detail_file = api_dir / "matches" / f"{row[0]}.json"
                detail = json.loads(detail_file.read_text())

                assert "match_id" in detail
                assert "home_team" in detail
                assert "away_team" in detail
                assert "utc_date" in detail
                assert "competition" in detail
                assert "status" in detail

    def test_export_match_detail_with_predictions(self, seeded_con):
        """Test match detail includes predictions when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api_dir = Path(tmpdir) / "api"
            api_dir.mkdir()

            row = seeded_con.execute(
                "SELECT match_id FROM matches WHERE status='SCHEDULED' LIMIT 1"
            ).fetchone()

            if row:
                _export_match_detail(seeded_con, api_dir, row[0])

                detail_file = api_dir / "matches" / f"{row[0]}.json"
                detail = json.loads(detail_file.read_text())

                assert "prediction" in detail

    def test_export_match_detail_elo_section(self, seeded_con):
        """Test that match detail includes ELO ratings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api_dir = Path(tmpdir) / "api"
            api_dir.mkdir()

            row = seeded_con.execute(
                "SELECT match_id FROM matches LIMIT 1"
            ).fetchone()

            if row:
                _export_match_detail(seeded_con, api_dir, row[0])

                detail_file = api_dir / "matches" / f"{row[0]}.json"
                detail = json.loads(detail_file.read_text())

                assert "elo" in detail


class TestJSONValidity:
    """Tests for JSON output validity."""

    def test_value_bets_json_valid(self):
        """Test that value bets output is valid JSON."""
        matches = [
            {
                "match_id": 1,
                "home_team": "A",
                "away_team": "B",
                "competition": "PL",
                "utc_date": "2025-03-01",
                "p_home": 0.65,
                "p_draw": 0.2,
                "p_away": 0.15,
                "b365h": 1.6,
                "b365d": 4.0,
                "b365a": 6.0,
                "confidence": 0.7,
                "model_agreement": 0.75,
            }
        ]

        bets = _build_value_bets(matches, min_edge=0.01)
        json_str = json.dumps(bets)

        # Should not raise
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)

    def test_league_table_json_valid(self, seeded_con):
        """Test that league table output is valid JSON."""
        standings = _build_league_table(seeded_con, "PL")
        json_str = json.dumps(standings)

        parsed = json.loads(json_str)
        assert isinstance(parsed, list)

    def test_stats_json_valid(self, seeded_con):
        """Test that stats output is valid JSON."""
        stats = _build_stats(seeded_con)
        json_str = json.dumps(stats)

        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_performance_json_valid(self, seeded_con):
        """Test that performance output is valid JSON."""
        perf = _build_performance(seeded_con)
        json_str = json.dumps(perf, default=str)

        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


class TestErrorHandling:
    """Tests for error handling in export functions."""

    def test_build_stats_handles_empty_database(self):
        """Test stats building with empty database."""
        import duckdb
        from footy.db import SCHEMA_SQL

        con = duckdb.connect(":memory:")
        con.execute(SCHEMA_SQL)

        stats = _build_stats(con)

        assert stats["total_matches"] == 0
        con.close()

    def test_build_performance_handles_no_predictions(self):
        """Test performance building with no predictions."""
        import duckdb
        from footy.db import SCHEMA_SQL

        con = duckdb.connect(":memory:")
        con.execute(SCHEMA_SQL)

        perf = _build_performance(con)

        assert perf["metrics"] is None
        assert perf["recent"] == []
        con.close()

    def test_export_match_detail_handles_missing_match(self, seeded_con):
        """Test that missing match is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api_dir = Path(tmpdir) / "api"
            api_dir.mkdir()

            # Try to export non-existent match
            _export_match_detail(seeded_con, api_dir, 999999)

            # Should not create file
            detail_file = api_dir / "matches" / "999999.json"
            assert not detail_file.exists()


class TestDirectoryCreation:
    """Tests for directory structure creation."""

    def test_export_creates_nested_directories(self, seeded_con):
        """Test that export creates nested directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api_dir = Path(tmpdir) / "api"
            api_dir.mkdir()

            row = seeded_con.execute(
                "SELECT match_id FROM matches LIMIT 1"
            ).fetchone()

            if row:
                _export_match_detail(seeded_con, api_dir, row[0])

                # Should create matches/ID/ directory
                detail_dir = api_dir / "matches" / str(row[0])
                assert detail_dir.parent.exists()

    def test_dump_creates_league_table_directory(self):
        """Test that league table dump creates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "league-table" / "PL.json"
            data = {"standings": []}

            _dump(path, data)

            assert (Path(tmpdir) / "league-table").exists()

    def test_dump_creates_form_table_directory(self):
        """Test that form table dump creates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "form-table" / "SA.json"
            data = {"table": []}

            _dump(path, data)

            assert (Path(tmpdir) / "form-table").exists()
