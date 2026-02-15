"""Unit tests — Database schema & connection."""
import duckdb
import pytest


class TestSchema:
    """DuckDB schema creation and table structure."""

    EXPECTED_TABLES = [
        "matches", "predictions", "prediction_scores", "metrics",
        "elo_state", "elo_applied", "poisson_state", "match_extras",
        "news", "llm_insights", "h2h_stats", "h2h_venue_stats",
        "match_xg", "team_mappings", "team_name_lookups",
    ]

    def test_schema_creates_all_tables(self, con):
        tables = [
            r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        ]
        for t in self.EXPECTED_TABLES:
            assert t in tables, f"Missing table: {t}"

    def test_schema_idempotent(self, con):
        """Applying schema twice must not fail."""
        from footy.db import SCHEMA_SQL
        con.execute(SCHEMA_SQL)  # second application
        count = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='main'"
        ).fetchone()[0]
        assert count >= len(self.EXPECTED_TABLES)

    def test_matches_pk(self, con):
        con.execute(
            "INSERT INTO matches(match_id, provider, home_team, away_team) VALUES (1,'t','A','B')"
        )
        with pytest.raises(duckdb.ConstraintException):
            con.execute(
                "INSERT INTO matches(match_id, provider, home_team, away_team) VALUES (1,'t','C','D')"
            )

    def test_predictions_composite_pk(self, con):
        con.execute(
            "INSERT INTO predictions(match_id, model_version, p_home, p_draw, p_away) "
            "VALUES (1, 'v7_council', 0.5, 0.3, 0.2)"
        )
        # Same match, different model → OK
        con.execute(
            "INSERT INTO predictions(match_id, model_version, p_home, p_draw, p_away) "
            "VALUES (1, 'other', 0.4, 0.3, 0.3)"
        )
        assert con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2


class TestSeededData:
    """Seeded test data integrity checks."""

    def test_match_count(self, seeded_con):
        total = seeded_con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        assert total > 80, f"Expected 80+ matches, got {total}"

    def test_has_all_statuses(self, seeded_con):
        statuses = {
            r[0] for r in seeded_con.execute(
                "SELECT DISTINCT status FROM matches"
            ).fetchall()
        }
        assert "FINISHED" in statuses
        assert "SCHEDULED" in statuses

    def test_has_elo_ratings(self, seeded_con):
        n = seeded_con.execute("SELECT COUNT(*) FROM elo_state").fetchone()[0]
        assert n >= 10

    def test_has_predictions(self, seeded_con):
        n = seeded_con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert n > 0

    def test_has_odds(self, seeded_con):
        n = seeded_con.execute("SELECT COUNT(*) FROM match_extras WHERE b365h IS NOT NULL").fetchone()[0]
        assert n > 0
