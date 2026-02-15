"""Unit tests — H2H statistics module."""
import pytest


class TestH2hTables:
    """Table creation and basic operations."""

    def test_ensure_h2h_tables_idempotent(self, con):
        from footy.h2h import ensure_h2h_tables
        ensure_h2h_tables(con)
        ensure_h2h_tables(con)  # should not fail

    def test_recompute_on_seeded_data(self, seeded_con):
        from footy.h2h import recompute_h2h_stats
        result = recompute_h2h_stats(seeded_con, verbose=False)
        # Should process some pairs
        n_pairs = seeded_con.execute("SELECT COUNT(*) FROM h2h_stats").fetchone()[0]
        assert n_pairs > 0

    def test_get_h2h_any_venue_found(self, seeded_con):
        from footy.h2h import recompute_h2h_stats, get_h2h_any_venue
        recompute_h2h_stats(seeded_con, verbose=False)
        result = get_h2h_any_venue(seeded_con, "Arsenal", "Chelsea", limit=5)
        assert isinstance(result, dict)

    def test_get_h2h_any_venue_unknown_teams(self, seeded_con):
        from footy.h2h import get_h2h_any_venue
        result = get_h2h_any_venue(seeded_con, "NoTeam1", "NoTeam2")
        assert isinstance(result, dict)


class TestH2hVenue:
    def test_venue_stats_populated(self, seeded_con):
        from footy.h2h import recompute_h2h_stats
        recompute_h2h_stats(seeded_con, verbose=False)
        n = seeded_con.execute("SELECT COUNT(*) FROM h2h_venue_stats").fetchone()[0]
        assert n > 0
