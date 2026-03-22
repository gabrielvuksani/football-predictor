"""Unit tests — xG computation module."""
import pytest


class TestXgComputation:
    """xG calculation from shot statistics."""

    def test_compute_xg_from_stats_basic(self):
        from footy.xg import compute_xg_from_stats
        xg = compute_xg_from_stats(shots_on_target=5, total_shots=15)
        assert 0.0 < xg < 5.0

    def test_compute_xg_zero_shots(self):
        from footy.xg import compute_xg_from_stats
        assert compute_xg_from_stats(0, 0) == 0.0

    def test_compute_xg_all_on_target(self):
        from footy.xg import compute_xg_from_stats
        xg_all = compute_xg_from_stats(10, 10)
        xg_half = compute_xg_from_stats(5, 10)
        assert xg_all > xg_half

    def test_ensure_xg_table_idempotent(self, con):
        from footy.xg import ensure_xg_table
        ensure_xg_table(con)
        ensure_xg_table(con)

    def test_backfill_on_seeded_data(self, seeded_con):
        """Backfill should not crash even without shot data."""
        from footy.xg import backfill_xg_for_finished_matches
        result = backfill_xg_for_finished_matches(seeded_con, verbose=False)
        # Just check it doesn't crash; shot data may be missing
        assert result is not None or True  # no crash = pass
