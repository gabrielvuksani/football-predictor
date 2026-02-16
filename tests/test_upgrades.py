"""Tests for v8 upgrade features."""
import pytest


class TestGoalPatternExpert:
    """Test the new GoalPatternExpert."""

    def test_expert_exists(self):
        from footy.models.council import ALL_EXPERTS
        names = [type(e).__name__ for e in ALL_EXPERTS]
        assert "GoalPatternExpert" in names

    def test_expert_count(self):
        from footy.models.council import ALL_EXPERTS
        assert len(ALL_EXPERTS) == 8


class TestLeagueTableExpert:
    def test_expert_exists(self):
        from footy.models.council import ALL_EXPERTS
        names = [type(e).__name__ for e in ALL_EXPERTS]
        assert "LeagueTableExpert" in names


class TestModelVersion:
    def test_model_version_v8(self):
        from footy.models.council import MODEL_VERSION
        assert MODEL_VERSION == "v8_council"


class TestTeamMapping:
    def test_negative_cache(self):
        from footy.team_mapping import get_canonical_id, _NEGATIVE_CACHE
        # A clearly nonsensical name should get cached
        _NEGATIVE_CACHE.clear()
        result, exact = get_canonical_id("ZZZ Nonexistent Team 9999")
        assert result is None
        # Should be in negative cache now
        from footy.team_mapping import _normalize_for_lookup
        assert _normalize_for_lookup("ZZZ Nonexistent Team 9999") in _NEGATIVE_CACHE

    def test_provider_hints(self):
        from footy.team_mapping import get_canonical_id
        # Provider-specific lookup should work
        cid, exact = get_canonical_id("Man City", provider="football-data.co.uk")
        assert cid == "manchester-city"

    def test_canonical_name(self):
        from footy.team_mapping import get_canonical_name
        assert get_canonical_name("FC Bayern München") == "Bayern München"
        assert get_canonical_name("PSG") == "Paris Saint-Germain"


class TestXg:
    def test_compute_xg_from_stats(self):
        from footy.xg import compute_xg_from_stats
        xg = compute_xg_from_stats(5, 12)
        assert xg > 0
        assert compute_xg_from_stats(0, 0) == 0.0

    def test_learn_conversion_rates_structure(self):
        """Test that learn_conversion_rates returns proper structure."""
        # Can't test with real DB but test import works
        from footy.xg import learn_conversion_rates
        assert callable(learn_conversion_rates)


class TestOptaProvider:
    def test_import(self):
        from footy.providers.opta_analyst import fetch_opta_predictions
        assert callable(fetch_opta_predictions)

    def test_ensure_table(self):
        import duckdb
        from footy.providers.opta_analyst import ensure_opta_table
        con = duckdb.connect(":memory:")
        ensure_opta_table(con)
        # Should not raise
        con.execute("SELECT * FROM opta_predictions")
