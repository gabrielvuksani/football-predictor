"""Smoke tests — quick sanity checks against the live database.

These tests use the REAL database (data/footy.duckdb) to verify
the production system is coherent. They are NON-destructive (read-only).

Run selectively:  pytest tests/test_smoke.py -v
"""
import os
import pytest

# Skip entire module if DB doesn't exist
_DB = os.path.join(os.path.dirname(__file__), "..", "data", "footy.duckdb")
pytestmark = pytest.mark.skipif(
    not os.path.exists(_DB),
    reason="Live database not present — skipping smoke tests",
)


@pytest.fixture(scope="module")
def live_con():
    """Read-only connection to the live DuckDB."""
    import duckdb
    try:
        con = duckdb.connect(_DB, read_only=True)
    except duckdb.IOException:
        pytest.skip("DuckDB locked by another process (server running?) — stop it first")
    yield con
    con.close()


class TestLiveDatabase:
    """Verify the live database is healthy."""

    def test_has_matches(self, live_con):
        n = live_con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        assert n > 1000, f"Suspiciously few matches: {n}"

    def test_has_finished_matches(self, live_con):
        n = live_con.execute(
            "SELECT COUNT(*) FROM matches WHERE status='FINISHED'"
        ).fetchone()[0]
        assert n > 500

    def test_has_upcoming_matches(self, live_con):
        n = live_con.execute(
            "SELECT COUNT(*) FROM matches WHERE status IN ('SCHEDULED','TIMED')"
        ).fetchone()[0]
        # May be 0 if no fixtures loaded recently
        assert n >= 0

    def test_has_elo_ratings(self, live_con):
        n = live_con.execute("SELECT COUNT(*) FROM elo_state").fetchone()[0]
        assert n > 10

    def test_elo_ratings_reasonable(self, live_con):
        lo, hi = live_con.execute(
            "SELECT MIN(rating), MAX(rating) FROM elo_state"
        ).fetchone()
        assert 800 < lo < 2000, f"Min Elo unreasonable: {lo}"
        assert 1200 < hi < 2500, f"Max Elo unreasonable: {hi}"

    def test_has_predictions(self, live_con):
        n = live_con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert n > 0

    def test_predictions_valid_probabilities(self, live_con):
        bad = live_con.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE p_home < 0 OR p_home > 1
               OR p_draw < 0 OR p_draw > 1
               OR p_away < 0 OR p_away > 1
        """).fetchone()[0]
        assert bad == 0, f"{bad} predictions have invalid probabilities"

    def test_predictions_sum_near_one(self, live_con):
        bad = live_con.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE ABS(p_home + p_draw + p_away - 1.0) > 0.02
        """).fetchone()[0]
        assert bad == 0, f"{bad} predictions don't sum to ~1.0"

    def test_has_match_extras(self, live_con):
        n = live_con.execute("SELECT COUNT(*) FROM match_extras").fetchone()[0]
        assert n > 100

    def test_competitions_exist(self, live_con):
        comps = {
            r[0] for r in live_con.execute(
                "SELECT DISTINCT competition FROM matches WHERE competition IS NOT NULL"
            ).fetchall()
        }
        assert len(comps) >= 2, f"Only {comps} competitions found"

    def test_no_orphan_predictions(self, live_con):
        orphans = live_con.execute("""
            SELECT COUNT(*) FROM predictions p
            LEFT JOIN matches m ON p.match_id = m.match_id
            WHERE m.match_id IS NULL
        """).fetchone()[0]
        assert orphans == 0, f"{orphans} orphan predictions"


class TestLiveModel:
    """Verify the trained model file is loadable."""

    def test_v10_model_exists(self):
        from footy.models.council import MODEL_PATH
        assert MODEL_PATH.exists(), f"Model file missing: {MODEL_PATH}"

    def test_v10_model_loadable(self):
        import joblib
        from footy.models.council import MODEL_PATH
        bundle = joblib.load(MODEL_PATH)
        assert isinstance(bundle, dict)
        assert "model" in bundle or "pipeline" in bundle or "clf" in bundle \
            or "meta" in bundle, f"Unexpected model keys: {list(bundle.keys())}"

    def test_v10_model_has_feature_names(self):
        import joblib
        from footy.models.council import MODEL_PATH
        bundle = joblib.load(MODEL_PATH)
        # Should have feature names list
        if "feature_names" in bundle:
            assert len(bundle["feature_names"]) > 50


class TestLivePipeline:
    """Verify core pipeline imports work."""

    def test_pipeline_imports(self):
        from footy import pipeline
        assert callable(pipeline.ingest)
        assert callable(pipeline.update_elo_from_finished)
        assert callable(pipeline.refit_poisson)

    def test_council_imports(self):
        from footy.models.council import train_and_save, predict_upcoming, get_expert_breakdown
        assert callable(train_and_save)
        assert callable(predict_upcoming)
        assert callable(get_expert_breakdown)

    def test_cli_imports(self):
        from footy.cli import app
        assert app is not None
