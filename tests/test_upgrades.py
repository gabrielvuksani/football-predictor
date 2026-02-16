"""Tests for v8/v9/v10 upgrade features."""
import pytest


class TestGoalPatternExpert:
    """Test the GoalPatternExpert."""

    def test_expert_exists(self):
        from footy.models.council import ALL_EXPERTS
        names = [type(e).__name__ for e in ALL_EXPERTS]
        assert "GoalPatternExpert" in names

    def test_expert_count(self):
        from footy.models.council import ALL_EXPERTS
        assert len(ALL_EXPERTS) == 11  # 11 experts: +BayesianRate+Injury in v10


class TestLeagueTableExpert:
    def test_expert_exists(self):
        from footy.models.council import ALL_EXPERTS
        names = [type(e).__name__ for e in ALL_EXPERTS]
        assert "LeagueTableExpert" in names


class TestMomentumExpert:
    def test_expert_exists(self):
        from footy.models.council import ALL_EXPERTS
        names = [type(e).__name__ for e in ALL_EXPERTS]
        assert "MomentumExpert" in names

    def test_expert_attributes(self):
        from footy.models.council import MomentumExpert
        expert = MomentumExpert()
        assert expert.name == "momentum"
        assert hasattr(expert, "compute")
        assert expert.FAST_N == 3
        assert expert.SLOW_N == 8
        assert expert.SLOPE_N == 10
        assert expert.LONG_TERM_N == 30

    def test_compute_returns_expert_result(self, finished_df):
        """MomentumExpert.compute() should return ExpertResult with proper shape."""
        from footy.models.council import MomentumExpert, ExpertResult
        expert = MomentumExpert()
        result = expert.compute(finished_df)
        assert isinstance(result, ExpertResult)
        n = len(finished_df)
        assert result.probs.shape == (n, 3)
        assert result.confidence.shape == (n,)
        assert isinstance(result.features, dict)

    def test_feature_names(self, finished_df):
        """MomentumExpert should produce all expected feature keys."""
        from footy.models.council import MomentumExpert
        expert = MomentumExpert()
        result = expert.compute(finished_df)
        expected_keys = {
            "mom_ema_cross_h", "mom_ema_cross_a", "mom_ema_cross_diff",
            "mom_gf_slope_h", "mom_gf_slope_a",
            "mom_ga_slope_h", "mom_ga_slope_a",
            "mom_pts_slope_h", "mom_pts_slope_a", "mom_pts_slope_diff",
            "mom_vol_h", "mom_vol_a", "mom_vol_diff",
            "mom_regress_h", "mom_regress_a",
            "mom_burst_h", "mom_burst_a",
            "mom_drought_h", "mom_drought_a",
            "mom_def_tighten_h", "mom_def_tighten_a",
        }
        assert expected_keys.issubset(set(result.features.keys()))

    def test_probs_sum_to_one(self, finished_df):
        """Each row of MomentumExpert probs should sum to ~1."""
        import numpy as np
        from footy.models.council import MomentumExpert
        expert = MomentumExpert()
        result = expert.compute(finished_df)
        sums = result.probs.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_confidence_bounds(self, finished_df):
        """Confidence should be in [0, 1]."""
        from footy.models.council import MomentumExpert
        expert = MomentumExpert()
        result = expert.compute(finished_df)
        assert (result.confidence >= 0).all()
        assert (result.confidence <= 1).all()

    def test_momentum_after_matches(self, finished_df):
        """After enough matches, EMA crossover should be non-zero for some teams."""
        import numpy as np
        from footy.models.council import MomentumExpert
        expert = MomentumExpert()
        result = expert.compute(finished_df)
        # At least some non-zero values after the warm-up period
        assert np.any(result.features["mom_ema_cross_h"] != 0)

    def test_drought_detection(self, finished_df):
        """Drought values should be non-negative integers."""
        import numpy as np
        from footy.models.council import MomentumExpert
        expert = MomentumExpert()
        result = expert.compute(finished_df)
        assert (result.features["mom_drought_h"] >= 0).all()
        assert (result.features["mom_drought_a"] >= 0).all()


class TestMultiModelStack:
    """Test the multi-model stacking architecture."""

    def test_all_experts_have_compute(self):
        from footy.models.council import ALL_EXPERTS
        for expert in ALL_EXPERTS:
            assert hasattr(expert, "compute"), f"{type(expert).__name__} missing compute()"

    def test_all_experts_have_name(self):
        from footy.models.council import ALL_EXPERTS
        for expert in ALL_EXPERTS:
            assert hasattr(expert, "name"), f"{type(expert).__name__} missing name"
            assert isinstance(expert.name, str)
            assert len(expert.name) > 0

    def test_expert_names_unique(self):
        from footy.models.council import ALL_EXPERTS
        names = [e.name for e in ALL_EXPERTS]
        assert len(names) == len(set(names)), f"Duplicate expert names: {names}"


class TestSchedulerJobTypes:
    """Test that scheduler supports retrain and full_refresh job types."""

    def test_retrain_job_type_valid(self):
        from footy.scheduler import TrainingScheduler
        sched = TrainingScheduler.__new__(TrainingScheduler)
        assert hasattr(sched, "_job_retrain")

    def test_full_refresh_job_type_valid(self):
        from footy.scheduler import TrainingScheduler
        sched = TrainingScheduler.__new__(TrainingScheduler)
        assert hasattr(sched, "_job_full_refresh")


class TestModelVersion:
    def test_model_version_v10(self):
        from footy.models.council import MODEL_VERSION
        assert MODEL_VERSION == "v10_council"


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
