"""Tests for v10 Expert Council upgrades.

Tests new experts (BayesianRate, Injury), upgraded features (DC, MC, Skellam),
and the enhanced meta-learner.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """Create a small DataFrame suitable for expert testing."""
    rows = []
    teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
    np.random.seed(42)
    for i in range(40):
        h = teams[i % 4]
        a = teams[(i + 1) % 4]
        hg = np.random.randint(0, 4)
        ag = np.random.randint(0, 3)
        rows.append({
            "utc_date": pd.Timestamp(f"2024-01-{(i % 28) + 1}", tz="UTC"),
            "competition": "PL",
            "home_team": h,
            "away_team": a,
            "home_goals": hg,
            "away_goals": ag,
            "b365h": round(np.random.uniform(1.5, 4.0), 2),
            "b365d": round(np.random.uniform(2.5, 4.5), 2),
            "b365a": round(np.random.uniform(1.5, 6.0), 2),
            "raw_json": None,
            "b365ch": None, "b365cd": None, "b365ca": None,
            "psh": round(np.random.uniform(1.5, 4.0), 2),
            "psd": round(np.random.uniform(2.5, 4.5), 2),
            "psa": round(np.random.uniform(1.5, 6.0), 2),
            "avgh": round(np.random.uniform(1.5, 4.0), 2),
            "avgd": round(np.random.uniform(2.5, 4.5), 2),
            "avga": round(np.random.uniform(1.5, 6.0), 2),
            "maxh": round(np.random.uniform(1.5, 4.0), 2),
            "maxd": round(np.random.uniform(2.5, 4.5), 2),
            "maxa": round(np.random.uniform(1.5, 6.0), 2),
            "b365_o25": round(np.random.uniform(1.5, 2.5), 2),
            "b365_u25": round(np.random.uniform(1.5, 2.5), 2),
            "avg_o25": round(np.random.uniform(1.5, 2.5), 2),
            "avg_u25": round(np.random.uniform(1.5, 2.5), 2),
            "max_o25": round(np.random.uniform(1.5, 2.5), 2),
            "max_u25": round(np.random.uniform(1.5, 2.5), 2),
            "hs": np.random.randint(3, 20),
            "hst": np.random.randint(1, 10),
            "hc": np.random.randint(0, 12),
            "hy": np.random.randint(0, 4),
            "hr": 0,
            "as_": np.random.randint(3, 20),
            "ast": np.random.randint(1, 10),
            "ac": np.random.randint(0, 12),
            "ay": np.random.randint(0, 4),
            "ar": 0,
            "hthg": max(0, hg - 1),
            "htag": max(0, ag - 1),
        })
    return pd.DataFrame(rows)


class TestBayesianRateExpert:
    """Test the new BayesianRateExpert."""

    def test_expert_exists(self):
        from footy.models.council import BayesianRateExpert
        expert = BayesianRateExpert()
        assert expert.name == "bayesian_rate"

    def test_compute_returns_expert_result(self, sample_df):
        from footy.models.council import BayesianRateExpert, ExpertResult
        expert = BayesianRateExpert()
        result = expert.compute(sample_df)
        assert isinstance(result, ExpertResult)
        assert result.probs.shape == (len(sample_df), 3)
        assert result.confidence.shape == (len(sample_df),)

    def test_probabilities_valid(self, sample_df):
        from footy.models.council import BayesianRateExpert
        expert = BayesianRateExpert()
        result = expert.compute(sample_df)
        # All probabilities in [0, 1]
        assert np.all(result.probs >= 0)
        assert np.all(result.probs <= 1)
        # Row sums ≈ 1
        row_sums = result.probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=0.01)

    def test_features_present(self, sample_df):
        from footy.models.council import BayesianRateExpert
        expert = BayesianRateExpert()
        result = expert.compute(sample_df)
        expected_keys = [
            "bayes_wr_h", "bayes_wr_a", "bayes_cs_h", "bayes_cs_a",
            "bayes_btts_h", "bayes_btts_a", "bayes_o25_h", "bayes_o25_a",
            "bayes_sf_h", "bayes_sf_a", "bayes_gpm_h", "bayes_gpm_a",
            "bayes_home_wr", "bayes_away_wr",
            "bayes_wr_diff", "bayes_cs_diff", "bayes_gpm_diff", "bayes_gpm_sum",
        ]
        for key in expected_keys:
            assert key in result.features, f"Missing feature: {key}"

    def test_shrinkage_effect(self, sample_df):
        from footy.models.council import BayesianRateExpert
        expert = BayesianRateExpert()
        result = expert.compute(sample_df)
        # After enough data, win rates should be between 0 and 1
        wr_h = result.features["bayes_wr_h"]
        assert np.all((wr_h >= 0) & (wr_h <= 1))


class TestInjuryAvailabilityExpert:
    """Test the new InjuryAvailabilityExpert."""

    def test_expert_exists(self):
        from footy.models.council import InjuryAvailabilityExpert
        expert = InjuryAvailabilityExpert()
        assert expert.name == "injury"

    def test_compute_returns_expert_result(self, sample_df):
        from footy.models.council import InjuryAvailabilityExpert, ExpertResult
        expert = InjuryAvailabilityExpert()
        result = expert.compute(sample_df)
        assert isinstance(result, ExpertResult)
        assert result.probs.shape == (len(sample_df), 3)

    def test_features_present(self, sample_df):
        from footy.models.council import InjuryAvailabilityExpert
        expert = InjuryAvailabilityExpert()
        result = expert.compute(sample_df)
        expected_keys = ["inj_score_h", "inj_score_a", "inj_injured_h", "inj_injured_a",
                        "inj_doubtful_h", "inj_doubtful_a", "inj_fdr_h", "inj_fdr_a",
                        "inj_af_h", "inj_af_a", "inj_diff", "inj_squad_diff"]
        for key in expected_keys:
            assert key in result.features

    def test_low_confidence_without_data(self, sample_df):
        from footy.models.council import InjuryAvailabilityExpert
        expert = InjuryAvailabilityExpert()
        result = expert.compute(sample_df)
        # Without injury data, confidence should be 0
        assert np.all(result.confidence <= 0.3)


class TestPoissonV10Features:
    """Test the upgraded PoissonExpert with DC, Skellam, MC features."""

    def test_dc_features_present(self, sample_df):
        from footy.models.council import PoissonExpert
        expert = PoissonExpert()
        result = expert.compute(sample_df)
        dc_keys = ["pois_dc_ph", "pois_dc_pd", "pois_dc_pa",
                   "pois_dc_cs00", "pois_dc_btts"]
        for key in dc_keys:
            assert key in result.features, f"Missing DC feature: {key}"

    def test_skellam_features_present(self, sample_df):
        from footy.models.council import PoissonExpert
        expert = PoissonExpert()
        result = expert.compute(sample_df)
        sk_keys = ["pois_sk_mean_gd", "pois_sk_var_gd", "pois_sk_skew"]
        for key in sk_keys:
            assert key in result.features, f"Missing Skellam feature: {key}"

    def test_mc_features_present(self, sample_df):
        from footy.models.council import PoissonExpert
        expert = PoissonExpert()
        result = expert.compute(sample_df)
        mc_keys = ["pois_mc_ph", "pois_mc_pd", "pois_mc_pa",
                   "pois_mc_btts", "pois_mc_o25", "pois_mc_o35",
                   "pois_mc_var_total"]
        for key in mc_keys:
            assert key in result.features, f"Missing MC feature: {key}"

    def test_dc_probs_valid(self, sample_df):
        from footy.models.council import PoissonExpert
        expert = PoissonExpert()
        result = expert.compute(sample_df)
        dc_ph = result.features["pois_dc_ph"]
        dc_pd = result.features["pois_dc_pd"]
        dc_pa = result.features["pois_dc_pa"]
        # Should be valid probabilities
        for arr in [dc_ph, dc_pd, dc_pa]:
            assert np.all(arr >= 0), "DC probabilities should be non-negative"
            assert np.all(arr <= 1), "DC probabilities should be ≤ 1"

    def test_mc_probs_reasonable(self, sample_df):
        from footy.models.council import PoissonExpert
        expert = PoissonExpert()
        result = expert.compute(sample_df)
        # MC probs may be 0 for early rows (no data yet), but later should be valid
        mc_ph = result.features["pois_mc_ph"]
        mc_pd = result.features["pois_mc_pd"]
        mc_pa = result.features["pois_mc_pa"]
        # At least some rows should have non-zero MC probs
        assert np.any(mc_ph > 0) or np.any(mc_pd > 0) or np.any(mc_pa > 0)


class TestFormV10Features:
    """Test the upgraded FormExpert with Bayesian shrinkage and schedule difficulty."""

    def test_shrinkage_features_present(self, sample_df):
        from footy.models.council import FormExpert
        expert = FormExpert()
        result = expert.compute(sample_df)
        shrink_keys = ["form_shrunk_wr_h", "form_shrunk_wr_a",
                      "form_shrunk_cs_h", "form_shrunk_cs_a",
                      "form_shrunk_btts_h", "form_shrunk_btts_a"]
        for key in shrink_keys:
            assert key in result.features, f"Missing shrinkage feature: {key}"

    def test_schedule_difficulty_present(self, sample_df):
        from footy.models.council import FormExpert
        expert = FormExpert()
        result = expert.compute(sample_df)
        assert "form_sched_diff_h" in result.features
        assert "form_sched_diff_a" in result.features

    def test_conversion_present(self, sample_df):
        from footy.models.council import FormExpert
        expert = FormExpert()
        result = expert.compute(sample_df)
        assert "form_conversion_h" in result.features
        assert "form_conversion_a" in result.features


class TestEloV10Features:
    """Test the upgraded EloExpert with nonlinear transforms."""

    def test_tanh_feature_present(self, sample_df):
        from footy.models.council import EloExpert
        expert = EloExpert()
        result = expert.compute(sample_df)
        assert "elo_tanh_diff" in result.features

    def test_log_feature_present(self, sample_df):
        from footy.models.council import EloExpert
        expert = EloExpert()
        result = expert.compute(sample_df)
        assert "elo_log_diff" in result.features

    def test_weighted_momentum_present(self, sample_df):
        from footy.models.council import EloExpert
        expert = EloExpert()
        result = expert.compute(sample_df)
        assert "elo_weighted_mom_h" in result.features
        assert "elo_weighted_mom_a" in result.features


class TestMarketV10Features:
    """Test the upgraded MarketExpert with logit-space features."""

    def test_logit_features_present(self, sample_df):
        from footy.models.council import MarketExpert
        expert = MarketExpert()
        result = expert.compute(sample_df)
        logit_keys = ["mkt_logit_h", "mkt_logit_d", "mkt_logit_a"]
        for key in logit_keys:
            assert key in result.features

    def test_entropy_feature_present(self, sample_df):
        from footy.models.council import MarketExpert
        expert = MarketExpert()
        result = expert.compute(sample_df)
        assert "mkt_entropy" in result.features

    def test_pinnacle_features_present(self, sample_df):
        from footy.models.council import MarketExpert
        expert = MarketExpert()
        result = expert.compute(sample_df)
        pin_keys = ["mkt_pin_ph", "mkt_pin_pd", "mkt_pin_pa", "mkt_has_pin"]
        for key in pin_keys:
            assert key in result.features


class TestMetaLearnV10:
    """Test the upgraded _build_meta_X function."""

    def test_feature_count_increased(self, sample_df):
        from footy.models.council import _run_experts, _build_meta_X, ExpertResult
        results = _run_experts(sample_df)
        # Add a dummy DC expert result
        n = len(sample_df)
        dc_result = ExpertResult(
            probs=np.full((n, 3), 1/3),
            confidence=np.zeros(n),
            features={"dc_eg_h": np.zeros(n), "dc_eg_a": np.zeros(n),
                      "dc_o25": np.zeros(n), "dc_has": np.zeros(n)},
        )
        all_results = results + [dc_result]
        comps = sample_df["competition"].to_numpy()
        X = _build_meta_X(all_results, competitions=comps)
        # v10 should have 300+ features (was ~200 in v9)
        assert X.shape[1] > 250, f"Expected >250 features, got {X.shape[1]}"
        assert X.shape[0] == n

    def test_no_nan_in_features(self, sample_df):
        from footy.models.council import _run_experts, _build_meta_X, ExpertResult
        results = _run_experts(sample_df)
        n = len(sample_df)
        dc_result = ExpertResult(
            probs=np.full((n, 3), 1/3),
            confidence=np.zeros(n),
            features={"dc_eg_h": np.zeros(n), "dc_eg_a": np.zeros(n),
                      "dc_o25": np.zeros(n), "dc_has": np.zeros(n)},
        )
        all_results = results + [dc_result]
        comps = sample_df["competition"].to_numpy()
        X = _build_meta_X(all_results, competitions=comps)
        assert not np.any(np.isnan(X)), "Feature matrix should have no NaN"
        assert not np.any(np.isinf(X)), "Feature matrix should have no Inf"

    def test_onehot_competition_encoding(self, sample_df):
        from footy.models.council import _run_experts, _build_meta_X, ExpertResult
        results = _run_experts(sample_df)
        n = len(sample_df)
        dc_result = ExpertResult(
            probs=np.full((n, 3), 1/3),
            confidence=np.zeros(n),
            features={"dc_eg_h": np.zeros(n), "dc_eg_a": np.zeros(n),
                      "dc_o25": np.zeros(n), "dc_has": np.zeros(n)},
        )
        all_results = results + [dc_result]
        comps = sample_df["competition"].to_numpy()
        X_with = _build_meta_X(all_results, competitions=comps)
        X_without = _build_meta_X(all_results, competitions=None)
        # With competitions should have 5 more columns (one-hot for 5 leagues)
        assert X_with.shape[1] == X_without.shape[1] + 5


class TestAllExpertsV10:
    """Integration test: all 11 experts run end-to-end."""

    def test_all_experts_run(self, sample_df):
        from footy.models.council import ALL_EXPERTS, ExpertResult
        assert len(ALL_EXPERTS) == 11
        for expert in ALL_EXPERTS:
            result = expert.compute(sample_df)
            assert isinstance(result, ExpertResult)
            assert result.probs.shape == (len(sample_df), 3)
            assert result.confidence.shape == (len(sample_df),)
            # All probs valid
            assert np.all(result.probs >= 0)
            assert np.all(result.probs <= 1)
            row_sums = result.probs.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=0.02), \
                f"{expert.name}: row sums not 1: {row_sums}"

    def test_unique_expert_names(self):
        from footy.models.council import ALL_EXPERTS
        names = [e.name for e in ALL_EXPERTS]
        assert len(names) == len(set(names))

    def test_all_experts_have_features(self, sample_df):
        from footy.models.council import ALL_EXPERTS
        for expert in ALL_EXPERTS:
            result = expert.compute(sample_df)
            assert len(result.features) > 0, f"{expert.name} has no features"
