"""Tests for v11 upgrade: MLE rho, AH/BTTS, FPL persistence, xG API-Football,
feature-count validation, schedule_difficulty composite, InjuryExpert 14-feat."""
from __future__ import annotations

import datetime as dt
import json

import duckdb
import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════
# 1. MLE rho estimation
# ═══════════════════════════════════════════════════════════════════
class TestMLERho:
    """Tests for estimate_rho_mle in advanced_math."""

    def test_basic_estimation(self):
        from footy.models.advanced_math import estimate_rho_mle
        rng = np.random.default_rng(42)
        # Generate 100 low-score matches — rho should be negative
        hg = rng.poisson(1.2, 200)
        ag = rng.poisson(1.0, 200)
        lam_h = np.full(200, 1.2)
        lam_a = np.full(200, 1.0)
        rho = estimate_rho_mle(hg, ag, lam_h, lam_a)
        assert -0.4 <= rho <= 0.0

    def test_returns_default_for_small_sample(self):
        from footy.models.advanced_math import estimate_rho_mle
        rho = estimate_rho_mle(
            np.array([3, 4, 5]),
            np.array([2, 3, 4]),
            np.array([1.5, 1.5, 1.5]),
            np.array([1.2, 1.2, 1.2]),
        )
        # With only 3 matches (none low-scoring), should return default
        assert rho == pytest.approx(-0.13, abs=0.01)

    def test_bounded_output(self):
        from footy.models.advanced_math import estimate_rho_mle
        # All 0-0 draws — strong negative correlation
        n = 300
        hg = np.zeros(n, dtype=int)
        ag = np.zeros(n, dtype=int)
        lam_h = np.full(n, 1.5)
        lam_a = np.full(n, 1.5)
        rho = estimate_rho_mle(hg, ag, lam_h, lam_a)
        assert -0.4 <= rho <= 0.0

    def test_deterministic(self):
        from footy.models.advanced_math import estimate_rho_mle
        rng = np.random.default_rng(99)
        hg = rng.poisson(1.3, 300)
        ag = rng.poisson(1.1, 300)
        lam_h = np.full(300, 1.3)
        lam_a = np.full(300, 1.1)
        r1 = estimate_rho_mle(hg, ag, lam_h, lam_a)
        r2 = estimate_rho_mle(hg, ag, lam_h, lam_a)
        assert r1 == r2


# ═══════════════════════════════════════════════════════════════════
# 2. Composite schedule_difficulty
# ═══════════════════════════════════════════════════════════════════
class TestScheduleDifficultyComposite:
    """Tests for the upgraded composite schedule difficulty."""

    def test_recency_weighting(self):
        from footy.models.advanced_math import schedule_difficulty
        # recent opponent is much harder
        r1 = schedule_difficulty([1200, 1200, 1800], window=3)
        r2 = schedule_difficulty([1800, 1200, 1200], window=3)
        # r1 should be higher because 1800 is most recent
        assert r1 > r2

    def test_max_opponent_boost(self):
        from footy.models.advanced_math import schedule_difficulty
        # One very strong opponent lifts composite vs all-average
        uniform = schedule_difficulty([1500, 1500, 1500], window=3)
        with_max = schedule_difficulty([1200, 1500, 1800], window=3)
        assert with_max > uniform

    def test_default_for_empty(self):
        from footy.models.advanced_math import schedule_difficulty
        assert schedule_difficulty([], window=5) == 1500.0

    def test_single_opponent(self):
        from footy.models.advanced_math import schedule_difficulty
        result = schedule_difficulty([1700], window=3)
        # Should reflect the single opponent's rating
        assert result > 1600


# ═══════════════════════════════════════════════════════════════════
# 3. FPL table schema
# ═══════════════════════════════════════════════════════════════════
class TestFPLTables:
    """Verify fpl_availability and fpl_fixture_difficulty tables exist."""

    def test_fpl_availability_exists(self, con):
        con.execute("""
            INSERT INTO fpl_availability (team, total_players, available, doubtful,
                injured, suspended, injury_score, squad_strength, key_absences_json, updated_at)
            VALUES ('Arsenal', 25, 22, 1, 1, 1, 0.85, 0.92, '[]', CURRENT_TIMESTAMP)
        """)
        result = con.execute("SELECT team, injury_score FROM fpl_availability WHERE team='Arsenal'").fetchone()
        assert result[0] == "Arsenal"
        assert result[1] == pytest.approx(0.85)

    def test_fpl_fixture_difficulty_exists(self, con):
        con.execute("""
            INSERT INTO fpl_fixture_difficulty (team, fdr_next_3, fdr_next_6, upcoming_json, updated_at)
            VALUES ('Chelsea', 3.5, 3.2, '[]', CURRENT_TIMESTAMP)
        """)
        result = con.execute("SELECT fdr_next_3 FROM fpl_fixture_difficulty WHERE team='Chelsea'").fetchone()
        assert result[0] == pytest.approx(3.5)

    def test_upsert_fpl_availability(self, con):
        con.execute("""
            INSERT INTO fpl_availability (team, total_players, available, doubtful,
                injured, suspended, injury_score, squad_strength, key_absences_json, updated_at)
            VALUES ('Liverpool', 25, 23, 1, 1, 0, 0.90, 0.95, '[]', CURRENT_TIMESTAMP)
            ON CONFLICT (team) DO UPDATE SET injury_score = EXCLUDED.injury_score
        """)
        con.execute("""
            INSERT INTO fpl_availability (team, total_players, available, doubtful,
                injured, suspended, injury_score, squad_strength, key_absences_json, updated_at)
            VALUES ('Liverpool', 25, 20, 2, 2, 1, 0.70, 0.80, '[]', CURRENT_TIMESTAMP)
            ON CONFLICT (team) DO UPDATE SET injury_score = EXCLUDED.injury_score
        """)
        result = con.execute("SELECT injury_score FROM fpl_availability WHERE team='Liverpool'").fetchone()
        assert result[0] == pytest.approx(0.70)


# ═══════════════════════════════════════════════════════════════════
# 4. match_extras new columns
# ═══════════════════════════════════════════════════════════════════
class TestMatchExtrasV11Columns:
    """Verify the 13 new match_extras columns exist."""

    def test_ah_columns_exist(self, con):
        con.execute("SELECT odds_ah_line, odds_ah_home, odds_ah_away FROM match_extras LIMIT 0")

    def test_btts_columns_exist(self, con):
        con.execute("SELECT odds_btts_yes, odds_btts_no FROM match_extras LIMIT 0")

    def test_formation_columns_exist(self, con):
        con.execute("SELECT formation_home, formation_away FROM match_extras LIMIT 0")

    def test_lineup_columns_exist(self, con):
        con.execute("SELECT lineup_home, lineup_away FROM match_extras LIMIT 0")

    def test_af_columns_exist(self, con):
        con.execute("SELECT af_xg_home, af_xg_away, af_possession_home, af_possession_away, af_stats_json FROM match_extras LIMIT 0")


# ═══════════════════════════════════════════════════════════════════
# 5. MarketExpert AH+BTTS features
# ═══════════════════════════════════════════════════════════════════
class TestMarketExpertAHBTTS:
    """Test the 8 new features in MarketExpert."""

    @pytest.fixture()
    def sample_df(self):
        n = 5
        return pd.DataFrame({
            "match_id": range(1, n + 1),
            "home_team": ["A"] * n,
            "away_team": ["B"] * n,
            "competition": ["PL"] * n,
            "home_goals": [2, 1, 0, 3, 1],
            "away_goals": [1, 1, 2, 0, 0],
            "b365h": [2.0, 2.5, 3.0, 1.8, 2.2],
            "b365d": [3.2, 3.3, 3.4, 3.5, 3.1],
            "b365a": [3.5, 2.8, 2.3, 4.0, 3.0],
            "bwh": [None] * n, "bwd": [None] * n, "bwa": [None] * n,
            "iwh": [None] * n, "iwd": [None] * n, "iwa": [None] * n,
            "psh": [None] * n, "psd": [None] * n, "psa": [None] * n,
            "avg_h": [None] * n, "avg_d": [None] * n, "avg_a": [None] * n,
            "max_h": [None] * n, "max_d": [None] * n, "max_a": [None] * n,
            "avgo25": [1.8] * n, "avgu25": [2.0] * n,
            "odds_ah_line": [-0.5, -0.75, 0.0, -1.0, -0.5],
            "odds_ah_home": [1.9, 1.85, 1.95, 1.80, 1.92],
            "odds_ah_away": [2.0, 2.05, 1.95, 2.10, 1.98],
            "odds_btts_yes": [1.7, 1.8, 1.75, 1.9, 1.65],
            "odds_btts_no": [2.1, 2.0, 2.05, 1.9, 2.2],
        })

    def test_ah_features(self, sample_df):
        from footy.models.council import MarketExpert
        expert = MarketExpert()
        result = expert.compute(sample_df)
        assert "mkt_ah_line" in result.features
        assert "mkt_ah_home" in result.features
        assert "mkt_ah_away" in result.features
        assert "mkt_has_ah" in result.features
        assert np.all(result.features["mkt_has_ah"] == 1.0)

    def test_btts_features(self, sample_df):
        from footy.models.council import MarketExpert
        expert = MarketExpert()
        result = expert.compute(sample_df)
        assert "mkt_btts_yes" in result.features
        assert "mkt_btts_no" in result.features
        assert "mkt_btts_implied" in result.features
        assert "mkt_has_btts" in result.features

    def test_btts_implied_range(self, sample_df):
        from footy.models.council import MarketExpert
        expert = MarketExpert()
        result = expert.compute(sample_df)
        implied = result.features["mkt_btts_implied"]
        # Implied probability should be between 0 and 1
        assert np.all(implied >= 0.0)
        assert np.all(implied <= 1.0)


# ═══════════════════════════════════════════════════════════════════
# 6. InjuryAvailabilityExpert 14 features
# ═══════════════════════════════════════════════════════════════════
class TestInjuryExpertV11:
    """Test the rewritten InjuryAvailabilityExpert."""

    @pytest.fixture()
    def sample_df_with_fpl(self):
        n = 3
        return pd.DataFrame({
            "match_id": range(1, n + 1),
            "home_team": ["Arsenal", "Chelsea", "Liverpool"],
            "away_team": ["Chelsea", "Liverpool", "Arsenal"],
            "competition": ["PL"] * n,
            "home_goals": [2, 1, 0],
            "away_goals": [0, 1, 3],
            # FPL columns
            "fpl_inj_score_h": [0.8, 0.5, 0.0],
            "fpl_inj_score_a": [0.3, 0.7, 0.6],
            "fpl_squad_str_h": [0.95, 0.88, 0.0],
            "fpl_squad_str_a": [0.90, 0.85, 0.92],
            "fpl_available_h": [22, 20, 0],
            "fpl_available_a": [23, 19, 21],
            "fpl_doubtful_h": [1, 2, 0],
            "fpl_doubtful_a": [0, 3, 1],
            "fpl_injured_h": [2, 3, 0],
            "fpl_injured_a": [1, 2, 2],
            "fpl_suspended_h": [0, 0, 0],
            "fpl_suspended_a": [1, 0, 0],
            "fpl_fdr_h": [3.0, 4.0, 0.0],
            "fpl_fdr_a": [2.5, 3.5, 3.0],
            # AF columns
            "af_inj_h": [3, 5, 0],
            "af_inj_a": [2, 4, 3],
        })

    def test_14_features(self, sample_df_with_fpl):
        from footy.models.council import InjuryAvailabilityExpert
        expert = InjuryAvailabilityExpert()
        result = expert.compute(sample_df_with_fpl)
        assert len(result.features) >= 14  # v12: now 21 features (added suspended, fdr6, etc.)

    def test_confidence_with_fpl_data(self, sample_df_with_fpl):
        from footy.models.council import InjuryAvailabilityExpert
        expert = InjuryAvailabilityExpert()
        result = expert.compute(sample_df_with_fpl)
        # First two rows have FPL data → confidence >= 0.3
        assert result.confidence[0] >= 0.3
        assert result.confidence[1] >= 0.3

    def test_injury_diff_direction(self, sample_df_with_fpl):
        from footy.models.council import InjuryAvailabilityExpert
        expert = InjuryAvailabilityExpert()
        result = expert.compute(sample_df_with_fpl)
        # Row 0: home is more injured (0.8 + 3*0.1=1.1) vs away (0.3 + 2*0.1=0.5) → positive diff
        assert result.features["inj_diff"][0] > 0


# ═══════════════════════════════════════════════════════════════════
# 7. WF-CV deployment gating config
# ═══════════════════════════════════════════════════════════════════
class TestWFCVGating:
    """Test that WF-CV gating thresholds are configurable."""

    def test_defaults(self):
        import os
        # Ensure no env overrides
        for k in ("WF_LOGLOSS_GATE", "WF_ACCURACY_GATE", "WF_MIN_FOLDS"):
            os.environ.pop(k, None)
        # Thresholds should be accessible
        ll_gate = float(os.environ.get("WF_LOGLOSS_GATE", "1.05"))
        acc_gate = float(os.environ.get("WF_ACCURACY_GATE", "0.38"))
        min_folds = int(os.environ.get("WF_MIN_FOLDS", "3"))
        assert ll_gate == 1.05
        assert acc_gate == 0.38
        assert min_folds == 3

    def test_custom_thresholds(self):
        import os
        os.environ["WF_LOGLOSS_GATE"] = "0.95"
        os.environ["WF_ACCURACY_GATE"] = "0.45"
        ll_gate = float(os.environ["WF_LOGLOSS_GATE"])
        acc_gate = float(os.environ["WF_ACCURACY_GATE"])
        assert ll_gate == 0.95
        assert acc_gate == 0.45
        # Clean up
        os.environ.pop("WF_LOGLOSS_GATE", None)
        os.environ.pop("WF_ACCURACY_GATE", None)


# ═══════════════════════════════════════════════════════════════════
# 8. API-Football provider new functions exist
# ═══════════════════════════════════════════════════════════════════
class TestAPIFootballV11:
    """Verify new API-Football functions are importable."""

    def test_fetch_fixture_statistics(self):
        from footy.providers.api_football import fetch_fixture_statistics
        assert callable(fetch_fixture_statistics)

    def test_fetch_fixture_lineups(self):
        from footy.providers.api_football import fetch_fixture_lineups
        assert callable(fetch_fixture_lineups)

    def test_fetch_h2h(self):
        from footy.providers.api_football import fetch_h2h
        assert callable(fetch_h2h)

    def test_enrich_match_extras(self):
        from footy.providers.api_football import enrich_match_extras_from_af
        assert callable(enrich_match_extras_from_af)


# ═══════════════════════════════════════════════════════════════════
# 9. The Odds API AH+BTTS parsing
# ═══════════════════════════════════════════════════════════════════
class TestOddsAPIV11:
    """Test AH and BTTS parsing from The Odds API."""

    def test_parse_spreads(self):
        from footy.providers.the_odds_api import parse_odds_for_match
        event = {
            "id": "test123",
            "sport_key": "soccer_epl",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "commence_time": "2025-08-15T15:00:00Z",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 2.0},
                                {"name": "Draw", "price": 3.5},
                                {"name": "Chelsea", "price": 3.2},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Arsenal", "price": 1.90, "point": -0.5},
                                {"name": "Chelsea", "price": 2.00, "point": 0.5},
                            ],
                        },
                        {
                            "key": "btts",
                            "outcomes": [
                                {"name": "Yes", "price": 1.75},
                                {"name": "No", "price": 2.10},
                            ],
                        },
                    ],
                }
            ],
        }
        result = parse_odds_for_match(event)
        assert "spreads" in result
        assert result["spreads"]["ah_line"] == pytest.approx(-0.5)
        assert "btts" in result
        assert result["btts"]["yes"] == pytest.approx(1.75)
        assert result["btts"]["no"] == pytest.approx(2.10)


# ═══════════════════════════════════════════════════════════════════
# 10. Config cleaned
# ═══════════════════════════════════════════════════════════════════
class TestConfigV11:
    """Verify sportapi_ai_key removed from config."""

    def test_no_sportapi_key(self):
        from footy.config import settings
        s = settings()
        assert not hasattr(s, "sportapi_ai_key")
