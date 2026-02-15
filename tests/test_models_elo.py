"""Unit tests — Elo rating model."""
import math
import pytest


class TestEloHelpers:
    """Pure-function Elo calculations."""

    def test_expected_equal_ratings(self):
        from footy.models.elo import _expected
        assert abs(_expected(1500, 1500) - 0.5) < 1e-6

    def test_expected_stronger_home(self):
        from footy.models.elo import _expected
        assert _expected(1600, 1400) > 0.5

    def test_expected_range(self):
        from footy.models.elo import _expected
        e = _expected(2000, 1000)
        assert 0.0 < e < 1.0

    def test_dynamic_k_new_team_higher(self):
        from footy.models.elo import _dynamic_k, _team_match_counts
        _team_match_counts.clear()
        k_new = _dynamic_k("NewTeam", 1)
        _team_match_counts["OldTeam"] = 200
        k_old = _dynamic_k("OldTeam", 1)
        assert k_new > k_old

    def test_dynamic_k_blowout_higher(self):
        from footy.models.elo import _dynamic_k
        k1 = _dynamic_k("T", 1)
        k5 = _dynamic_k("T", 5)
        assert k5 > k1


class TestEloDb:
    """Elo operations with DuckDB."""

    def test_ensure_team_default_rating(self, con):
        from footy.models.elo import ensure_team, DEFAULT_RATING
        ensure_team(con, "TestFC")
        r = con.execute("SELECT rating FROM elo_state WHERE team='TestFC'").fetchone()
        assert r is not None
        assert abs(float(r[0]) - DEFAULT_RATING) < 1e-6

    def test_get_set_rating(self, con):
        from footy.models.elo import get_rating, set_rating
        r0 = get_rating(con, "Club")
        set_rating(con, "Club", 1600.0)
        assert abs(get_rating(con, "Club") - 1600.0) < 1e-6

    def test_update_from_match_home_win(self, con):
        from footy.models.elo import get_rating, update_from_match
        r_home_before = get_rating(con, "HomeFC")
        r_away_before = get_rating(con, "AwayFC")
        update_from_match(con, "HomeFC", "AwayFC", 3, 0)
        assert get_rating(con, "HomeFC") > r_home_before
        assert get_rating(con, "AwayFC") < r_away_before

    def test_update_from_match_draw(self, con):
        from footy.models.elo import get_rating, update_from_match, DEFAULT_RATING
        update_from_match(con, "Draw1", "Draw2", 1, 1)
        # With home advantage, a draw against equal-rated team means home loses rating
        r1 = get_rating(con, "Draw1")
        r2 = get_rating(con, "Draw2")
        assert abs(r1 + r2 - 2 * DEFAULT_RATING) < 1e-6  # zero-sum

    def test_predict_probs_sum_to_one(self, con):
        from footy.models.elo import predict_probs
        ph, pd_, pa = predict_probs(con, "TeamA", "TeamB")
        assert abs(ph + pd_ + pa - 1.0) < 1e-6
        assert all(0 < p < 1 for p in [ph, pd_, pa])

    def test_predict_probs_home_advantage(self, con):
        from footy.models.elo import predict_probs
        ph, _, pa = predict_probs(con, "TeamX", "TeamX_clone")
        # Both teams default 1500, home advantage should give home edge
        assert ph > pa
