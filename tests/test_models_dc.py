"""Unit tests — Dixon-Coles model."""
import numpy as np
import pandas as pd
import pytest
import datetime as dt


def _make_dc_df(n: int = 400, n_teams: int = 10, seed: int = 99):
    rng = np.random.default_rng(seed)
    teams = [f"Team_{i}" for i in range(n_teams)]
    rows = []
    for k in range(n):
        h, a = rng.choice(n_teams, 2, replace=False)
        rows.append({
            "home_team": teams[h],
            "away_team": teams[a],
            "home_goals": int(rng.poisson(1.4)),
            "away_goals": int(rng.poisson(1.1)),
            "utc_date": dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
            + dt.timedelta(days=k),
        })
    return pd.DataFrame(rows)


class TestDixonColes:
    """Dixon-Coles fitting and 1×2 prediction."""

    def test_fit_returns_model(self):
        from footy.models.dixon_coles import fit_dc

        df = _make_dc_df()
        model = fit_dc(df)
        assert model is not None
        assert len(model.teams) == 10
        assert len(model.atk) == 10
        assert len(model.dfn) == 10

    def test_fit_too_few_matches_returns_none(self):
        from footy.models.dixon_coles import fit_dc

        df = _make_dc_df(n=50)
        assert fit_dc(df) is None

    def test_fit_rho_bounded(self):
        from footy.models.dixon_coles import fit_dc

        model = fit_dc(_make_dc_df())
        assert -0.35 <= model.rho <= 0.35

    def test_predict_1x2_sums_to_one(self):
        from footy.models.dixon_coles import fit_dc, predict_1x2

        model = fit_dc(_make_dc_df())
        # Returns (p_home, p_draw, p_away, eg_home, eg_away, p_over25)
        p = predict_1x2(model, model.teams[0], model.teams[1])
        assert len(p) == 6
        ph, pd_, pa = p[0], p[1], p[2]
        assert abs(ph + pd_ + pa - 1.0) < 0.01
        assert all(0 < x < 1 for x in [ph, pd_, pa])

    def test_predict_unknown_team_returns_zeros(self):
        from footy.models.dixon_coles import fit_dc, predict_1x2

        model = fit_dc(_make_dc_df())
        p = predict_1x2(model, "Unknown_FC", model.teams[0])
        # Unknown team → all zeros (graceful fail)
        assert p == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_tau_low_scores(self):
        from footy.models.dixon_coles import _tau

        # 0-0 with negative rho should increase probability
        t00 = _tau(0, 0, 1.0, 1.0, -0.1)
        assert t00 > 1.0  # rho negative → 1 - (lam*mu*rho) > 1

        # 1-1 with negative rho should increase probability
        t11 = _tau(1, 1, 1.0, 1.0, -0.1)
        assert t11 > 1.0

        # High-scoring, tau = 1
        assert _tau(3, 2, 1.0, 1.0, -0.1) == 1.0
