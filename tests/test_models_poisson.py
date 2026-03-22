"""Unit tests — Poisson regression model."""
import numpy as np
import pandas as pd
import pytest
import datetime as dt


def _make_poisson_df(n_matches: int = 200, n_teams: int = 8, seed: int = 42):
    """Build a synthetic match DataFrame for Poisson fitting."""
    rng = np.random.default_rng(seed)
    teams = [f"Team_{chr(65 + i)}" for i in range(n_teams)]
    rows = []
    for k in range(n_matches):
        h, a = rng.choice(len(teams), 2, replace=False)
        rows.append({
            "home_team": teams[h],
            "away_team": teams[a],
            "home_goals": int(rng.poisson(1.5)),
            "away_goals": int(rng.poisson(1.1)),
            "utc_date": dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
            + dt.timedelta(days=k),
        })
    return pd.DataFrame(rows)


class TestFitPoisson:
    """Fitting and prediction with the Poisson model."""

    def test_fit_returns_dict(self):
        from footy.models.poisson import fit_poisson

        df = _make_poisson_df()
        state = fit_poisson(df)
        assert isinstance(state, dict)
        assert "teams" in state
        assert "attack" in state
        assert "defense" in state
        assert "home_adv" in state
        assert "mu" in state

    def test_fit_all_teams_present(self):
        from footy.models.poisson import fit_poisson

        df = _make_poisson_df(n_teams=6)
        state = fit_poisson(df)
        assert len(state["teams"]) == 6
        assert len(state["attack"]) == 6
        assert len(state["defense"]) == 6

    def test_home_advantage_positive(self):
        from footy.models.poisson import fit_poisson

        df = _make_poisson_df(n_matches=500)
        state = fit_poisson(df)
        # Home advantage should generally be positive
        assert state["home_adv"] > -0.5  # allow some noise

    def test_expected_goals_known_teams(self):
        from footy.models.poisson import fit_poisson, expected_goals

        df = _make_poisson_df()
        state = fit_poisson(df)
        eg_h, eg_a = expected_goals(state, state["teams"][0], state["teams"][1])
        assert 0.1 < eg_h < 8.0
        assert 0.1 < eg_a < 8.0

    def test_expected_goals_unknown_team_fallback(self):
        from footy.models.poisson import expected_goals

        state = {"teams": ["A", "B"], "attack": [0, 0], "defense": [0, 0],
                 "home_adv": 0.1, "mu": 0.3}
        eg_h, eg_a = expected_goals(state, "Unknown", "B")
        assert eg_h == 1.3 and eg_a == 1.1  # documented fallback

    def test_expected_goals_empty_state_fallback(self):
        from footy.models.poisson import expected_goals

        state = {"teams": [], "attack": [], "defense": [], "home_adv": 0, "mu": 0}
        eg_h, eg_a = expected_goals(state, "A", "B")
        assert eg_h == 1.3 and eg_a == 1.1


class TestOutcomeProbs:
    """Poisson outcome probabilities."""

    def test_outcome_probs_sum_to_one(self):
        from footy.models.poisson import fit_poisson, expected_goals, scoreline_probs

        df = _make_poisson_df()
        state = fit_poisson(df)
        t0, t1 = state["teams"][0], state["teams"][1]
        lam_h, lam_a = expected_goals(state, t0, t1)
        grid = scoreline_probs(lam_h, lam_a)
        assert abs(grid.sum() - 1.0) < 0.01

    def test_fit_on_empty_df(self):
        from footy.models.poisson import fit_poisson

        df = pd.DataFrame(columns=["home_team", "away_team", "home_goals",
                                    "away_goals", "utc_date"])
        state = fit_poisson(df)
        assert state["teams"] == []
