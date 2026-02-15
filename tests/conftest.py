"""
Shared fixtures for the Footy Predictor test suite.

Provides:
    - In-memory DuckDB connection with full schema
    - Seeded test data (matches, extras, elo, predictions)
    - Mock settings that don't require real API keys
    - Reusable helper factories
"""
from __future__ import annotations

import os
import datetime as dt

import duckdb
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Patch settings BEFORE any footy imports so config.settings() never fails
# ---------------------------------------------------------------------------
os.environ.setdefault("FOOTBALL_DATA_ORG_TOKEN", "test-token-fd")
os.environ.setdefault("API_FOOTBALL_KEY", "test-token-af")
os.environ.setdefault("DB_PATH", ":memory:")
os.environ.setdefault("TRACKED_COMPETITIONS", "PL,PD,SA,BL1")


# ---------------------------------------------------------------------------
# In-memory DuckDB connection with full schema
# ---------------------------------------------------------------------------
@pytest.fixture()
def con():
    """Fresh in-memory DuckDB connection with full schema applied."""
    from footy.db import SCHEMA_SQL

    c = duckdb.connect(":memory:")
    c.execute(SCHEMA_SQL)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------
_SEED_TEAMS = [
    ("Arsenal", "PL"),
    ("Chelsea", "PL"),
    ("Liverpool", "PL"),
    ("Manchester City", "PL"),
    ("Tottenham Hotspur", "PL"),
    ("Manchester United", "PL"),
    ("Real Madrid", "PD"),
    ("Barcelona", "PD"),
    ("Atletico Madrid", "PD"),
    ("Juventus", "SA"),
    ("Internazionale", "SA"),
    ("Milan", "SA"),
    ("Bayern München", "BL1"),
    ("Borussia Dortmund", "BL1"),
]


def _make_match(match_id, home, away, comp, status="FINISHED",
                hg=None, ag=None, date=None):
    """Build a match row dict."""
    if date is None:
        date = dt.datetime(2025, 8, 1, 15, 0, tzinfo=dt.timezone.utc) + dt.timedelta(days=match_id)
    if status == "FINISHED" and hg is None:
        rng = np.random.default_rng(match_id)
        hg = int(rng.poisson(1.5))
        ag = int(rng.poisson(1.2))
    return {
        "match_id": match_id,
        "provider": "test",
        "competition": comp,
        "season": 2025,
        "utc_date": date,
        "status": status,
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "raw_json": None,
    }


@pytest.fixture()
def seeded_con(con):
    """DuckDB connection pre-loaded with realistic test data.

    Creates ~100 finished matches + 8 scheduled matches across 4 leagues,
    plus Elo ratings, match extras, and predictions.
    """
    rng = np.random.default_rng(42)
    mid = 1
    matches = []

    # Generate finished matches (round-robin within each league, 2 cycles)
    league_teams = {}
    for team, comp in _SEED_TEAMS:
        league_teams.setdefault(comp, []).append(team)

    for comp, teams in league_teams.items():
        for cycle in range(2):
            for i, home in enumerate(teams):
                for j, away in enumerate(teams):
                    if i == j:
                        continue
                    m = _make_match(
                        mid, home, away, comp, "FINISHED",
                        date=dt.datetime(2025, 1, 1, 15, 0, tzinfo=dt.timezone.utc)
                        + dt.timedelta(days=mid),
                    )
                    matches.append(m)
                    mid += 1

    # Generate scheduled (upcoming) matches
    for comp, teams in league_teams.items():
        for i in range(0, len(teams) - 1, 2):
            m = _make_match(
                mid, teams[i], teams[i + 1], comp, "SCHEDULED",
                date=dt.datetime(2026, 2, 20, 15, 0, tzinfo=dt.timezone.utc)
                + dt.timedelta(days=i),
            )
            matches.append(m)
            mid += 1

    # Insert matches
    for m in matches:
        con.execute(
            """INSERT INTO matches(match_id, provider, competition, season,
               utc_date, status, home_team, away_team, home_goals, away_goals, raw_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [m["match_id"], m["provider"], m["competition"], m["season"],
             m["utc_date"], m["status"], m["home_team"], m["away_team"],
             m["home_goals"], m["away_goals"], m["raw_json"]],
        )

    # Seed Elo ratings
    for team, _comp in _SEED_TEAMS:
        rating = 1500 + rng.normal(0, 100)
        con.execute(
            "INSERT OR REPLACE INTO elo_state(team, rating) VALUES (?,?)",
            [team, rating],
        )

    # Seed match extras (odds) for finished matches
    finished = [m for m in matches if m["status"] == "FINISHED"]
    for m in finished[:40]:
        h_strength = rng.uniform(0.3, 0.5)
        d_strength = rng.uniform(0.2, 0.35)
        a_strength = 1.0 - h_strength - d_strength
        # odds = 1/implied * (1 + overround)
        overround = 1.05
        con.execute(
            """INSERT OR REPLACE INTO match_extras(match_id, provider, competition,
               b365h, b365d, b365a)
               VALUES (?,?,?,?,?,?)""",
            [m["match_id"], "test", m["competition"],
             round(overround / h_strength, 2),
             round(overround / d_strength, 2),
             round(overround / a_strength, 2)],
        )

    # Seed some predictions for scheduled matches
    scheduled = [m for m in matches if m["status"] == "SCHEDULED"]
    for m in scheduled:
        probs = rng.dirichlet([2, 1.5, 1.5])
        con.execute(
            """INSERT OR REPLACE INTO predictions(match_id, model_version,
               p_home, p_draw, p_away, eg_home, eg_away)
               VALUES (?,?,?,?,?,?,?)""",
            [m["match_id"], "v7_council",
             float(probs[0]), float(probs[1]), float(probs[2]),
             round(float(rng.exponential(1.4)), 2),
             round(float(rng.exponential(1.1)), 2)],
        )

    yield con


@pytest.fixture()
def finished_df(seeded_con):
    """DataFrame of finished matches from seeded database."""
    return seeded_con.execute("""
        SELECT match_id, home_team, away_team, home_goals, away_goals,
               utc_date, competition, status
        FROM matches WHERE status='FINISHED'
        ORDER BY utc_date
    """).fetchdf()
