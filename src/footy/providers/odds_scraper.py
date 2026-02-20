"""
Odds utilities for upcoming matches.

Provides:
- update_upcoming_match_odds(): fetch real-time odds via The Odds API
- fill_upcoming_odds_from_predictions(): generate implied odds from model predictions
- implied_odds_from_probability(): convert probabilities to decimal odds
"""

from __future__ import annotations

import logging
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)


def update_upcoming_match_odds(
    con: duckdb.DuckDBPyConnection,
    verbose: bool = False,
) -> dict:
    """Fetch real-time odds for upcoming matches via The Odds API.

    Delegates to the_odds_api provider which handles h2h, totals,
    Asian Handicap, and BTTS markets from 40+ bookmakers.
    """
    from footy.providers.the_odds_api import fetch_odds, ingest_odds_to_db

    try:
        events = fetch_odds()
    except Exception as e:
        logger.warning("Could not fetch odds from The Odds API: %s", e)
        return {"updated": 0, "error": str(e)}

    if not events:
        if verbose:
            logger.info("[odds] no events returned from The Odds API")
        return {"updated": 0, "attempted": 0}

    updated = ingest_odds_to_db(con, events, verbose=verbose)
    return {"updated": updated, "attempted": len(events)}


def implied_odds_from_probability(
    p_home: float,
    p_draw: float,
    p_away: float,
    margin: float = 0.02,
) -> dict:
    """Convert predicted probabilities to decimal odds (European format).

    Args:
        p_home: Home win probability (0-1)
        p_draw: Draw probability (0-1)
        p_away: Away win probability (0-1)
        margin: Bookmaker margin (default 2%)

    Returns:
        dict with home_odds, draw_odds, away_odds
    """
    total = p_home + p_draw + p_away
    if total <= 0:
        return {"home_odds": 2.0, "draw_odds": 3.0, "away_odds": 2.0}

    p_home_norm = p_home / total
    p_draw_norm = p_draw / total
    p_away_norm = p_away / total

    home_odds = (1.0 + margin) / p_home_norm
    draw_odds = (1.0 + margin) / p_draw_norm
    away_odds = (1.0 + margin) / p_away_norm

    return {
        "home_odds": float(home_odds),
        "draw_odds": float(draw_odds),
        "away_odds": float(away_odds),
    }


def fill_upcoming_odds_from_predictions(
    con: duckdb.DuckDBPyConnection,
    verbose: bool = False,
) -> dict:
    """Fill odds for upcoming matches using model predictions as fallback.

    For matches without external odds, generates implied odds from the
    latest model predictions.
    """
    upcoming = con.execute("""
        SELECT m.match_id, m.home_team, m.away_team
        FROM matches m
        LEFT JOIN match_extras me ON m.match_id = me.match_id
        WHERE m.status IN ('SCHEDULED', 'TIMED')
          AND (me.match_id IS NULL OR me.b365h IS NULL OR me.b365h = 0)
        ORDER BY m.utc_date ASC
        LIMIT 100
    """).df()

    if upcoming.empty:
        if verbose:
            logger.info("[odds] no upcoming matches needing implied odds")
        return {"filled": 0, "attempted": 0}

    filled = 0

    for _, row in upcoming.iterrows():
        match_id = int(row["match_id"])
        home_team = row["home_team"]
        away_team = row["away_team"]

        pred = con.execute("""
            SELECT p_home, p_draw, p_away, model_version
            FROM predictions
            WHERE match_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, [match_id]).df()

        if not pred.empty:
            row_pred = pred.iloc[0]
            p_h = float(row_pred["p_home"])
            p_d = float(row_pred["p_draw"])
            p_a = float(row_pred["p_away"])

            implied = implied_odds_from_probability(p_h, p_d, p_a, margin=0.02)

            try:
                con.execute("""
                    INSERT INTO match_extras (
                        match_id, provider, competition,
                        b365h, b365d, b365a, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(match_id) DO UPDATE SET
                        b365h = COALESCE(excluded.b365h, match_extras.b365h),
                        b365d = COALESCE(excluded.b365d, match_extras.b365d),
                        b365a = COALESCE(excluded.b365a, match_extras.b365a)
                """, [
                    match_id,
                    "implied_odds",
                    "upcoming",
                    implied["home_odds"],
                    implied["draw_odds"],
                    implied["away_odds"],
                    f'{{"model": "{row_pred["model_version"]}", "source": "prediction"}}',
                ])

                filled += 1

                if verbose:
                    logger.info(
                        "[odds] %s vs %s: p=%.3f-%.3f-%.3f -> odds=%.2f-%.2f-%.2f",
                        home_team, away_team, p_h, p_d, p_a,
                        implied["home_odds"], implied["draw_odds"], implied["away_odds"],
                    )

            except Exception as e:
                if verbose:
                    logger.warning("[odds] Failed to insert implied odds for match %d: %s", match_id, e)

    return {"filled": filled, "attempted": len(upcoming)}
