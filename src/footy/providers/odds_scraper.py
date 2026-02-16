"""
Alternative odds sources for upcoming matches.

Currently supports:
- ODD-API (free tier, if available)
- BetExplorer (web scraping, free)
- Pinnacle API (if available)

Goal: Fill in odds for upcoming matches where football-data.co.uk doesn't have them yet.
"""

from __future__ import annotations
import asyncio
import os
import httpx
from typing import Optional
import duckdb
import logging

logger = logging.getLogger(__name__)

# ODD-API free tier config
ODD_API_KEY = os.getenv("ODD_API_KEY", "")
ODD_API_BASE = "https://api.api-odds.com"

async def fetch_odds_from_odd_api(
    home_team: str,
    away_team: str,
    league: str,
) -> Optional[dict]:
    """
    Fetch odds from ODD-API (free tier).
    
    Returns:
        dict with odds from multiple bookmakers or None if not available
    """
    if not ODD_API_KEY:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Query upcoming matches
            resp = await client.get(
                f"{ODD_API_BASE}/odds",
                params={
                    "apiKey": ODD_API_KEY,
                    "sport": "soccer",
                    "region": "uk",
                    "oddsFormat": "decimal",
                }
            )
            
            if resp.status_code != 200:
                logger.warning(f"ODD-API error: {resp.status_code}")
                return None
            
            data = resp.json()
            
            # Parse response and find match
            if "games" not in data:
                return None
            
            for game in data["games"]:
                h = game.get("home_team", "").lower()
                a = game.get("away_team", "").lower()
                
                if home_team.lower() in h and away_team.lower() in a:
                    # Found match - extract odds
                    odds = {}
                    if "bookmakers" in game:
                        for bookie in game["bookmakers"]:
                            name = bookie.get("key", "")
                            markets = bookie.get("markets", [])
                            
                            for market in markets:
                                if market.get("key") == "h2h":
                                    outcomes = market.get("outcomes", [])
                                    if len(outcomes) >= 3:
                                        odds[name] = {
                                            "home": float(outcomes[0].get("price", 0)),
                                            "draw": float(outcomes[1].get("price", 0)),
                                            "away": float(outcomes[2].get("price", 0)),
                                        }
                    
                    return odds if odds else None
            
            return None
    
    except Exception as e:
        logger.error(f"ODD-API fetch error: {e}")
        return None


async def fetch_odds_from_betexplorer(
    home_team: str,
    away_team: str,
    league: str,
) -> Optional[dict]:
    """
    Fetch odds from BetExplorer (web scraping).
    
    Note: BetExplorer requires scraping their website directly.
    This is a placeholder - actual implementation requires BeautifulSoup/Selenium
    and careful rate limiting to comply with their ToS.
    
    Returns:
        dict with Pinnacle-style odds or None
    """
    # This would require BeautifulSoup + careful scraping
    # For now, return None - implement if you want to add this source
    logger.info(f"BetExplorer scraping not yet implemented for {home_team} vs {away_team}")
    return None


async def fetch_odds_from_thesportsdb(
    home_team: str,
    away_team: str,
    league: str,
) -> Optional[dict]:
    """
    Fetch event odds from TheSportsDB (free API).
    
    TheSportsDB has free API for event metadata but limited odds data.
    Mostly useful for event IDs and match metadata.
    
    Returns:
        dict with whatever odds TheSportsDB provides or None
    """
    try:
        # Map league codes to TheSportsDB league IDs
        LEAGUE_IDS = {
            "PL": "4328", "PD": "4335", "SA": "4332",
            "BL1": "4331", "FL1": "4334",
        }
        league_id = LEAGUE_IDS.get(league, "4328")

        async with httpx.AsyncClient(timeout=10) as client:
            # Search for upcoming events
            resp = await client.get(
                "https://www.thesportsdb.com/api/v1/json/3/eventslast.php",
                params={
                    "id": league_id,
                }
            )
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            if "results" not in data:
                return None
            
            # TheSportsDB provides minimal odds; mostly useful for metadata
            # Real odds would need to come from dedicated sports betting APIs
            return None
    
    except Exception as e:
        logger.warning(f"TheSportsDB fetch error: {e}")
        return None


async def fetch_odds_multi_source(
    home_team: str,
    away_team: str,
    league: str,
    sources: list[str] = None,
) -> dict:
    """
    Fetch odds from multiple sources in parallel.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        league: League code
        sources: List of sources to try ('odd-api', 'betexplorer', 'thesportsdb')
    
    Returns:
        dict with format {source: odds_dict, ...}
    """
    if sources is None:
        sources = ["odd-api", "thesportsdb"]  # betexplorer requires special handling
    
    tasks = []
    source_map = {}
    
    if "odd-api" in sources:
        tasks.append(fetch_odds_from_odd_api(home_team, away_team, league))
        source_map[len(tasks) - 1] = "odd-api"
    
    if "thesportsdb" in sources:
        tasks.append(fetch_odds_from_thesportsdb(home_team, away_team, league))
        source_map[len(tasks) - 1] = "thesportsdb"
    
    if "betexplorer" in sources:
        tasks.append(fetch_odds_from_betexplorer(home_team, away_team, league))
        source_map[len(tasks) - 1] = "betexplorer"
    
    if not tasks:
        return {}
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    output = {}
    for idx, result in enumerate(results):
        source = source_map.get(idx)
        if source and not isinstance(result, Exception):
            if result:
                output[source] = result
    
    return output


def update_upcoming_match_odds(
    con: duckdb.DuckDBPyConnection,
    verbose: bool = False,
) -> dict:
    """
    Update odds for all upcoming matches using alternative sources.
    
    Args:
        con: DuckDB connection
        verbose: Whether to print progress
    
    Returns:
        dict with stats on how many matches were updated
    """
    # Get upcoming matches without odds
    upcoming = con.execute("""
        SELECT m.match_id, m.home_team, m.away_team, m.competition
        FROM matches m
        LEFT JOIN match_extras me ON m.match_id = me.match_id
        WHERE m.status IN ('SCHEDULED', 'TIMED')
          AND (me.match_id IS NULL OR me.b365h IS NULL)
        ORDER BY m.utc_date ASC
        LIMIT 50
    """).df()
    
    if upcoming.empty:
        if verbose:
            logger.info("[odds] no upcoming matches to update")
        return {"updated": 0, "attempted": 0}
    
    updated = 0

    async def _fetch_all():
        nonlocal updated
        for _, row in upcoming.iterrows():
            match_id = int(row["match_id"])
            home = row["home_team"]
            away = row["away_team"]
            league = row["competition"]

            try:
                odds_sources = await fetch_odds_multi_source(home, away, league)

                if odds_sources:
                    # Choose best source (prefer odd-api for completeness)
                    best_odds = None
                    best_source = "unknown"

                    if "odd-api" in odds_sources:
                        # odd-api returns dict of bookmakers with h/d/a odds
                        # Try to find Bet365 or Pinnacle equivalent
                        for bookie, odds in odds_sources["odd-api"].items():
                            if bookie in ["bet365", "pinnacle"]:
                                best_odds = odds
                                best_source = bookie
                                break

                        if not best_odds and odds_sources["odd-api"]:
                            # Fall back to first available
                            best_odds = next(iter(odds_sources["odd-api"].values()))
                            best_source = "odd-api"

                    if best_odds:
                        # Store in match_extras
                        con.execute("""
                            INSERT INTO match_extras (
                                match_id, provider, competition,
                                b365h, b365d, b365a
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(match_id) DO UPDATE SET
                                b365h = ?,
                                b365d = ?,
                                b365a = ?
                        """, [
                            match_id, f"odds_{best_source}", league,
                            best_odds.get("home", 0),
                            best_odds.get("draw", 0),
                            best_odds.get("away", 0),
                            best_odds.get("home", 0),
                            best_odds.get("draw", 0),
                            best_odds.get("away", 0),
                        ])

                        updated += 1

                        if verbose:
                            logger.info(
                                f"[odds] {home} vs {away}: "
                                f"{best_odds['home']:.2f}-{best_odds['draw']:.2f}-{best_odds['away']:.2f} "
                                f"({best_source})"
                            )

            except Exception as e:
                if verbose:
                    logger.warning(f"[odds] Failed to fetch odds for match {match_id}: {e}")

    asyncio.run(_fetch_all())
    return {"updated": updated, "attempted": len(upcoming)}


def get_odds_ensemble(
    con: duckdb.DuckDBPyConnection,
    match_id: int,
) -> Optional[dict]:
    """
    Get consensus odds from multiple sources (if available).
    
    For upcoming matches, we may have odds from multiple bookmakers.
    This function returns a consensus or best estimate.
    
    Returns:
        dict with home_odds, draw_odds, away_odds or None
    """
    result = con.execute("""
        SELECT b365h, b365d, b365a, provider
        FROM match_extras
        WHERE match_id = ?
        ORDER BY provider ASC
    """, [match_id]).df()
    
    if result.empty:
        return None
    
    # Simple average across all sources (could be weighted)
    h = result["b365h"].mean()
    d = result["b365d"].mean()
    a = result["b365a"].mean()
    
    # Convert back to probabilities if needed
    return {
        "match_id": match_id,
        "home_odds": float(h),
        "draw_odds": float(d),
        "away_odds": float(a),
        "num_sources": len(result),
    }


def implied_odds_from_probability(
    p_home: float,
    p_draw: float,
    p_away: float,
    margin: float = 0.02,
) -> dict:
    """
    Convert predicted probabilities to decimal odds using European format.
    
    Args:
        p_home: Predicted home win probability (0-1)
        p_draw: Predicted draw probability (0-1)
        p_away: Predicted away win probability (0-1)
        margin: Bookmaker margin (typically 2-5%, default 2%)
    
    Returns:
        dict with {"home_odds": float, "draw_odds": float, "away_odds": float}
    
    Formula:
        1. Adjust probabilities for bookmaker margin: p' = p / (1 + margin)
        2. Convert to decimal odds: odd = 1 / p'
    """
    # Normalize probabilities
    total = p_home + p_draw + p_away
    if total <= 0:
        return {"home_odds": 2.0, "draw_odds": 3.0, "away_odds": 2.0}
    
    p_home_norm = p_home / total
    p_draw_norm = p_draw / total
    p_away_norm = p_away / total
    
    # Apply margin and convert to odds
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
    """
    Fill odds for upcoming matches using latest model predictions.
    
    This is a fallback when external odds sources aren't available.
    Uses the latest model predictions to generate implied odds.
    
    Args:
        con: DuckDB connection
        verbose: Whether to print progress
    
    Returns:
        dict with stats on how many matches were filled
    """
    # Get upcoming matches without odds
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
            logger.info("[odds] no upcoming matches to fill")
        return {"filled": 0, "attempted": 0}
    
    filled = 0
    
    for _, row in upcoming.iterrows():
        match_id = int(row["match_id"])
        home_team = row["home_team"]
        away_team = row["away_team"]
        
        # Get latest prediction for this match
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
            
            # Generate implied odds
            implied = implied_odds_from_probability(p_h, p_d, p_a, margin=0.02)
            
            # Store in match_extras
            try:
                con.execute("""
                    INSERT INTO match_extras (
                        match_id, provider, competition,
                        b365h, b365d, b365a, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(match_id) DO UPDATE SET
                        b365h = COALESCE(excluded.b365h, b365h),
                        b365d = COALESCE(excluded.b365d, b365d),
                        b365a = COALESCE(excluded.b365a, b365a)
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
                        f"[odds] {home_team} vs {away_team}: "
                        f"p={p_h:.3f}-{p_d:.3f}-{p_a:.3f} → "
                        f"odds={implied['home_odds']:.2f}-{implied['draw_odds']:.2f}-{implied['away_odds']:.2f}"
                    )
            
            except Exception as e:
                if verbose:
                    logger.warning(f"[odds] Failed to insert odds for match {match_id}: {e}")
    
    return {"filled": filled, "attempted": len(upcoming)}
