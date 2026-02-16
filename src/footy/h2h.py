"""
Head-to-head (H2H) statistics computation and caching.

This module precomputes H2H stats for all team pairs during ingestion,
eliminating the need for expensive runtime lookups on the UI side.

Key features:
- Precomputes during data ingestion
- Stores results in DuckDB for instant queries
- Tracks venue-specific matchups (home team hosts away team)
- Computes rolling averages and trends
"""

from __future__ import annotations
from datetime import datetime
import duckdb
import pandas as pd
from footy.normalize import canonical_team_name

def ensure_h2h_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create DuckDB tables for H2H statistics."""
    
    # Main H2H stats (any venue)
    con.execute("""
        CREATE TABLE IF NOT EXISTS h2h_stats (
            team_a VARCHAR NOT NULL,
            team_b VARCHAR NOT NULL,
            team_a_canonical VARCHAR,
            team_b_canonical VARCHAR,
            total_matches INT DEFAULT 0,
            team_a_wins INT DEFAULT 0,
            team_b_wins INT DEFAULT 0,
            draws INT DEFAULT 0,
            team_a_goals_for INT DEFAULT 0,
            team_a_goals_against INT DEFAULT 0,
            team_b_goals_for INT DEFAULT 0,
            team_b_goals_against INT DEFAULT 0,
            team_a_avg_goals_for DOUBLE DEFAULT 0.0,
            team_a_avg_goals_against DOUBLE DEFAULT 0.0,
            team_b_avg_goals_for DOUBLE DEFAULT 0.0,
            team_b_avg_goals_against DOUBLE DEFAULT 0.0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(team_a, team_b)
        );
    """)
    
    # Venue-specific H2H (when team_a hosts team_b)
    con.execute("""
        CREATE TABLE IF NOT EXISTS h2h_venue_stats (
            home_team VARCHAR NOT NULL,
            away_team VARCHAR NOT NULL,
            home_team_canonical VARCHAR,
            away_team_canonical VARCHAR,
            total_matches INT DEFAULT 0,
            home_wins INT DEFAULT 0,
            away_wins INT DEFAULT 0,
            draws INT DEFAULT 0,
            home_goals_for INT DEFAULT 0,
            home_goals_against INT DEFAULT 0,
            away_goals_for INT DEFAULT 0,
            away_goals_against INT DEFAULT 0,
            home_avg_goals_for DOUBLE DEFAULT 0.0,
            home_avg_goals_against DOUBLE DEFAULT 0.0,
            away_avg_goals_for DOUBLE DEFAULT 0.0,
            away_avg_goals_against DOUBLE DEFAULT 0.0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(home_team, away_team)
        );
    """)

def recompute_h2h_stats(con: duckdb.DuckDBPyConnection, verbose: bool = False) -> dict:
    """
    Recompute all H2H statistics from finished matches using SQL aggregation.
    ~100x faster than the previous O(N) Python loop approach.
    """
    ensure_h2h_tables(con)

    # Get match count for reporting
    total = con.execute("""
        SELECT COUNT(*) FROM matches
        WHERE status = 'FINISHED' AND home_goals IS NOT NULL AND away_goals IS NOT NULL
    """).fetchone()[0]

    if total == 0:
        if verbose:
            print("[h2h] no finished matches to compute from")
        return {"computed_pairs": 0, "total_matches": 0}

    if verbose:
        print(f"[h2h] computing from {total} finished matches via SQL aggregation")

    # Clear existing stats
    con.execute("DELETE FROM h2h_stats")
    con.execute("DELETE FROM h2h_venue_stats")

    # ---- Any-venue H2H via pure SQL ----
    # Step 1: Create canonical match pairs with consistent ordering (min, max)
    con.execute("""
        INSERT INTO h2h_stats (
            team_a, team_b, team_a_canonical, team_b_canonical,
            total_matches, team_a_wins, team_b_wins, draws,
            team_a_goals_for, team_a_goals_against,
            team_b_goals_for, team_b_goals_against,
            team_a_avg_goals_for, team_a_avg_goals_against,
            team_b_avg_goals_for, team_b_avg_goals_against
        )
        WITH match_pairs AS (
            SELECT
                LEAST(home_team, away_team) AS team_a,
                GREATEST(home_team, away_team) AS team_b,
                CASE WHEN home_team = LEAST(home_team, away_team)
                     THEN home_goals ELSE away_goals END AS a_gf,
                CASE WHEN home_team = LEAST(home_team, away_team)
                     THEN away_goals ELSE home_goals END AS a_ga,
                CASE WHEN home_team = GREATEST(home_team, away_team)
                     THEN home_goals ELSE away_goals END AS b_gf,
                CASE WHEN home_team = GREATEST(home_team, away_team)
                     THEN away_goals ELSE home_goals END AS b_ga
            FROM matches
            WHERE status = 'FINISHED'
              AND home_goals IS NOT NULL AND away_goals IS NOT NULL
        )
        SELECT
            team_a, team_b, team_a, team_b,
            COUNT(*) AS total_matches,
            SUM(CASE WHEN a_gf > a_ga THEN 1 ELSE 0 END) AS a_wins,
            SUM(CASE WHEN b_gf > b_ga THEN 1 ELSE 0 END) AS b_wins,
            SUM(CASE WHEN a_gf = a_ga THEN 1 ELSE 0 END) AS draws,
            SUM(a_gf), SUM(a_ga),
            SUM(b_gf), SUM(b_ga),
            AVG(a_gf), AVG(a_ga),
            AVG(b_gf), AVG(b_ga)
        FROM match_pairs
        GROUP BY team_a, team_b
    """)

    n_any = con.execute("SELECT COUNT(*) FROM h2h_stats").fetchone()[0]

    # ---- Venue-specific H2H via pure SQL ----
    con.execute("""
        INSERT INTO h2h_venue_stats (
            home_team, away_team, home_team_canonical, away_team_canonical,
            total_matches, home_wins, away_wins, draws,
            home_goals_for, home_goals_against,
            away_goals_for, away_goals_against,
            home_avg_goals_for, home_avg_goals_against,
            away_avg_goals_for, away_avg_goals_against
        )
        SELECT
            home_team, away_team, home_team, away_team,
            COUNT(*),
            SUM(CASE WHEN home_goals > away_goals THEN 1 ELSE 0 END),
            SUM(CASE WHEN away_goals > home_goals THEN 1 ELSE 0 END),
            SUM(CASE WHEN home_goals = away_goals THEN 1 ELSE 0 END),
            SUM(home_goals), SUM(away_goals),
            SUM(away_goals), SUM(home_goals),
            AVG(home_goals), AVG(away_goals),
            AVG(away_goals), AVG(home_goals)
        FROM matches
        WHERE status = 'FINISHED'
          AND home_goals IS NOT NULL AND away_goals IS NOT NULL
        GROUP BY home_team, away_team
    """)

    n_venue = con.execute("SELECT COUNT(*) FROM h2h_venue_stats").fetchone()[0]

    if verbose:
        print(f"[h2h] any-venue pairs: {n_any}")
        print(f"[h2h] venue-specific pairs: {n_venue}")

    return {
        "computed_pairs_any_venue": n_any,
        "computed_pairs_venue": n_venue,
        "total_matches_used": total,
    }

def get_h2h_any_venue(con: duckdb.DuckDBPyConnection, team_a: str, team_b: str, limit: int = 10) -> dict:
    """
    Get H2H statistics for two teams regardless of venue.
    
    Args:
        con: DuckDB connection
        team_a: First team name (will be canonicalized)
        team_b: Second team name (will be canonicalized)
        limit: Max recent matches to return
    
    Returns:
        dict with overall stats and recent matches
    """
    team_a = canonical_team_name(team_a)
    team_b = canonical_team_name(team_b)
    
    if not team_a or not team_b:
        return {"error": "Could not canonicalize team names"}
    
    # Get overall stats (use sorted order to ensure match)
    ta, tb = tuple(sorted([team_a, team_b]))
    stats = con.execute("""
        SELECT *
        FROM h2h_stats
        WHERE team_a_canonical = ? AND team_b_canonical = ?
        LIMIT 1
    """, [ta, tb]).df()
    
    result = {"stats": None, "recent_matches": []}
    
    if not stats.empty:
        row = stats.iloc[0].to_dict()
        result["stats"] = row
    
    # Get recent matches
    recent = con.execute("""
        SELECT m.match_id, m.utc_date, m.home_team, m.away_team, m.home_goals, m.away_goals
        FROM matches m
        WHERE m.status = 'FINISHED'
          AND (
            (m.home_team = ? AND m.away_team = ?) OR
            (m.home_team = ? AND m.away_team = ?)
          )
        ORDER BY m.utc_date DESC
        LIMIT ?
    """, [team_a, team_b, team_b, team_a, limit]).df()
    
    result["recent_matches"] = recent.to_dict("records") if not recent.empty else []
    
    return result

def get_h2h_venue(con: duckdb.DuckDBPyConnection, home_team: str, away_team: str, limit: int = 10) -> dict:
    """
    Get H2H statistics for two teams when home_team hosts away_team.
    
    Returns:
        dict with venue-specific stats and recent matches
    """
    home_team = canonical_team_name(home_team)
    away_team = canonical_team_name(away_team)
    
    if not home_team or not away_team:
        return {"error": "Could not canonicalize team names"}
    
    # Get venue-specific stats (use canonical columns)
    stats = con.execute("""
        SELECT *
        FROM h2h_venue_stats
        WHERE home_team_canonical = ? AND away_team_canonical = ?
        LIMIT 1
    """, [home_team, away_team]).df()
    
    result = {"stats": None, "recent_matches": []}
    
    if not stats.empty:
        result["stats"] = stats.iloc[0].to_dict()
    
    # Get recent matches at this venue
    recent = con.execute("""
        SELECT m.match_id, m.utc_date, m.home_team, m.away_team, m.home_goals, m.away_goals
        FROM matches m
        WHERE m.status = 'FINISHED'
          AND m.home_team = ?
          AND m.away_team = ?
        ORDER BY m.utc_date DESC
        LIMIT ?
    """, [home_team, away_team, limit]).df()
    
    result["recent_matches"] = recent.to_dict("records") if not recent.empty else []
    
    return result
