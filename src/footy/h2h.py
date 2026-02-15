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
from footy.normalize import canonical_team_name, get_canonical_id

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
    
    # Recent matches for trend analysis (last N matches per pair)
    con.execute("""
        CREATE TABLE IF NOT EXISTS h2h_recent (
            match_id BIGINT PRIMARY KEY,
            team_a VARCHAR NOT NULL,
            team_b VARCHAR NOT NULL,
            utc_date TIMESTAMP,
            a_goals INT,
            b_goals INT,
            outcome VARCHAR,  -- 'a_win', 'b_win', 'draw'
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

def recompute_h2h_stats(con: duckdb.DuckDBPyConnection, verbose: bool = False) -> dict:
    """
    Recompute all H2H statistics from finished matches in the database.
    Call this after ingesting new historical data.
    
    Returns stats on how many pairs were computed.
    """
    ensure_h2h_tables(con)
    
    # Get all finished matches
    matches = con.execute("""
        SELECT match_id, utc_date, home_team, away_team, home_goals, away_goals
        FROM matches
        WHERE status = 'FINISHED'
          AND home_team IS NOT NULL
          AND away_team IS NOT NULL
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
        ORDER BY utc_date ASC
    """).df()
    
    if matches.empty:
        if verbose:
            print("[h2h] no finished matches to compute from")
        return {"computed_pairs": 0, "total_matches": 0}
    
    # Canonicalize team names
    matches["home_canon"] = matches["home_team"].map(canonical_team_name)
    matches["away_canon"] = matches["away_team"].map(canonical_team_name)
    
    # Filter to matches where both teams are recognized
    matches = matches.dropna(subset=["home_canon", "away_canon"])
    
    if matches.empty:
        if verbose:
            print("[h2h] no matches with recognized team names")
        return {"computed_pairs": 0, "total_matches": 0}
    
    if verbose:
        print(f"[h2h] computing from {len(matches)} finished matches", flush=True)
    
    # Clear existing stats
    con.execute("DELETE FROM h2h_stats")
    con.execute("DELETE FROM h2h_venue_stats")
    con.execute("DELETE FROM h2h_recent")
    
    # Compute any-venue H2H
    pairs_any = {}
    for _, row in matches.iterrows():
        home = row["home_canon"]
        away = row["away_canon"]
        
        # Ensure consistent ordering: (min, max)
        key = tuple(sorted([home, away]))
        team_a, team_b = key
        
        if key not in pairs_any:
            pairs_any[key] = {
                "team_a": team_a,
                "team_b": team_b,
                "total": 0,
                "a_wins": 0,
                "b_wins": 0,
                "draws": 0,
                "a_gf": 0,
                "a_ga": 0,
                "b_gf": 0,
                "b_ga": 0,
            }
        
        p = pairs_any[key]
        p["total"] += 1
        
        if home == team_a:
            p["a_gf"] += row["home_goals"]
            p["a_ga"] += row["away_goals"]
            p["b_gf"] += row["away_goals"]
            p["b_ga"] += row["home_goals"]
            if row["home_goals"] > row["away_goals"]:
                p["a_wins"] += 1
            elif row["away_goals"] > row["home_goals"]:
                p["b_wins"] += 1
            else:
                p["draws"] += 1
        else:
            p["b_gf"] += row["home_goals"]
            p["b_ga"] += row["away_goals"]
            p["a_gf"] += row["away_goals"]
            p["a_ga"] += row["home_goals"]
            if row["home_goals"] > row["away_goals"]:
                p["b_wins"] += 1
            elif row["away_goals"] > row["home_goals"]:
                p["a_wins"] += 1
            else:
                p["draws"] += 1
    
    # Insert any-venue stats
    for (team_a, team_b), stats in pairs_any.items():
        a_avg_gf = stats["a_gf"] / max(1, stats["total"])
        a_avg_ga = stats["a_ga"] / max(1, stats["total"])
        b_avg_gf = stats["b_gf"] / max(1, stats["total"])
        b_avg_ga = stats["b_ga"] / max(1, stats["total"])
        
        con.execute("""
            INSERT INTO h2h_stats (
                team_a, team_b,
                team_a_canonical, team_b_canonical,
                total_matches, team_a_wins, team_b_wins, draws,
                team_a_goals_for, team_a_goals_against,
                team_b_goals_for, team_b_goals_against,
                team_a_avg_goals_for, team_a_avg_goals_against,
                team_b_avg_goals_for, team_b_avg_goals_against
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            team_a, team_b, team_a, team_b,
            stats["total"], stats["a_wins"], stats["b_wins"], stats["draws"],
            stats["a_gf"], stats["a_ga"], stats["b_gf"], stats["b_ga"],
            a_avg_gf, a_avg_ga, b_avg_gf, b_avg_ga,
        ])
    
    # Compute venue-specific H2H
    pairs_venue = {}
    for _, row in matches.iterrows():
        home = row["home_canon"]
        away = row["away_canon"]
        
        key = (home, away)
        if key not in pairs_venue:
            pairs_venue[key] = {
                "home": home,
                "away": away,
                "total": 0,
                "home_wins": 0,
                "away_wins": 0,
                "draws": 0,
                "home_gf": 0,
                "home_ga": 0,
                "away_gf": 0,
                "away_ga": 0,
            }
        
        p = pairs_venue[key]
        p["total"] += 1
        p["home_gf"] += row["home_goals"]
        p["home_ga"] += row["away_goals"]
        p["away_gf"] += row["away_goals"]
        p["away_ga"] += row["home_goals"]
        
        if row["home_goals"] > row["away_goals"]:
            p["home_wins"] += 1
        elif row["away_goals"] > row["home_goals"]:
            p["away_wins"] += 1
        else:
            p["draws"] += 1
    
    # Insert venue-specific stats
    for (home, away), stats in pairs_venue.items():
        h_avg_gf = stats["home_gf"] / max(1, stats["total"])
        h_avg_ga = stats["home_ga"] / max(1, stats["total"])
        a_avg_gf = stats["away_gf"] / max(1, stats["total"])
        a_avg_ga = stats["away_ga"] / max(1, stats["total"])
        
        con.execute("""
            INSERT INTO h2h_venue_stats (
                home_team, away_team,
                home_team_canonical, away_team_canonical,
                total_matches, home_wins, away_wins, draws,
                home_goals_for, home_goals_against,
                away_goals_for, away_goals_against,
                home_avg_goals_for, home_avg_goals_against,
                away_avg_goals_for, away_avg_goals_against
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            home, away, home, away,
            stats["total"], stats["home_wins"], stats["away_wins"], stats["draws"],
            stats["home_gf"], stats["home_ga"], stats["away_gf"], stats["away_ga"],
            h_avg_gf, h_avg_ga, a_avg_gf, a_avg_ga,
        ])
    
    if verbose:
        print(f"[h2h] any-venue pairs: {len(pairs_any)}", flush=True)
        print(f"[h2h] venue-specific pairs: {len(pairs_venue)}", flush=True)
    
    return {
        "computed_pairs_any_venue": len(pairs_any),
        "computed_pairs_venue": len(pairs_venue),
        "total_matches_used": len(matches),
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
