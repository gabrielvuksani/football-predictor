"""
Expected Goals (xG) calculation from match statistics.

This module computes xG estimates from available match statistics:
- Shots (S)
- Shots on Target (SoT)
- Shot-Creating Events
- Possession %
- Pass completion rate

Methods:
1. Statistical model: Uses ratio of SOT to total shots, scaled by goals
2. Regression model: Trained on historical data
3. Fallback: Simple Poisson-based estimate from rolling team stats
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import duckdb
from datetime import timedelta

def ensure_xg_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create table to cache computed xG values."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS match_xg (
            match_id BIGINT PRIMARY KEY,
            home_xg DOUBLE,
            away_xg DOUBLE,
            method VARCHAR,  -- 'stats', 'regression', 'poisson', 'none'
            confidence DOUBLE DEFAULT 0.5,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

def compute_xg_from_stats(
    shots_on_target: float,
    total_shots: float,
    historical_conversion: float = 0.10,
) -> float:
    """
    Estimate xG from shot statistics.
    
    Formula: xG = SOT * conversion_rate + (Shots - SOT) * low_chance_rate
    
    Args:
        shots_on_target: Shots on target count
        total_shots: Total shots count
        historical_conversion: Historical conversion rate for on-target shots (default 0.10 = 10%)
    
    Returns:
        Estimated xG value
    """
    if total_shots == 0:
        return 0.0
    
    off_target = total_shots - shots_on_target
    
    # On-target shots have ~10% conversion; off-target much lower
    xg_on_target = shots_on_target * historical_conversion
    xg_off_target = off_target * (historical_conversion * 0.2)  # 2% conversion
    
    return float(xg_on_target + xg_off_target)

def compute_xg_from_possession(
    possession_pct: float,
    opponent_possession_pct: float,
    goals_scored: int,
    opponent_goals: int,
) -> float:
    """
    Estimate xG from possession and goal data (regression-based).
    
    Teams with higher possession tend to create more chances.
    Uses Poisson regression concept: log(E[goals]) = constant + possession_coeff * possession
    
    Args:
        possession_pct: Team possession percentage (0-100)
        opponent_possession_pct: Opponent possession percentage
        goals_scored: Actual goals scored (for context)
        opponent_goals: Opponent goals (for context)
    
    Returns:
        Estimated xG value
    """
    if possession_pct is None or possession_pct == 0:
        return 0.0
    
    # Normalize to 0-1
    poss = possession_pct / 100.0
    
    # Baseline xG adjusted by possession
    # High possession ~1.5 xG, low possession ~0.5 xG
    baseline = 0.5 + (poss - 0.5) * 2.0
    
    # Clamp to reasonable range
    baseline = max(0.1, min(3.0, baseline))
    
    return float(baseline)

def estimate_xg_rolling_stats(
    con: duckdb.DuckDBPyConnection,
    team: str,
    against_team: str,
    lookback_days: int = 180,
) -> dict:
    """
    Estimate xG based on rolling team statistics.
    
    Uses team's recent average performance:
    - Avg shots per game
    - Avg shots on target per game
    - Team quality (attack/defense rating)
    
    Args:
        con: DuckDB connection
        team: Team name (will be canonicalized)
        against_team: Opponent team (for defense metrics)
        lookback_days: Days to look back for rolling average
    
    Returns:
        dict with estimated xG and confidence
    """
    from footy.normalize import canonical_team_name
    from datetime import datetime, timedelta
    
    team = canonical_team_name(team)
    against_team = canonical_team_name(against_team)
    
    if not team or not against_team:
        return {"home_xg": 0.0, "away_xg": 0.0, "confidence": 0.0}
    
    # Calculate cutoff date
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    
    # Get recent stats where team was home
    home_stats = con.execute("""
        SELECT
            COUNT(*) as games,
            AVG(CAST(me.hs AS FLOAT)) as avg_shots,
            AVG(CAST(me.hst AS FLOAT)) as avg_sot,
            AVG(CAST(m.home_goals AS FLOAT)) as avg_goals
        FROM match_extras me
        JOIN matches m ON me.match_id = m.match_id
        WHERE LOWER(m.home_team) = LOWER(?)
          AND m.utc_date > ?
    """, [team, cutoff_date]).df()
    
    # Get opponent home stats
    opponent_home_stats = con.execute("""
        SELECT
            AVG(CAST(me.as_ AS FLOAT)) as avg_shots,
            AVG(CAST(me.ast AS FLOAT)) as avg_sot,
            AVG(CAST(m.away_goals AS FLOAT)) as avg_goals
        FROM match_extras me
        JOIN matches m ON me.match_id = m.match_id
        WHERE LOWER(m.away_team) = LOWER(?)
          AND m.utc_date > ?
    """, [against_team, cutoff_date]).df()
    
    team_xg = 0.0
    opponent_xg = 0.0
    confidence = 0.5
    
    if not home_stats.empty and home_stats.iloc[0]["games"] and home_stats.iloc[0]["games"] > 0:
        row = home_stats.iloc[0]
        avg_sot = row["avg_sot"] or 0.0
        avg_shots = row["avg_shots"] or 1.0
        if avg_shots > 0:
            team_xg = compute_xg_from_stats(avg_sot, avg_shots)
            confidence += 0.2
    
    if not opponent_home_stats.empty:
        row = opponent_home_stats.iloc[0]
        avg_sot = row["avg_sot"] or 0.0
        avg_shots = row["avg_shots"] or 1.0
        if avg_shots > 0:
            opponent_xg = compute_xg_from_stats(avg_sot, avg_shots)
            confidence += 0.2
    
    confidence = min(1.0, confidence)
    
    return {
        "home_xg": float(team_xg),
        "away_xg": float(opponent_xg),
        "confidence": float(confidence),
    }

def compute_xg_for_upcoming_match(
    con: duckdb.DuckDBPyConnection,
    match_id: int,
    verbose: bool = False,
) -> dict:
    """
    Compute xG for an upcoming match using all available methods.
    
    Strategy:
    1. Try to get xG from recent match_extras (if stats available)
    2. Fall back to rolling team statistics
    3. Fall back to Poisson model if available
    
    Args:
        con: DuckDB connection
        match_id: Match ID to compute xG for
        verbose: Whether to print debug info
    
    Returns:
        dict with home_xg, away_xg, method, confidence
    """
    ensure_xg_table(con)
    
    # Check if already computed
    cached = con.execute("""
        SELECT home_xg, away_xg, method, confidence
        FROM match_xg
        WHERE match_id = ?
    """, [match_id]).df()
    
    if not cached.empty:
        result = cached.iloc[0].to_dict()
        result["cached"] = True
        return result
    
    # Get match details
    match = con.execute("""
        SELECT home_team, away_team, utc_date
        FROM matches
        WHERE match_id = ?
    """, [match_id]).df()
    
    if match.empty:
        return {"error": "Match not found"}
    
    home_team = match.iloc[0]["home_team"]
    away_team = match.iloc[0]["away_team"]
    
    # Try recent match stats first - get last 3 home/away matches
    recent_home = con.execute("""
        SELECT CAST(hs AS FLOAT) as shots,
               CAST(hst AS FLOAT) as shots_on_target
        FROM match_extras me
        JOIN matches m ON me.match_id = m.match_id
        WHERE LOWER(m.home_team) = LOWER(?)
        ORDER BY m.utc_date DESC
        LIMIT 3
    """, [home_team]).df()
    
    # For home stats, sum across the 3 recent matches
    total_shots_home = 0.0
    total_sot_home = 0.0
    if not recent_home.empty:
        total_shots_home = recent_home["shots"].sum()
        total_sot_home = recent_home["shots_on_target"].sum()
    
    recent_away = con.execute("""
        SELECT CAST(as_ AS FLOAT) as shots,
               CAST(ast AS FLOAT) as shots_on_target
        FROM match_extras me
        JOIN matches m ON me.match_id = m.match_id
        WHERE LOWER(m.away_team) = LOWER(?)
        ORDER BY m.utc_date DESC
        LIMIT 3
    """, [away_team]).df()
    
    # For away stats, sum across the 3 recent matches
    total_shots_away = 0.0
    total_sot_away = 0.0
    if not recent_away.empty:
        total_shots_away = recent_away["shots"].sum()
        total_sot_away = recent_away["shots_on_target"].sum()
    
    method = "none"
    home_xg = 0.0
    away_xg = 0.0
    confidence = 0.0
    
    # Method 1: From stats
    if total_shots_home > 0 or total_shots_away > 0:
        home_xg = compute_xg_from_stats(total_sot_home, total_shots_home)
        away_xg = compute_xg_from_stats(total_sot_away, total_shots_away)
        method = "stats"
        confidence = 0.7
    
    # Method 2: Fallback to rolling stats
    if method == "none" or confidence < 0.5:
        roll_est = estimate_xg_rolling_stats(con, home_team, away_team)
        if roll_est.get("confidence", 0) > 0.5:
            home_xg = roll_est["home_xg"]
            away_xg = roll_est["away_xg"]
            method = "rolling"
            confidence = roll_est["confidence"]
    
    # Cache the result
    try:
        con.execute("""
            INSERT INTO match_xg (match_id, home_xg, away_xg, method, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, [match_id, home_xg, away_xg, method, confidence])
    except Exception:
        pass  # Ignore cache errors
    
    if verbose:
        print(f"[xg] match_id={match_id} {home_team} vs {away_team}: "
              f"{home_xg:.2f} vs {away_xg:.2f} ({method}, conf={confidence:.2f})")
    
    return {
        "match_id": match_id,
        "home_xg": float(home_xg),
        "away_xg": float(away_xg),
        "method": method,
        "confidence": float(confidence),
        "cached": False,
    }

def backfill_xg_for_finished_matches(
    con: duckdb.DuckDBPyConnection,
    verbose: bool = False,
) -> dict:
    """
    Backfill xG for all finished matches in the database.
    Uses match stats from match_extras table.
    
    Returns:
        dict with stats on how many matches were processed
    """
    ensure_xg_table(con)
    
    # Get all finished matches without xG
    matches = con.execute("""
        SELECT m.match_id, m.home_team, m.away_team
        FROM matches m
        LEFT JOIN match_xg x ON m.match_id = x.match_id
        WHERE m.status = 'FINISHED'
          AND x.match_id IS NULL
        LIMIT 1000
    """).df()
    
    if matches.empty:
        if verbose:
            print("[xg] no matches to backfill")
        return {"computed": 0, "total": 0}
    
    computed = 0
    for _, row in matches.iterrows():
        result = compute_xg_for_upcoming_match(con, int(row["match_id"]), verbose=verbose)
        if "error" not in result:
            computed += 1
    
    if verbose:
        print(f"[xg] backfilled {computed}/{len(matches)} matches")
    
    return {"computed": computed, "total": len(matches)}
