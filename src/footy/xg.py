"""
Expected Goals (xG) calculation from match statistics.

This module computes xG estimates from available match statistics:
- Shots (S)
- Shots on Target (SoT)
- Shot-Creating Events

Methods:
1. Statistical model: Uses ratio of SOT to total shots with learned conversion rates
2. Advanced model: Learned rates + opponent defence quality + recent form
3. Rolling stats: Simple fallback using rolling team averages
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
    off_target_conversion: float = 0.02,
) -> float:
    """
    Estimate xG from shot statistics.
    
    Formula: xG = SOT * conversion_rate + (Shots - SOT) * off_target_rate
    
    Args:
        shots_on_target: Shots on target count
        total_shots: Total shots count
        historical_conversion: Conversion rate for on-target shots (default 0.10 = 10%)
        off_target_conversion: Conversion rate for off-target shots (default 0.02 = 2%)
    
    Returns:
        Estimated xG value
    """
    if total_shots == 0:
        return 0.0
    
    off_target = total_shots - shots_on_target
    
    xg_on_target = shots_on_target * historical_conversion
    xg_off_target = off_target * off_target_conversion
    
    return float(xg_on_target + xg_off_target)


def learn_conversion_rates(con: duckdb.DuckDBPyConnection) -> dict:
    """
    Learn shot conversion rates from historical match data.

    Queries match_extras for finished matches with shot and goal data,
    computing per-league and overall on-target and off-target conversion rates.

    Returns:
        dict with 'overall' rates and per-league rates keyed by competition code.
        Example::

            {
                "overall": {"on_target": 0.11, "off_target": 0.02, "n_matches": 500},
                "PL": {"on_target": 0.12, "off_target": 0.025, "n_matches": 120},
                ...
            }
    """
    rows = con.execute("""
        SELECT m.competition,
               SUM(CAST(m.home_goals AS FLOAT) + CAST(m.away_goals AS FLOAT)) AS total_goals,
               SUM(CAST(me.hst AS FLOAT) + CAST(me.ast AS FLOAT)) AS total_sot,
               SUM(CAST(me.hs AS FLOAT) + CAST(me.as_ AS FLOAT)) AS total_shots,
               COUNT(*) AS n_matches
        FROM match_extras me
        JOIN matches m ON me.match_id = m.match_id
        WHERE m.status = 'FINISHED'
          AND m.home_goals IS NOT NULL
          AND me.hs IS NOT NULL AND me.hst IS NOT NULL
          AND me.as_ IS NOT NULL AND me.ast IS NOT NULL
        GROUP BY m.competition
    """).fetchall()

    rates: dict = {}
    agg_goals, agg_sot, agg_shots, agg_n = 0.0, 0.0, 0.0, 0

    for comp, goals, sot, shots, n_matches in rows:
        if not sot or sot == 0 or not shots or shots == 0:
            continue
        off_target = shots - sot
        on_target_rate = goals / sot if sot > 0 else 0.10
        off_target_rate = max(0, (goals - sot * on_target_rate)) / off_target if off_target > 0 else 0.0
        # Clamp off-target to a sane range (can be noisy with small data)
        off_target_rate = max(0.005, min(0.05, off_target_rate)) if off_target > 0 else 0.02

        rates[comp] = {
            "on_target": round(float(on_target_rate), 4),
            "off_target": round(float(off_target_rate), 4),
            "n_matches": int(n_matches),
        }
        agg_goals += goals
        agg_sot += sot
        agg_shots += shots
        agg_n += int(n_matches)

    # Overall fallback
    if agg_sot > 0:
        overall_on = agg_goals / agg_sot
        overall_off_target = agg_shots - agg_sot
        overall_off = max(0.005, min(0.05,
            max(0, (agg_goals - agg_sot * overall_on)) / overall_off_target
        )) if overall_off_target > 0 else 0.02
    else:
        overall_on, overall_off = 0.10, 0.02

    rates["overall"] = {
        "on_target": round(float(overall_on), 4),
        "off_target": round(float(overall_off), 4),
        "n_matches": agg_n,
    }
    return rates


def compute_xg_advanced(
    con: duckdb.DuckDBPyConnection,
    home_team: str,
    away_team: str,
    competition: str | None = None,
    lookback_days: int = 180,
) -> dict:
    """
    Advanced xG estimation using learned conversion rates, opponent defensive
    quality, and recent shot-quality form.

    Steps:
        1. Learn conversion rates from the database (per-league when possible).
        2. Adjust for opponent defensive quality (avg goals conceded).
        3. Weight recent form (last 5 matches shot quality).
        4. Return xG estimates with a confidence score.

    Args:
        con: DuckDB connection
        home_team: Home team name (will be canonicalized)
        away_team: Away team name (will be canonicalized)
        competition: Optional competition code for league-specific rates
        lookback_days: Days to look back for rolling averages

    Returns:
        dict with home_xg, away_xg, method, confidence, details.
    """
    from footy.normalize import canonical_team_name
    from datetime import datetime

    home_team = canonical_team_name(home_team)
    away_team = canonical_team_name(away_team)

    if not home_team or not away_team:
        return {"home_xg": 0.0, "away_xg": 0.0, "method": "advanced",
                "confidence": 0.0, "details": "teams not resolved"}

    cutoff = datetime.now() - timedelta(days=lookback_days)

    # --- 1. Learned conversion rates ---
    try:
        rates = learn_conversion_rates(con)
    except Exception:
        rates = {"overall": {"on_target": 0.10, "off_target": 0.02, "n_matches": 0}}

    league_rates = rates.get(competition, rates["overall"])
    on_conv = league_rates["on_target"]
    off_conv = league_rates["off_target"]

    confidence = 0.3  # base
    if league_rates["n_matches"] >= 50:
        confidence += 0.15

    # --- 2. Recent form: last 5 matches shot quality per team ---
    def _team_shot_form(team: str, is_home: bool, n: int = 5) -> dict:
        if is_home:
            q = """
                SELECT CAST(me.hs AS FLOAT) AS shots,
                       CAST(me.hst AS FLOAT) AS sot,
                       CAST(m.home_goals AS FLOAT) AS goals
                FROM match_extras me
                JOIN matches m ON me.match_id = m.match_id
                WHERE LOWER(m.home_team) = LOWER(?)
                  AND m.status = 'FINISHED' AND m.utc_date > ?
                  AND me.hs IS NOT NULL AND me.hst IS NOT NULL
                ORDER BY m.utc_date DESC LIMIT ?
            """
        else:
            q = """
                SELECT CAST(me.as_ AS FLOAT) AS shots,
                       CAST(me.ast AS FLOAT) AS sot,
                       CAST(m.away_goals AS FLOAT) AS goals
                FROM match_extras me
                JOIN matches m ON me.match_id = m.match_id
                WHERE LOWER(m.away_team) = LOWER(?)
                  AND m.status = 'FINISHED' AND m.utc_date > ?
                  AND me.as_ IS NOT NULL AND me.ast IS NOT NULL
                ORDER BY m.utc_date DESC LIMIT ?
            """
        df = con.execute(q, [team, cutoff, n]).df()
        if df.empty:
            return {"avg_shots": 0, "avg_sot": 0, "avg_goals": 0, "n": 0}
        return {
            "avg_shots": float(df["shots"].mean()),
            "avg_sot": float(df["sot"].mean()),
            "avg_goals": float(df["goals"].mean()),
            "n": len(df),
        }

    home_form = _team_shot_form(home_team, is_home=True)
    away_form = _team_shot_form(away_team, is_home=False)

    if home_form["n"] >= 3:
        confidence += 0.15
    if away_form["n"] >= 3:
        confidence += 0.15

    # --- 3. Opponent defensive quality (avg goals conceded) ---
    def _defensive_quality(team: str) -> float:
        """Return average goals conceded per game (lower = better defence)."""
        row = con.execute("""
            SELECT AVG(goals_conceded) AS avg_conceded, COUNT(*) AS n FROM (
                SELECT CAST(m.away_goals AS FLOAT) AS goals_conceded
                FROM matches m WHERE LOWER(m.home_team) = LOWER(?)
                  AND m.status = 'FINISHED' AND m.utc_date > ?
                UNION ALL
                SELECT CAST(m.home_goals AS FLOAT) AS goals_conceded
                FROM matches m WHERE LOWER(m.away_team) = LOWER(?)
                  AND m.status = 'FINISHED' AND m.utc_date > ?
            )
        """, [team, cutoff, team, cutoff]).fetchone()
        if row and row[0] is not None and row[1] >= 3:
            return float(row[0])
        return 1.3  # league average fallback

    away_def_quality = _defensive_quality(away_team)  # opponent of home
    home_def_quality = _defensive_quality(home_team)  # opponent of away

    # Defensive adjustment factor: ratio vs league-average conceded (~1.3)
    league_avg_conceded = 1.3
    home_def_adj = away_def_quality / league_avg_conceded  # >1 = weaker defence
    away_def_adj = home_def_quality / league_avg_conceded

    # --- 4. Compute xG ---
    if home_form["n"] > 0:
        raw_home_xg = compute_xg_from_stats(
            home_form["avg_sot"], home_form["avg_shots"], on_conv, off_conv
        )
        home_xg = raw_home_xg * home_def_adj
    else:
        home_xg = 0.0

    if away_form["n"] > 0:
        raw_away_xg = compute_xg_from_stats(
            away_form["avg_sot"], away_form["avg_shots"], on_conv, off_conv
        )
        away_xg = raw_away_xg * away_def_adj
    else:
        away_xg = 0.0

    confidence = min(1.0, confidence)

    return {
        "home_xg": round(float(home_xg), 3),
        "away_xg": round(float(away_xg), 3),
        "method": "advanced",
        "confidence": round(float(confidence), 3),
        "details": {
            "conversion_rates": {"on_target": on_conv, "off_target": off_conv,
                                 "source": competition or "overall"},
            "home_form": home_form,
            "away_form": away_form,
            "defensive_adj": {"home_opponent": round(home_def_adj, 3),
                              "away_opponent": round(away_def_adj, 3)},
        },
    }


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
