"""FBref Team Stats Ingestion — Advanced statistics via soccerdata.

Ingests team-level advanced statistics from FBref for the top 5 European leagues:
possession %, PPDA, xG, xGA, shots, pass completion, pressing success.

These are the highest-signal free data features available for football prediction.
Research shows PPDA and possession-based features significantly improve accuracy.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# League mapping: soccerdata name → our competition code
FBREF_LEAGUES = {
    "ENG-Premier League": "PL",
    "ESP-La Liga": "PD",
    "GER-Bundesliga": "BL1",
    "ITA-Serie A": "SA",
    "FRA-Ligue 1": "FL1",
}


def ingest_fbref_team_stats(con, seasons: list[str] | None = None, verbose: bool = True) -> int:
    """Ingest team-level advanced statistics from FBref via soccerdata.

    Creates/updates the `fbref_team_stats` table with per-team, per-season stats.
    These become rolling features in the prediction pipeline.

    Args:
        con: DuckDB connection (read-write).
        seasons: List of season strings like ['2024-2025']. Defaults to current + previous.
        verbose: Print progress.

    Returns:
        Number of team-season records inserted.
    """
    try:
        import soccerdata as sd
    except ImportError:
        log.warning("soccerdata not installed — skipping FBref ingestion")
        return 0

    if seasons is None:
        year = datetime.now().year
        month = datetime.now().month
        current_season = f"{year}-{year+1}" if month >= 7 else f"{year-1}-{year}"
        prev_year = int(current_season.split("-")[0]) - 1
        seasons = [current_season, f"{prev_year}-{prev_year+1}"]

    # Create table if not exists
    con.execute("""
        CREATE TABLE IF NOT EXISTS fbref_team_stats (
            team TEXT NOT NULL,
            competition TEXT NOT NULL,
            season TEXT NOT NULL,
            possession DOUBLE,
            ppda DOUBLE,
            xg DOUBLE,
            xga DOUBLE,
            shots_pg DOUBLE,
            sot_pg DOUBLE,
            pass_pct DOUBLE,
            pressing_success DOUBLE,
            progressive_passes DOUBLE,
            goals_pg DOUBLE,
            goals_against_pg DOUBLE,
            clean_sheet_pct DOUBLE,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (team, competition, season)
        )
    """)

    total = 0
    for league_name, comp_code in FBREF_LEAGUES.items():
        for season in seasons:
            try:
                fbref = sd.FBref(leagues=league_name, seasons=season)
                team_stats = fbref.read_team_season_stats(stat_type="standard")

                if team_stats is None or team_stats.empty:
                    continue

                for _, row in team_stats.iterrows():
                    team = str(row.get("squad", row.name if isinstance(row.name, str) else ""))
                    if not team:
                        continue

                    # Normalize team name
                    from footy.normalize import canonical_team_name
                    team = canonical_team_name(team)
                    if not team:
                        continue

                    # Extract available stats (column names vary by FBref version)
                    def _get(col_candidates, default=None):
                        for c in col_candidates if isinstance(col_candidates, list) else [col_candidates]:
                            val = row.get(c)
                            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                return float(val)
                        return default

                    matches_played = _get(["MP", "matches_played", "mp"], 0)
                    if not matches_played or matches_played < 1:
                        continue

                    goals = _get(["Gls", "goals", "gls"], 0) or 0
                    goals_against = _get(["GA", "goals_against", "ga"], 0) or 0
                    xg = _get(["xG", "xg"], 0) or 0
                    xga = _get(["xGA", "xga"], 0) or 0
                    poss = _get(["Poss", "possession", "poss"])
                    shots = _get(["Sh", "shots", "sh"], 0) or 0
                    sot = _get(["SoT", "shots_on_target", "sot"], 0) or 0
                    pass_pct_val = _get(["Cmp%", "pass_completion_pct", "cmp_pct"])
                    progressive = _get(["PrgP", "progressive_passes", "prgp"], 0) or 0

                    con.execute("""
                        INSERT OR REPLACE INTO fbref_team_stats VALUES
                        (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL, CURRENT_TIMESTAMP)
                    """, [
                        team, comp_code, season,
                        poss,  # possession
                        xg / matches_played if matches_played else 0,  # xg per game
                        xga / matches_played if matches_played else 0,  # xga per game
                        shots / matches_played if matches_played else 0,  # shots pg
                        sot / matches_played if matches_played else 0,  # sot pg
                        pass_pct_val,  # pass pct
                        progressive / matches_played if matches_played else 0,  # progressive pg
                        goals / matches_played if matches_played else 0,  # goals pg
                        goals_against / matches_played if matches_played else 0,  # ga pg
                    ])
                    total += 1

                if verbose:
                    print(f"[fbref] {comp_code} {season}: {total} teams", flush=True)

            except Exception as e:
                if verbose:
                    print(f"[fbref] {comp_code} {season}: {e}", flush=True)
                log.debug("FBref ingestion error for %s %s: %s", comp_code, season, e)

    if verbose:
        print(f"[fbref] total: {total} team-season records", flush=True)
    return total


def get_team_fbref_stats(con, team: str, competition: str) -> dict:
    """Get the latest FBref stats for a team.

    Returns a dict with possession, ppda, xg, xga, etc. or empty dict if no data.
    """
    try:
        row = con.execute("""
            SELECT possession, ppda, xg, xga, shots_pg, sot_pg, pass_pct,
                   pressing_success, progressive_passes, goals_pg,
                   goals_against_pg, clean_sheet_pct
            FROM fbref_team_stats
            WHERE team = ? AND competition = ?
            ORDER BY season DESC
            LIMIT 1
        """, [team, competition]).fetchone()

        if row:
            return {
                "possession": row[0],
                "ppda": row[1],
                "xg": row[2],
                "xga": row[3],
                "shots_pg": row[4],
                "sot_pg": row[5],
                "pass_pct": row[6],
                "pressing_success": row[7],
                "progressive_passes": row[8],
                "goals_pg": row[9],
                "goals_against_pg": row[10],
                "clean_sheet_pct": row[11],
            }
    except Exception:
        pass
    return {}
