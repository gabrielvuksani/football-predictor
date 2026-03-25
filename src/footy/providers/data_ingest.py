"""Data ingestion pipeline for enrichment data.

Orchestrates FBref stats, Transfermarkt values/injuries, referee stats,
venue stats, and referee extraction from football-data.co.uk CSVs.
Each function operates independently with circuit-breaker isolation
so a single source failure never blocks the rest.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from footy.config import settings
from footy.db import connect
from footy.normalize import canonical_team_name
from footy.providers.fbref_scraper import FBrefProvider, FBREF_LEAGUES
from footy.providers.transfermarkt import TransfermarktProvider, TRANSFERMARKT_LEAGUES
from footy.providers.fdcuk_history import DIV_MAP, download_division_csv

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table DDL — created on first use if they don't exist yet
# ---------------------------------------------------------------------------

_FBREF_TEAM_STATS_DDL = """
CREATE TABLE IF NOT EXISTS fbref_team_stats (
    team VARCHAR NOT NULL,
    competition VARCHAR NOT NULL,
    season VARCHAR NOT NULL,
    -- standard stats
    games INT,
    goals INT,
    assists INT,
    xg DOUBLE,
    npxg DOUBLE,
    xg_assist DOUBLE,
    xg_per90 DOUBLE,
    npxg_per90 DOUBLE,
    goals_per90 DOUBLE,
    assists_per90 DOUBLE,
    cards_yellow INT,
    cards_red INT,
    -- shooting stats
    shots INT,
    shots_on_target INT,
    shots_per90 DOUBLE,
    shots_on_target_per90 DOUBLE,
    shots_on_target_pct DOUBLE,
    goals_per_shot DOUBLE,
    goals_per_shot_on_target DOUBLE,
    average_shot_distance DOUBLE,
    npxg_per_shot DOUBLE,
    -- passing stats
    passes_completed INT,
    passes INT,
    passes_pct DOUBLE,
    passes_progressive INT,
    passes_completed_short INT,
    passes_pct_short DOUBLE,
    passes_completed_medium INT,
    passes_pct_medium DOUBLE,
    passes_completed_long INT,
    passes_pct_long DOUBLE,
    xa DOUBLE,
    -- defense stats
    tackles INT,
    tackles_won INT,
    tackles_won_pct DOUBLE,
    interceptions INT,
    blocks INT,
    clearances INT,
    errors INT,
    source VARCHAR DEFAULT 'fbref',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(team, competition, season)
);
"""

_REFEREE_STATS_DDL = """
CREATE TABLE IF NOT EXISTS referee_stats (
    referee VARCHAR NOT NULL,
    competition VARCHAR,
    total_matches INT DEFAULT 0,
    avg_home_goals DOUBLE,
    avg_away_goals DOUBLE,
    avg_yellow_home DOUBLE,
    avg_yellow_away DOUBLE,
    avg_red_home DOUBLE,
    avg_red_away DOUBLE,
    home_win_pct DOUBLE,
    draw_pct DOUBLE,
    away_win_pct DOUBLE,
    home_bias DOUBLE,
    card_strictness DOUBLE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(referee, competition)
);
"""

_VENUE_STATS_DDL = """
CREATE TABLE IF NOT EXISTS venue_stats (
    team VARCHAR NOT NULL,
    competition VARCHAR,
    home_matches INT DEFAULT 0,
    home_wins INT DEFAULT 0,
    home_draws INT DEFAULT 0,
    home_losses INT DEFAULT 0,
    home_win_pct DOUBLE,
    home_draw_pct DOUBLE,
    home_loss_pct DOUBLE,
    avg_goals_scored DOUBLE,
    avg_goals_conceded DOUBLE,
    clean_sheet_pct DOUBLE,
    btts_pct DOUBLE,
    home_ppg DOUBLE,
    home_advantage_strength DOUBLE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(team, competition)
);
"""


def _ensure_tables(con) -> None:
    """Create enrichment tables if they don't exist."""
    for ddl in (_FBREF_TEAM_STATS_DDL, _REFEREE_STATS_DDL, _VENUE_STATS_DDL):
        try:
            con.execute(ddl)
        except Exception as exc:
            log.debug("DDL execution note: %s", exc)


def _resolve_competitions(competitions: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Fall back to tracked competitions from settings when none specified."""
    if competitions:
        return tuple(competitions)
    return settings().tracked_competitions


# ---------------------------------------------------------------------------
# 1. FBref stats ingestion
# ---------------------------------------------------------------------------

def ingest_fbref_stats(
    competitions: list[str] | tuple[str, ...] | None = None,
    season: str = "2025-2026",
) -> int:
    """Fetch advanced team stats from FBref and store in fbref_team_stats.

    Scrapes four stat categories (standard, shooting, passing, defense) for
    each competition, normalizes team names, then upserts into DuckDB.

    Args:
        competitions: Competition codes to fetch. Defaults to tracked_competitions.
        season: FBref season string, e.g. "2025-2026".

    Returns:
        Number of team-rows upserted.
    """
    s = settings()
    con = connect()
    _ensure_tables(con)

    comps = _resolve_competitions(competitions)
    provider = FBrefProvider(enabled=s.enable_fbref)
    total = 0

    try:
        for comp in comps:
            if comp not in FBREF_LEAGUES:
                log.debug("fbref: no league mapping for %s, skipping", comp)
                continue

            log.info("fbref: fetching stats for %s/%s", comp, season)

            # Fetch all four stat categories with individual error isolation
            standard: list[dict[str, Any]] = []
            shooting: list[dict[str, Any]] = []
            passing: list[dict[str, Any]] = []
            defense: list[dict[str, Any]] = []

            try:
                standard = provider.fetch_team_stats(comp, season)
            except Exception as exc:
                log.warning("fbref: standard stats failed for %s: %s", comp, exc)

            try:
                shooting = provider.fetch_shooting_stats(comp, season)
            except Exception as exc:
                log.warning("fbref: shooting stats failed for %s: %s", comp, exc)

            try:
                passing = provider.fetch_passing_stats(comp, season)
            except Exception as exc:
                log.warning("fbref: passing stats failed for %s: %s", comp, exc)

            try:
                defense = provider.fetch_defense_stats(comp, season)
            except Exception as exc:
                log.warning("fbref: defense stats failed for %s: %s", comp, exc)

            # Index supplementary stats by raw team name for merging
            shooting_by_team = {s["team"]: s for s in shooting}
            passing_by_team = {p["team"]: p for p in passing}
            defense_by_team = {d["team"]: d for d in defense}

            for row in standard:
                raw_team = row.get("team", "")
                team = canonical_team_name(raw_team) or raw_team
                if not team:
                    continue

                # Merge supplementary stats
                shoot = shooting_by_team.get(raw_team, {})
                pass_ = passing_by_team.get(raw_team, {})
                def_ = defense_by_team.get(raw_team, {})

                con.execute(
                    """INSERT OR REPLACE INTO fbref_team_stats
                       (team, competition, season,
                        games, goals, assists, xg, npxg, xg_assist,
                        xg_per90, npxg_per90, goals_per90, assists_per90,
                        cards_yellow, cards_red,
                        shots, shots_on_target, shots_per90, shots_on_target_per90,
                        shots_on_target_pct, goals_per_shot, goals_per_shot_on_target,
                        average_shot_distance, npxg_per_shot,
                        passes_completed, passes, passes_pct, passes_progressive,
                        passes_completed_short, passes_pct_short,
                        passes_completed_medium, passes_pct_medium,
                        passes_completed_long, passes_pct_long, xa,
                        tackles, tackles_won, tackles_won_pct,
                        interceptions, blocks, clearances, errors,
                        source, updated_at)
                       VALUES (?, ?, ?,
                               ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?,
                               ?, ?,
                               ?, ?, ?, ?,
                               ?, ?, ?,
                               ?, ?,
                               ?, ?, ?, ?,
                               ?, ?,
                               ?, ?,
                               ?, ?, ?,
                               ?, ?, ?,
                               ?, ?, ?, ?,
                               'fbref', CURRENT_TIMESTAMP)""",
                    [
                        team, comp, season,
                        # standard
                        row.get("games", 0),
                        row.get("goals", 0),
                        row.get("assists", 0),
                        row.get("xg", 0.0),
                        row.get("npxg", 0.0),
                        row.get("xg_assist", 0.0),
                        row.get("xg_per90", 0.0),
                        row.get("npxg_per90", 0.0),
                        row.get("goals_per90", 0.0),
                        row.get("assists_per90", 0.0),
                        row.get("cards_yellow", 0),
                        row.get("cards_red", 0),
                        # shooting
                        shoot.get("shots", 0),
                        shoot.get("shots_on_target", 0),
                        shoot.get("shots_per90", 0.0),
                        shoot.get("shots_on_target_per90", 0.0),
                        shoot.get("shots_on_target_pct", 0.0),
                        shoot.get("goals_per_shot", 0.0),
                        shoot.get("goals_per_shot_on_target", 0.0),
                        shoot.get("average_shot_distance", 0.0),
                        shoot.get("npxg_per_shot", 0.0),
                        # passing
                        pass_.get("passes_completed", 0),
                        pass_.get("passes", 0),
                        pass_.get("passes_pct", 0.0),
                        pass_.get("passes_progressive", 0),
                        pass_.get("passes_completed_short", 0),
                        pass_.get("passes_pct_short", 0.0),
                        pass_.get("passes_completed_medium", 0),
                        pass_.get("passes_pct_medium", 0.0),
                        pass_.get("passes_completed_long", 0),
                        pass_.get("passes_pct_long", 0.0),
                        pass_.get("xa", 0.0),
                        # defense
                        def_.get("tackles", 0),
                        def_.get("tackles_won", 0),
                        def_.get("tackles_won_pct", 0.0),
                        def_.get("interceptions", 0),
                        def_.get("blocks", 0),
                        def_.get("clearances", 0),
                        def_.get("errors", 0),
                    ],
                )
                total += 1

            log.info("fbref: stored %d teams for %s/%s", len(standard), comp, season)
    finally:
        provider.close()

    log.info("fbref: total %d team-rows upserted across all competitions", total)
    return total


# ---------------------------------------------------------------------------
# 2. Transfermarkt squad values
# ---------------------------------------------------------------------------

def ingest_transfermarkt_values(
    competitions: list[str] | tuple[str, ...] | None = None,
    season: str = "2025",
) -> int:
    """Fetch squad market values from Transfermarkt and store in market_values.

    Args:
        competitions: Competition codes. Defaults to tracked_competitions.
        season: Transfermarkt season year, e.g. "2025".

    Returns:
        Number of team-rows upserted.
    """
    s = settings()
    con = connect()
    comps = _resolve_competitions(competitions)
    provider = TransfermarktProvider(enabled=s.enable_transfermarkt)
    total = 0

    try:
        for comp in comps:
            if comp not in TRANSFERMARKT_LEAGUES:
                log.debug("transfermarkt: no league mapping for %s, skipping", comp)
                continue

            log.info("transfermarkt: fetching squad values for %s/%s", comp, season)

            try:
                values = provider.fetch_squad_values(comp, season)
            except Exception as exc:
                log.warning("transfermarkt: squad values failed for %s: %s", comp, exc)
                continue

            for team_data in values:
                raw_team = team_data.get("team", "")
                team = canonical_team_name(raw_team) or raw_team
                if not team:
                    continue

                squad_value = team_data.get("squad_market_value_eur", 0)
                avg_value = team_data.get("average_player_value_eur", 0)
                squad_size = team_data.get("squad_size", 0)
                avg_age = team_data.get("average_age", 0)

                con.execute(
                    """INSERT OR REPLACE INTO market_values
                       (team, competition, squad_market_value_eur, average_player_value_eur,
                        squad_size, average_age, source, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, 'transfermarkt', CURRENT_TIMESTAMP)""",
                    [team, comp, squad_value, avg_value, squad_size, avg_age],
                )
                total += 1

            log.info("transfermarkt: stored %d teams for %s", len(values), comp)
    finally:
        provider.close()

    log.info("transfermarkt: total %d market value rows upserted", total)
    return total


# ---------------------------------------------------------------------------
# 3. Transfermarkt injuries
# ---------------------------------------------------------------------------

def ingest_transfermarkt_injuries(
    competitions: list[str] | tuple[str, ...] | None = None,
) -> int:
    """Fetch current injury data from Transfermarkt and store in transfermarkt_injuries.

    Args:
        competitions: Competition codes. Defaults to tracked_competitions.

    Returns:
        Number of injury rows upserted.
    """
    s = settings()
    con = connect()
    comps = _resolve_competitions(competitions)
    provider = TransfermarktProvider(enabled=s.enable_transfermarkt)
    total = 0

    try:
        for comp in comps:
            if comp not in TRANSFERMARKT_LEAGUES:
                log.debug("transfermarkt: no league mapping for %s, skipping", comp)
                continue

            log.info("transfermarkt: fetching injuries for %s", comp)

            try:
                injuries = provider.fetch_injuries(comp)
            except Exception as exc:
                log.warning("transfermarkt: injuries failed for %s: %s", comp, exc)
                continue

            for inj in injuries:
                raw_team = inj.get("team", "")
                team = canonical_team_name(raw_team) or raw_team
                player = inj.get("player_name", "")
                if not team or not player:
                    continue

                con.execute(
                    """INSERT OR REPLACE INTO transfermarkt_injuries
                       (team, player_name, injury_type, return_date, source, updated_at)
                       VALUES (?, ?, ?, ?, 'transfermarkt', CURRENT_TIMESTAMP)""",
                    [team, player, inj.get("injury_type", ""), inj.get("return_date", "")],
                )
                total += 1

            log.info("transfermarkt: stored %d injuries for %s", len(injuries), comp)
    finally:
        provider.close()

    log.info("transfermarkt: total %d injury rows upserted", total)
    return total


# ---------------------------------------------------------------------------
# 4. Referee stats (derived from match_extras)
# ---------------------------------------------------------------------------

def compute_referee_stats() -> int:
    """Derive referee performance stats from match_extras and matches tables.

    For each referee with sufficient data, computes average goals, cards,
    outcome percentages, home bias, and card strictness. Stores results
    in the referee_stats table.

    Returns:
        Number of referee-competition rows upserted.
    """
    con = connect()
    _ensure_tables(con)

    # Verify referee_name column exists in match_extras
    try:
        cols = {
            r[0].lower()
            for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='match_extras'"
            ).fetchall()
        }
        if "referee_name" not in cols:
            log.warning("referee_stats: match_extras.referee_name column not found")
            return 0
    except Exception as exc:
        log.warning("referee_stats: cannot inspect match_extras schema: %s", exc)
        return 0

    # Join match_extras (referee + cards) with matches (goals + result)
    rows = con.execute("""
        SELECT
            me.referee_name,
            me.competition,
            COUNT(*)                                           AS total_matches,
            AVG(m.home_goals)                                  AS avg_home_goals,
            AVG(m.away_goals)                                  AS avg_away_goals,
            AVG(COALESCE(me.hy, 0))                            AS avg_yellow_home,
            AVG(COALESCE(me.ay, 0))                            AS avg_yellow_away,
            AVG(COALESCE(me.hr, 0))                            AS avg_red_home,
            AVG(COALESCE(me.ar, 0))                            AS avg_red_away,
            SUM(CASE WHEN m.home_goals > m.away_goals THEN 1 ELSE 0 END) AS home_wins,
            SUM(CASE WHEN m.home_goals = m.away_goals THEN 1 ELSE 0 END) AS draws,
            SUM(CASE WHEN m.home_goals < m.away_goals THEN 1 ELSE 0 END) AS away_wins
        FROM match_extras me
        JOIN matches m ON m.match_id = me.match_id
        WHERE me.referee_name IS NOT NULL
          AND me.referee_name != ''
          AND m.status = 'FINISHED'
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
        GROUP BY me.referee_name, me.competition
        HAVING COUNT(*) >= 3
    """).fetchall()

    total = 0
    for row in rows:
        (ref_name, comp, n_matches,
         avg_hg, avg_ag,
         avg_yh, avg_ya, avg_rh, avg_ra,
         hw, dr, aw) = row

        n = float(n_matches)
        home_win_pct = hw / n
        draw_pct = dr / n
        away_win_pct = aw / n

        # Home bias: how much this ref's home win rate deviates from 50%
        # Positive = favours home, negative = favours away
        home_bias = home_win_pct - away_win_pct

        # Card strictness: total cards per match (yellow + red, both teams)
        card_strictness = (avg_yh + avg_ya + avg_rh + avg_ra)

        con.execute(
            """INSERT OR REPLACE INTO referee_stats
               (referee, competition, total_matches,
                avg_home_goals, avg_away_goals,
                avg_yellow_home, avg_yellow_away,
                avg_red_home, avg_red_away,
                home_win_pct, draw_pct, away_win_pct,
                home_bias, card_strictness, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            [
                ref_name, comp, int(n_matches),
                float(avg_hg), float(avg_ag),
                float(avg_yh), float(avg_ya),
                float(avg_rh), float(avg_ra),
                float(home_win_pct), float(draw_pct), float(away_win_pct),
                float(home_bias), float(card_strictness),
            ],
        )
        total += 1

    log.info("referee_stats: computed stats for %d referee-competition pairs", total)
    return total


# ---------------------------------------------------------------------------
# 5. Venue stats (derived from matches)
# ---------------------------------------------------------------------------

def compute_venue_stats() -> int:
    """Derive home-venue performance stats from the matches table.

    For each home_team per competition, computes W/D/L percentages,
    average goals scored/conceded, clean sheet %, BTTS %, and a
    z-score home_advantage_strength relative to the league average.

    Returns:
        Number of team-competition rows upserted.
    """
    con = connect()
    _ensure_tables(con)

    # Step 1: Per-team home stats
    rows = con.execute("""
        SELECT
            home_team,
            competition,
            COUNT(*)                                             AS home_matches,
            SUM(CASE WHEN home_goals > away_goals THEN 1 ELSE 0 END)  AS home_wins,
            SUM(CASE WHEN home_goals = away_goals THEN 1 ELSE 0 END)  AS home_draws,
            SUM(CASE WHEN home_goals < away_goals THEN 1 ELSE 0 END)  AS home_losses,
            AVG(home_goals)                                      AS avg_goals_scored,
            AVG(away_goals)                                      AS avg_goals_conceded,
            SUM(CASE WHEN away_goals = 0 THEN 1 ELSE 0 END)     AS clean_sheets,
            SUM(CASE WHEN home_goals > 0 AND away_goals > 0 THEN 1 ELSE 0 END) AS btts_count
        FROM matches
        WHERE status = 'FINISHED'
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
          AND home_team IS NOT NULL
          AND competition IS NOT NULL
        GROUP BY home_team, competition
        HAVING COUNT(*) >= 3
    """).fetchall()

    if not rows:
        log.info("venue_stats: no qualifying home data found")
        return 0

    # Step 2: Compute per-team PPG and league-level stats for z-score
    team_stats: list[dict[str, Any]] = []
    for row in rows:
        (team, comp, n_matches, hw, hd, hl,
         avg_gs, avg_gc, cs, btts_count) = row
        n = float(n_matches)
        ppg = (hw * 3 + hd * 1) / n
        team_stats.append({
            "team": team,
            "competition": comp,
            "home_matches": int(n_matches),
            "home_wins": int(hw),
            "home_draws": int(hd),
            "home_losses": int(hl),
            "home_win_pct": hw / n,
            "home_draw_pct": hd / n,
            "home_loss_pct": hl / n,
            "avg_goals_scored": float(avg_gs),
            "avg_goals_conceded": float(avg_gc),
            "clean_sheet_pct": cs / n,
            "btts_pct": btts_count / n,
            "home_ppg": ppg,
        })

    # Step 3: Per-competition league average PPG and stddev for z-scores
    from collections import defaultdict
    comp_ppgs: dict[str, list[float]] = defaultdict(list)
    for ts in team_stats:
        comp_ppgs[ts["competition"]].append(ts["home_ppg"])

    comp_avg: dict[str, float] = {}
    comp_std: dict[str, float] = {}
    for comp, ppgs in comp_ppgs.items():
        avg = sum(ppgs) / len(ppgs) if ppgs else 0.0
        variance = sum((p - avg) ** 2 for p in ppgs) / len(ppgs) if len(ppgs) > 1 else 1.0
        std = max(variance ** 0.5, 0.01)  # floor to prevent division by zero
        comp_avg[comp] = avg
        comp_std[comp] = std

    # Step 4: Upsert
    total = 0
    for ts in team_stats:
        comp = ts["competition"]
        advantage = (ts["home_ppg"] - comp_avg.get(comp, 0.0)) / comp_std.get(comp, 1.0)

        con.execute(
            """INSERT OR REPLACE INTO venue_stats
               (team, competition, home_matches,
                home_win_pct, home_draw_pct, home_loss_pct,
                avg_home_scored, avg_home_conceded,
                home_clean_sheet_pct, home_btts_pct,
                home_advantage_strength, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            [
                ts["team"], comp, ts["home_matches"],
                ts["home_win_pct"], ts["home_draw_pct"], ts["home_loss_pct"],
                ts["avg_goals_scored"], ts["avg_goals_conceded"],
                ts["clean_sheet_pct"], ts["btts_pct"],
                float(advantage),
            ],
        )
        total += 1

    log.info("venue_stats: computed stats for %d team-competition pairs", total)
    return total


# ---------------------------------------------------------------------------
# 6. Extract referee names from football-data.co.uk CSVs
# ---------------------------------------------------------------------------

def extract_referee_from_fdcuk() -> int:
    """Backfill referee_name in match_extras from football-data.co.uk CSVs.

    Finds match_extras rows where referee_name is NULL, re-downloads the
    original CSV for that season/division, extracts the Referee column,
    and updates the rows.

    Returns:
        Number of rows updated.
    """
    con = connect()

    # Verify column exists
    try:
        cols = {
            r[0].lower()
            for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='match_extras'"
            ).fetchall()
        }
        if "referee_name" not in cols:
            log.warning("extract_referee: match_extras.referee_name column not found")
            return 0
    except Exception as exc:
        log.warning("extract_referee: cannot inspect match_extras schema: %s", exc)
        return 0

    # Find distinct season_code/div_code combos that have NULL referee_name
    combos = con.execute("""
        SELECT DISTINCT season_code, div_code
        FROM match_extras
        WHERE (referee_name IS NULL OR referee_name = '')
          AND provider = 'football-data.co.uk'
          AND season_code IS NOT NULL
          AND div_code IS NOT NULL
    """).fetchall()

    if not combos:
        log.info("extract_referee: no NULL referee rows to backfill")
        return 0

    total_updated = 0

    for season_code, div_code in combos:
        log.info("extract_referee: downloading %s/%s.csv for referee data", season_code, div_code)

        try:
            df = download_division_csv(str(season_code), str(div_code))
        except Exception as exc:
            log.warning("extract_referee: failed to download %s/%s: %s", season_code, div_code, exc)
            continue

        if "Referee" not in df.columns:
            log.debug("extract_referee: no Referee column in %s/%s", season_code, div_code)
            continue

        # Find the competition code for this div
        comp = None
        for c, d in DIV_MAP.items():
            if d.div == div_code:
                comp = c
                break

        # Build a lookup: (date, home, away) -> referee
        need_cols = ["Date", "HomeTeam", "AwayTeam", "Referee"]
        if not all(c in df.columns for c in need_cols):
            log.debug("extract_referee: missing required columns in %s/%s", season_code, div_code)
            continue

        import pandas as pd

        for _, row in df.iterrows():
            referee = row.get("Referee")
            if not referee or (isinstance(referee, float) and pd.isna(referee)):
                continue
            referee = str(referee).strip()
            if not referee:
                continue

            home = canonical_team_name(str(row.get("HomeTeam", "")))
            away = canonical_team_name(str(row.get("AwayTeam", "")))
            if not home or not away:
                continue

            # Parse date
            date_str = str(row.get("Date", ""))
            try:
                dt = pd.to_datetime(date_str, dayfirst=True, format="mixed")
            except Exception:
                continue

            # Reconstruct the match_id the same way pipeline.py does
            key = f"fdcuk|{season_code}|{comp or ''}|{dt.date()}|{home}|{away}"
            h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
            match_id = int.from_bytes(h, byteorder="big", signed=False) & 0x7FFFFFFFFFFFFFFF
            if match_id == 0:
                match_id = 1

            result = con.execute(
                """UPDATE match_extras
                   SET referee_name = ?
                   WHERE match_id = ?
                     AND (referee_name IS NULL OR referee_name = '')""",
                [referee, match_id],
            )
            # DuckDB's execute doesn't return rowcount directly for UPDATE;
            # count by checking if the update matched
            try:
                result.fetchone()
            except Exception:
                pass
            total_updated += 1

    log.info("extract_referee: attempted update on %d rows across %d CSV files", total_updated, len(combos))
    return total_updated


def _backfill_understat_xg() -> int:
    """Backfill Understat xG for recent finished matches missing it.

    Uses the DataOrchestrator's Understat provider to fetch match-level xG
    for the 6 Understat-covered leagues (PL, PD, BL1, SA, FL1, RU1).
    Only processes matches from the last 60 days that don't already have Understat xG.
    """
    con = connect()

    # Find recent finished matches without Understat xG in covered leagues
    understat_leagues = ('PL', 'PD', 'BL1', 'SA', 'FL1')
    placeholders = ','.join(['?'] * len(understat_leagues))
    rows = con.execute(f"""
        SELECT m.match_id
        FROM matches m
        LEFT JOIN match_extras e ON e.match_id = m.match_id
        WHERE m.status = 'FINISHED'
          AND m.competition IN ({placeholders})
          AND m.utc_date >= CURRENT_TIMESTAMP - INTERVAL 60 DAY
          AND (e.understat_xg_home IS NULL OR e.understat_xg_home = 0)
        ORDER BY m.utc_date DESC
        LIMIT 200
    """, list(understat_leagues)).fetchall()

    if not rows:
        log.info("understat: no matches need xG backfill")
        return 0

    match_ids = [r[0] for r in rows]
    log.info("understat: backfilling xG for %d matches", len(match_ids))

    from footy.data_orchestrator import DataOrchestrator
    orchestrator = DataOrchestrator()
    try:
        updated = orchestrator.backfill_understat_xg(match_ids)
    finally:
        orchestrator.close()

    log.info("understat: updated xG for %d matches", updated)
    return updated


# ---------------------------------------------------------------------------
# 8. Understat team match stats (PPDA, xPTS, deep completions)
# ---------------------------------------------------------------------------

_UNDERSTAT_MATCH_STATS_DDL = """
CREATE TABLE IF NOT EXISTS understat_match_stats (
    match_id BIGINT NOT NULL PRIMARY KEY,
    home_team VARCHAR,
    away_team VARCHAR,
    home_xg DOUBLE,
    away_xg DOUBLE,
    home_npxg DOUBLE,
    away_npxg DOUBLE,
    home_ppda DOUBLE,
    away_ppda DOUBLE,
    home_deep DOUBLE,
    away_deep DOUBLE,
    home_xpts DOUBLE,
    away_xpts DOUBLE,
    season VARCHAR,
    competition VARCHAR,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Reverse map: soccerdata league names -> our competition codes
_SD_LEAGUE_TO_COMP = {
    "ENG-Premier League": "PL",
    "ESP-La Liga": "PD",
    "GER-Bundesliga": "BL1",
    "ITA-Serie A": "SA",
    "FRA-Ligue 1": "FL1",
}


def ingest_understat_match_stats(seasons: list[str] | None = None) -> int:
    """Fetch Understat team match stats (PPDA, xPTS, deep completions) via soccerdata.

    Covers 5 leagues: PL, PD, BL1, SA, FL1.
    PPDA = Passes Per Defensive Action (pressing intensity).
    xPTS = Expected points based on xG.
    Deep completions = passes completed within 20m of goal.

    Returns number of rows upserted.
    """
    try:
        import soccerdata as sd
    except ImportError:
        log.warning("soccerdata not installed, skipping Understat match stats")
        return 0

    con = connect()
    con.execute(_UNDERSTAT_MATCH_STATS_DDL)

    seasons = seasons or ["2024-2025", "2023-2024"]
    leagues = list(_SD_LEAGUE_TO_COMP.keys())

    total = 0
    try:
        us = sd.Understat(leagues=leagues, seasons=seasons)
        df = us.read_team_match_stats()
    except Exception as exc:
        log.warning("understat match stats fetch failed: %s", exc)
        return 0

    if df.empty:
        return 0

    # Reset MultiIndex (league/season/game) to flat columns
    df = df.reset_index()

    for _, row in df.iterrows():
        try:
            # 'league' column comes from the reset MultiIndex
            league = str(row.get("league", ""))
            comp = _SD_LEAGUE_TO_COMP.get(league, "")
            game_id = row.get("game_id")
            if not game_id:
                continue

            home_team = canonical_team_name(str(row.get("home_team", "")))
            away_team = canonical_team_name(str(row.get("away_team", "")))

            con.execute(
                """INSERT OR REPLACE INTO understat_match_stats
                   (match_id, home_team, away_team,
                    home_xg, away_xg, home_npxg, away_npxg,
                    home_ppda, away_ppda,
                    home_deep, away_deep,
                    home_xpts, away_xpts,
                    season, competition, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                [
                    int(game_id), home_team, away_team,
                    float(row.get("home_xg", 0) or 0),
                    float(row.get("away_xg", 0) or 0),
                    float(row.get("home_np_xg", 0) or 0),
                    float(row.get("away_np_xg", 0) or 0),
                    float(row.get("home_ppda", 0) or 0),
                    float(row.get("away_ppda", 0) or 0),
                    float(row.get("home_deep_completions", 0) or 0),
                    float(row.get("away_deep_completions", 0) or 0),
                    float(row.get("home_expected_points", 0) or 0),
                    float(row.get("away_expected_points", 0) or 0),
                    str(row.get("season_id", "")), comp,
                ],
            )
            total += 1
        except Exception as exc:
            log.debug("understat match stats insert failed: %s", exc)

    log.info("understat_match_stats: upserted %d rows across %d leagues", total, len(leagues))
    return total


# ---------------------------------------------------------------------------
# 9. Master refresh
# ---------------------------------------------------------------------------

def refresh_all_enrichment(verbose: bool = True) -> dict[str, Any]:
    """Run all enrichment ingestion and derivation steps.

    Uses circuit-breaker isolation: if any single source fails, the others
    continue uninterrupted. Returns a summary dict with counts and errors.

    Args:
        verbose: If True, emit INFO-level progress logs.

    Returns:
        Summary dict mapping step names to row counts or error messages.
    """
    summary: dict[str, Any] = {}
    start = time.monotonic()

    # FBref stats
    try:
        if verbose:
            log.info("=== Enrichment: FBref stats ===")
        summary["fbref_stats"] = ingest_fbref_stats()
    except Exception as exc:
        log.error("enrichment: fbref_stats failed: %s", exc)
        summary["fbref_stats"] = {"error": str(exc)}

    # Transfermarkt values
    try:
        if verbose:
            log.info("=== Enrichment: Transfermarkt values ===")
        summary["transfermarkt_values"] = ingest_transfermarkt_values()
    except Exception as exc:
        log.error("enrichment: transfermarkt_values failed: %s", exc)
        summary["transfermarkt_values"] = {"error": str(exc)}

    # Transfermarkt injuries
    try:
        if verbose:
            log.info("=== Enrichment: Transfermarkt injuries ===")
        summary["transfermarkt_injuries"] = ingest_transfermarkt_injuries()
    except Exception as exc:
        log.error("enrichment: transfermarkt_injuries failed: %s", exc)
        summary["transfermarkt_injuries"] = {"error": str(exc)}

    # Referee extraction from fdcuk CSVs
    try:
        if verbose:
            log.info("=== Enrichment: Referee extraction (fdcuk) ===")
        summary["referee_extraction"] = extract_referee_from_fdcuk()
    except Exception as exc:
        log.error("enrichment: referee_extraction failed: %s", exc)
        summary["referee_extraction"] = {"error": str(exc)}

    # Derived: referee stats
    try:
        if verbose:
            log.info("=== Enrichment: Referee stats ===")
        summary["referee_stats"] = compute_referee_stats()
    except Exception as exc:
        log.error("enrichment: referee_stats failed: %s", exc)
        summary["referee_stats"] = {"error": str(exc)}

    # Derived: venue stats
    try:
        if verbose:
            log.info("=== Enrichment: Venue stats ===")
        summary["venue_stats"] = compute_venue_stats()
    except Exception as exc:
        log.error("enrichment: venue_stats failed: %s", exc)
        summary["venue_stats"] = {"error": str(exc)}

    # Understat xG backfill (6 leagues: PL, PD, BL1, SA, FL1)
    try:
        if verbose:
            log.info("=== Enrichment: Understat xG ===")
        summary["understat_xg"] = _backfill_understat_xg()
    except Exception as exc:
        log.error("enrichment: understat_xg failed: %s", exc)
        summary["understat_xg"] = {"error": str(exc)}

    # Understat match stats (PPDA, xPTS, deep completions)
    try:
        if verbose:
            log.info("=== Enrichment: Understat PPDA/xPTS ===")
        summary["understat_ppda"] = ingest_understat_match_stats()
    except Exception as exc:
        log.error("enrichment: understat_ppda failed: %s", exc)
        summary["understat_ppda"] = {"error": str(exc)}

    elapsed = time.monotonic() - start

    if verbose:
        successes = sum(1 for v in summary.values() if isinstance(v, int))
        failures = sum(1 for v in summary.values() if isinstance(v, dict) and "error" in v)
        log.info(
            "enrichment complete in %.1fs — %d/%d steps succeeded, %d failed",
            elapsed, successes, successes + failures, failures,
        )
        for step, result in summary.items():
            if isinstance(result, int):
                log.info("  %s: %d rows", step, result)
            else:
                log.warning("  %s: FAILED — %s", step, result.get("error", "unknown"))

    summary["_elapsed_seconds"] = round(elapsed, 2)
    return summary
