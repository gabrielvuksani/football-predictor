"""
TheSportsDB integration — team metadata, venue info, badges, and event details.

Free API key (dev/test): 3
Provides enrichment data not available from other sources:
    - Team metadata (stadium, capacity, founded year, etc.)
    - Past/next events per team
    - Event details with venue info

Caching strategy:
    Team metadata is cached in DuckDB (thesportsdb_cache table) with a 30-day TTL.
    On first run, data is fetched from the API and stored in the cache.
    On subsequent runs, cached entries younger than 30 days are used directly,
    skipping API calls entirely. This saves 30+ minutes on typical pipeline runs.
"""
from __future__ import annotations

import json as _json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from footy.providers.ratelimit import RateLimiter, TRANSIENT_ERRORS
from footy.normalize import canonical_team_name

log = logging.getLogger(__name__)

_rl = RateLimiter(max_calls=8, period_seconds=60)

# Cache TTL: only re-fetch team data if older than this
CACHE_TTL_DAYS = 30


def _api_key() -> str:
    return os.getenv("THESPORTSDB_KEY", "3").strip() or "3"


def _base() -> str:
    return f"https://www.thesportsdb.com/api/v1/json/{_api_key()}"


def _get(path: str) -> dict | None:
    """GET request with rate limiting and retries."""
    _rl.wait()
    url = f"{_base()}/{path}"
    for attempt in range(3):
        try:
            with httpx.Client(timeout=15.0, headers={
                "User-Agent": "footy-predictor/0.1",
            }) as c:
                r = c.get(url)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                log.debug("TheSportsDB %d: %s", r.status_code, path)
                return None
            return r.json()
        except TRANSIENT_ERRORS as e:
            log.warning("TheSportsDB error (attempt %d): %s", attempt + 1, e)
            time.sleep(2)
    return None


# ---- In-memory cache (per-process, same as before) ----
_team_cache: dict[str, dict] = {}
_league_cache: dict[str, int] = {}

# TheSportsDB league IDs for our tracked competitions
LEAGUE_IDS: dict[str, int] = {
    "PL": 4328,
    "PD": 4335,
    "SA": 4332,
    "BL1": 4331,
    "FL1": 4334,
}


def _get_db_connection():
    """Lazily get the DuckDB connection for persistent caching."""
    try:
        from footy.db import connect
        return connect()
    except Exception as e:
        log.debug("TheSportsDB: could not get DB connection for caching: %s", e)
        return None


def _read_cache(team_name: str) -> dict | None:
    """Read team data from DuckDB cache if it exists and is fresh (< 30 days old)."""
    con = _get_db_connection()
    if con is None:
        return None
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)).isoformat()
        row = con.execute(
            "SELECT response_json, fetched_at FROM thesportsdb_cache "
            "WHERE team_name = ? AND fetched_at > ?",
            [team_name, cutoff],
        ).fetchone()
        if row and row[0]:
            return _json.loads(row[0])
    except Exception as e:
        # Table might not exist yet on very first run; that is fine
        log.debug("TheSportsDB cache read for '%s': %s", team_name, e)
    return None


def _write_cache(team_name: str, result: dict) -> None:
    """Write team data to DuckDB cache (upsert)."""
    con = _get_db_connection()
    if con is None:
        return
    try:
        now = datetime.now(timezone.utc).isoformat()
        con.execute(
            "INSERT INTO thesportsdb_cache (team_name, response_json, fetched_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT (team_name) DO UPDATE SET "
            "response_json = excluded.response_json, fetched_at = excluded.fetched_at",
            [team_name, _json.dumps(result), now],
        )
    except Exception as e:
        log.debug("TheSportsDB cache write for '%s': %s", team_name, e)


def search_team(name: str) -> dict | None:
    """
    Search for a team by name and return metadata.

    Uses a three-tier cache:
      1. In-memory dict (_team_cache) -- instant, lost on process restart
      2. DuckDB thesportsdb_cache table -- persistent, 30-day TTL
      3. Live API call -- only when both caches miss or are stale

    Returns dict with:
        team_id, name, short_name, stadium, capacity, founded,
        badge_url, country, league, description
    """
    canon = canonical_team_name(name)

    # Tier 1: in-memory cache (survives within a single process)
    if canon in _team_cache:
        return _team_cache[canon]

    # Tier 2: DuckDB persistent cache (survives across pipeline runs)
    if canon:
        cached = _read_cache(canon)
        if cached is not None:
            _team_cache[canon] = cached
            log.debug("TheSportsDB: cache hit for '%s'", canon)
            return cached

    # Tier 3: live API call
    data = _get(f"searchteams.php?t={canon or name}")
    if not data or not data.get("teams"):
        return None

    team = data["teams"][0]
    result = {
        "team_id": team.get("idTeam"),
        "name": team.get("strTeam"),
        "short_name": team.get("strTeamShort"),
        "stadium": team.get("strStadium"),
        "capacity": _safe_int(team.get("intStadiumCapacity")),
        "founded": _safe_int(team.get("intFormedYear")),
        "badge_url": team.get("strBadge"),
        "country": team.get("strCountry"),
        "league": team.get("strLeague"),
        "description": (team.get("strDescriptionEN") or "")[:500],
    }

    if canon:
        _team_cache[canon] = result
        _write_cache(canon, result)
    return result


def get_last_events(team_name: str, n: int = 5) -> list[dict]:
    """Get last N events for a team."""
    team_data = search_team(team_name)
    if not team_data or not team_data.get("team_id"):
        return []

    data = _get(f"eventslast.php?id={team_data['team_id']}")
    if not data or not data.get("results"):
        return []

    events = []
    for ev in data["results"][:n]:
        events.append({
            "event_id": ev.get("idEvent"),
            "date": ev.get("dateEvent"),
            "home_team": ev.get("strHomeTeam"),
            "away_team": ev.get("strAwayTeam"),
            "home_score": _safe_int(ev.get("intHomeScore")),
            "away_score": _safe_int(ev.get("intAwayScore")),
            "venue": ev.get("strVenue"),
            "league": ev.get("strLeague"),
        })
    return events


def get_next_events(team_name: str, n: int = 5) -> list[dict]:
    """Get next N events for a team."""
    team_data = search_team(team_name)
    if not team_data or not team_data.get("team_id"):
        return []

    data = _get(f"eventsnext.php?id={team_data['team_id']}")
    if not data or not data.get("events"):
        return []

    events = []
    for ev in data["events"][:n]:
        events.append({
            "event_id": ev.get("idEvent"),
            "date": ev.get("dateEvent"),
            "home_team": ev.get("strHomeTeam"),
            "away_team": ev.get("strAwayTeam"),
            "venue": ev.get("strVenue"),
            "league": ev.get("strLeague"),
        })
    return events


def get_venue_info(team_name: str) -> dict | None:
    """
    Get venue/stadium information for a team.

    Returns:
        {stadium, capacity, location, surface, description}
    """
    team_data = search_team(team_name)
    if not team_data:
        return None

    return {
        "stadium": team_data.get("stadium"),
        "capacity": team_data.get("capacity"),
        "country": team_data.get("country"),
    }


def enrich_matches_with_venue(con, verbose: bool = True) -> int:
    """
    Enrich upcoming matches with venue/stadium data from TheSportsDB.

    Stores venue info in raw_json of match_extras for any upcoming match
    that doesn't already have venue data.

    Team metadata is served from DuckDB cache when available (< 30 days old),
    avoiding redundant API calls on every pipeline run.
    """
    import json

    upcoming = con.execute("""
        SELECT m.match_id, m.home_team
        FROM matches m
        WHERE m.status IN ('SCHEDULED', 'TIMED')
    """).fetchall()

    if not upcoming:
        return 0

    enriched = 0
    seen_teams: dict[str, dict | None] = {}

    for mid, home in upcoming:
        if home not in seen_teams:
            seen_teams[home] = get_venue_info(home)

        venue = seen_teams[home]
        if not venue or not venue.get("stadium"):
            continue

        try:
            existing = con.execute(
                "SELECT raw_json FROM match_extras WHERE match_id = ?",
                [mid]
            ).fetchone()

            if existing and existing[0]:
                try:
                    rj = json.loads(existing[0])
                except Exception:
                    rj = {}
            else:
                rj = {}

            rj["venue"] = venue

            if existing:
                con.execute(
                    "UPDATE match_extras SET raw_json = ? WHERE match_id = ?",
                    [json.dumps(rj), mid]
                )
            else:
                con.execute(
                    "INSERT INTO match_extras (match_id, provider, raw_json) VALUES (?, ?, ?)",
                    [mid, "thesportsdb", json.dumps(rj)]
                )
            enriched += 1
        except Exception as e:
            log.warning("TheSportsDB venue enrichment failed for %s: %s", mid, e)

    # Report cache stats
    cache_stats = _cache_stats()
    if verbose:
        log.info(
            "TheSportsDB: enriched %d matches with venue data "
            "(cache: %d entries, %d fresh)",
            enriched, cache_stats["total"], cache_stats["fresh"],
        )
    return enriched


def _cache_stats() -> dict:
    """Return stats about the persistent DuckDB cache."""
    con = _get_db_connection()
    if con is None:
        return {"total": 0, "fresh": 0, "stale": 0}
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)).isoformat()
        total = con.execute(
            "SELECT COUNT(*) FROM thesportsdb_cache"
        ).fetchone()[0]
        fresh = con.execute(
            "SELECT COUNT(*) FROM thesportsdb_cache WHERE fetched_at > ?",
            [cutoff],
        ).fetchone()[0]
        return {"total": total, "fresh": fresh, "stale": total - fresh}
    except Exception:
        return {"total": 0, "fresh": 0, "stale": 0}


def _safe_int(v) -> int | None:
    """Safely convert to int, returning None for invalid values."""
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None
