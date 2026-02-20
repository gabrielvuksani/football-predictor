"""
TheSportsDB integration — team metadata, venue info, badges, and event details.

Free API key (dev/test): 3
Provides enrichment data not available from other sources:
    - Team metadata (stadium, capacity, founded year, etc.)
    - Past/next events per team
    - Event details with venue info
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from footy.providers.ratelimit import RateLimiter, TRANSIENT_ERRORS
from footy.normalize import canonical_team_name

log = logging.getLogger(__name__)

_rl = RateLimiter(max_calls=8, period_seconds=60)


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


# ---- Caches ----
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


def search_team(name: str) -> dict | None:
    """
    Search for a team by name and return metadata.

    Returns dict with:
        team_id, name, short_name, stadium, capacity, founded,
        badge_url, country, league, description
    """
    canon = canonical_team_name(name)
    if canon in _team_cache:
        return _team_cache[canon]

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

    if verbose:
        log.info("TheSportsDB: enriched %d matches with venue data", enriched)
    return enriched


def _safe_int(v) -> int | None:
    """Safely convert to int, returning None for invalid values."""
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None
