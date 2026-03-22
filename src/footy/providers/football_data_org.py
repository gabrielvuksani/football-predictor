from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
import math

import httpx

from footy.config import settings
from footy.providers.ratelimit import RateLimiter, TRANSIENT_ERRORS
from footy.normalize import canonical_team_name

log = logging.getLogger(__name__)

BASE = "https://api.football-data.org/v4"

_rl = RateLimiter(max_calls=10, period_seconds=60)

_MAX_RETRIES = 3
_RETRY_BACKOFF = (2.0, 5.0, 10.0)  # seconds

def _client():
    s = settings()
    return httpx.Client(
        headers={
            "X-Auth-Token": s.football_data_org_token,
            # X-Unfold headers — free tier provides expanded match detail
            "X-Unfold-Goals": "true",
            "X-Unfold-Lineups": "true",
            "X-Unfold-Bookings": "true",
            "X-Unfold-Subs": "true",
        },
        timeout=30.0
    )

def fetch_matches(date_from: date, date_to_exclusive: date) -> dict:
    _rl.wait()
    for attempt in range(_MAX_RETRIES):
        try:
            with _client() as c:
                r = c.get(
                    f"{BASE}/matches",
                    params={"dateFrom": str(date_from), "dateTo": str(date_to_exclusive)},
                )
                if r.status_code == 429:
                    wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                    log.warning("football-data.org rate limited (429), retrying in %.1fs", wait)
                    time.sleep(wait)
                    _rl.wait()
                    continue
                if r.status_code >= 500:
                    wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                    log.warning("football-data.org server error (%d), retrying in %.1fs", r.status_code, wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
        except httpx.TimeoutException:
            wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
            log.warning("football-data.org timeout, retrying in %.1fs", wait)
            time.sleep(wait)
        except TRANSIENT_ERRORS as exc:
            wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
            log.warning("football-data.org network error (%s), retrying in %.1fs", exc, wait)
            time.sleep(wait)
        except httpx.HTTPStatusError:
            raise
    # Final attempt — raise on failure
    with _client() as c:
        r = c.get(
            f"{BASE}/matches",
            params={"dateFrom": str(date_from), "dateTo": str(date_to_exclusive)},
        )
        r.raise_for_status()
        return r.json()

def fetch_matches_range(date_from: date, date_to_inclusive: date, chunk_days: int = 10, verbose: bool = False) -> list[dict]:
    all_matches: list[dict] = []
    seen_ids: set[int] = set()

    total_days = (date_to_inclusive - date_from).days + 1
    n_chunks = max(1, math.ceil(total_days / chunk_days))

    cur = date_from
    chunk_i = 0
    while cur <= date_to_inclusive:
        chunk_i += 1
        end_inclusive = min(cur + timedelta(days=chunk_days - 1), date_to_inclusive)
        end_exclusive = end_inclusive + timedelta(days=1)

        if verbose:
            log.debug("chunk %d/%d: %s → %s", chunk_i, n_chunks, cur, end_inclusive)

        data = fetch_matches(cur, end_exclusive)
        got = 0
        for m in data.get("matches", []):
            mid = int(m.get("id"))
            if mid not in seen_ids:
                seen_ids.add(mid)
                all_matches.append(m)
                got += 1

        if verbose:
            log.debug("received %d new matches", got)

        cur = end_inclusive + timedelta(days=1)

    return all_matches

def normalize_match(m: dict) -> dict:
    score = (m.get("score") or {}).get("fullTime") or {}
    home_goals = score.get("home")
    away_goals = score.get("away")
    comp = (m.get("competition") or {}).get("code")
    season = (m.get("season") or {}).get("startDate", "")[:4]

    # Half-time score (from score block, not stats)
    ht = (m.get("score") or {}).get("halfTime") or {}
    hthg = ht.get("home")
    htag = ht.get("away")

    # Extract lineups/formations if unfolded
    homeLineup = m.get("homeTeam", {}).get("lineup") or []
    awayLineup = m.get("awayTeam", {}).get("lineup") or []
    formation_home = m.get("homeTeam", {}).get("formation")
    formation_away = m.get("awayTeam", {}).get("formation")

    lineup_home_str = json.dumps([
        {"name": p.get("name"), "position": p.get("position"), "shirtNumber": p.get("shirtNumber")}
        for p in homeLineup
    ]) if homeLineup else None
    lineup_away_str = json.dumps([
        {"name": p.get("name"), "position": p.get("position"), "shirtNumber": p.get("shirtNumber")}
        for p in awayLineup
    ]) if awayLineup else None

    # Extract booking counts from unfolded data
    bookings = m.get("bookings") or []
    hy = sum(1 for b in bookings if b.get("card") == "YELLOW_CARD"
             and (b.get("team") or {}).get("id") == (m.get("homeTeam") or {}).get("id"))
    ay = sum(1 for b in bookings if b.get("card") == "YELLOW_CARD"
             and (b.get("team") or {}).get("id") == (m.get("awayTeam") or {}).get("id"))
    hr = sum(1 for b in bookings if b.get("card") == "RED_CARD"
             and (b.get("team") or {}).get("id") == (m.get("homeTeam") or {}).get("id"))
    ar = sum(1 for b in bookings if b.get("card") == "RED_CARD"
             and (b.get("team") or {}).get("id") == (m.get("awayTeam") or {}).get("id"))

    result = {
        "match_id": int(m["id"]),
        "provider": "football-data.org",
        "competition": comp,
        "season": int(season) if season.isdigit() else None,
        "utc_date": m.get("utcDate"),
        "status": m.get("status"),
        "home_team": canonical_team_name((m.get("homeTeam") or {}).get("name")),
        "away_team": canonical_team_name((m.get("awayTeam") or {}).get("name")),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "raw_json": json.dumps(m),
    }

    # Extra fields for match_extras upsert (only if data is present)
    extras = {}
    if hthg is not None:
        extras["hthg"] = hthg
    if htag is not None:
        extras["htag"] = htag
    if formation_home:
        extras["formation_home"] = formation_home
    if formation_away:
        extras["formation_away"] = formation_away
    if lineup_home_str:
        extras["lineup_home"] = lineup_home_str
    if lineup_away_str:
        extras["lineup_away"] = lineup_away_str
    if bookings:
        extras["hy"] = hy
        extras["ay"] = ay
        extras["hr"] = hr
        extras["ar"] = ar

    result["_extras"] = extras
    return result
