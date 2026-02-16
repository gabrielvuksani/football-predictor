"""
The Odds API v4 integration — multi-bookmaker odds for upcoming matches.

Free tier: 500 requests/month.  We cache aggressively and only fetch
for upcoming football matches.

Endpoints used:
    GET /v4/sports                      — list sports (soccer keys)
    GET /v4/sports/{sport}/odds         — odds for a sport
    GET /v4/sports/{sport}/events       — events list
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from footy.providers.ratelimit import RateLimiter
from footy.normalize import canonical_team_name

log = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com"

# Conservative rate limit (free tier is 500/month ≈ 16/day)
_rl = RateLimiter(max_calls=3, period_seconds=60)

# Soccer sport keys by competition (The Odds API naming)
SPORT_KEYS: dict[str, str] = {
    "PL": "soccer_epl",
    "PD": "soccer_spain_la_liga",
    "SA": "soccer_italy_serie_a",
    "BL1": "soccer_germany_bundesliga",
    "FL1": "soccer_france_ligue_one",
}


def _api_key() -> str | None:
    return os.getenv("THE_ODDS_API_KEY", "").strip() or None


def _get(path: str, params: dict[str, Any] | None = None) -> dict | list | None:
    """Make a GET request to The Odds API with rate limiting and error handling."""
    key = _api_key()
    if not key:
        return None

    _rl.wait()
    params = params or {}
    params["apiKey"] = key
    url = f"{BASE}{path}"

    for attempt in range(3):
        try:
            with httpx.Client(timeout=20.0) as c:
                r = c.get(url, params=params)
            if r.status_code == 401:
                log.warning("The Odds API: invalid API key")
                return None
            if r.status_code == 429:
                wait = min(60, 5 * (attempt + 1))
                log.warning("The Odds API rate limited, waiting %ds", wait)
                time.sleep(wait)
                continue
            if r.status_code == 422:
                log.debug("The Odds API 422: %s", r.text[:200])
                return None
            r.raise_for_status()

            # Log remaining quota
            remaining = r.headers.get("x-requests-remaining")
            used = r.headers.get("x-requests-used")
            if remaining:
                log.debug("Odds API quota: used=%s remaining=%s", used, remaining)

            return r.json()
        except httpx.TimeoutException:
            log.warning("The Odds API timeout (attempt %d)", attempt + 1)
            time.sleep(2)
        except httpx.HTTPStatusError as e:
            log.warning("The Odds API HTTP error: %s", e)
            return None
    return None


def fetch_odds(
    competition: str = "PL",
    markets: str = "h2h,totals",
    regions: str = "uk,eu",
) -> list[dict]:
    """
    Fetch current odds for upcoming matches in a competition.

    Returns a list of match dicts, each with:
        - id, sport_key, commence_time
        - home_team, away_team
        - bookmakers: [{key, title, markets: [{key, outcomes}]}]
    """
    sport_key = SPORT_KEYS.get(competition)
    if not sport_key:
        log.debug("No Odds API sport key for competition %s", competition)
        return []

    data = _get(f"/v4/sports/{sport_key}/odds", {
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
    })

    if not data or not isinstance(data, list):
        return []

    log.info("The Odds API: %d events for %s", len(data), competition)
    return data


def fetch_all_odds(
    competitions: list[str] | None = None,
) -> list[dict]:
    """Fetch odds for all tracked competitions."""
    if competitions is None:
        competitions = list(SPORT_KEYS.keys())

    all_events: list[dict] = []
    for comp in competitions:
        events = fetch_odds(comp)
        for ev in events:
            ev["_competition"] = comp
        all_events.extend(events)

    return all_events


def parse_odds_for_match(
    event: dict,
) -> dict[str, Any]:
    """
    Parse a single Odds API event into a structured odds dict.

    Returns dict with keys:
        home_team, away_team, commence_time, competition,
        best_h, best_d, best_a (best available odds),
        avg_h, avg_d, avg_a (average across bookmakers),
        n_bookmakers,
        odds_details: [{bookmaker, home, draw, away}],
        totals: {over_25, under_25} (best available)
    """
    home = canonical_team_name(event.get("home_team", ""))
    away = canonical_team_name(event.get("away_team", ""))
    commence = event.get("commence_time", "")

    bookmakers = event.get("bookmakers", [])
    h2h_odds: list[dict] = []
    totals_odds: list[dict] = []

    for bm in bookmakers:
        bm_name = bm.get("title", bm.get("key", "unknown"))
        for market in bm.get("markets", []):
            if market.get("key") == "h2h":
                outcomes = {o["name"]: float(o["price"])
                            for o in market.get("outcomes", [])}
                h = outcomes.get(event.get("home_team"), 0)
                d = outcomes.get("Draw", 0)
                a = outcomes.get(event.get("away_team"), 0)
                if h > 1 and d > 1 and a > 1:
                    h2h_odds.append({
                        "bookmaker": bm_name,
                        "home": h, "draw": d, "away": a,
                    })

            elif market.get("key") == "totals":
                for o in market.get("outcomes", []):
                    point = o.get("point", 0)
                    if point == 2.5:
                        name = o.get("name", "").lower()
                        price = float(o.get("price", 0))
                        totals_odds.append({
                            "bookmaker": bm_name,
                            "type": name,  # "over" or "under"
                            "line": point,
                            "price": price,
                        })

    # Compute best and average odds
    best_h = max((o["home"] for o in h2h_odds), default=0)
    best_d = max((o["draw"] for o in h2h_odds), default=0)
    best_a = max((o["away"] for o in h2h_odds), default=0)
    avg_h = sum(o["home"] for o in h2h_odds) / len(h2h_odds) if h2h_odds else 0
    avg_d = sum(o["draw"] for o in h2h_odds) / len(h2h_odds) if h2h_odds else 0
    avg_a = sum(o["away"] for o in h2h_odds) / len(h2h_odds) if h2h_odds else 0

    # Best over/under 2.5
    over_25 = max((o["price"] for o in totals_odds if o["type"] == "over"), default=0)
    under_25 = max((o["price"] for o in totals_odds if o["type"] == "under"), default=0)

    return {
        "home_team": home,
        "away_team": away,
        "commence_time": commence,
        "competition": event.get("_competition", ""),
        "best_h": best_h,
        "best_d": best_d,
        "best_a": best_a,
        "avg_h": avg_h,
        "avg_d": avg_d,
        "avg_a": avg_a,
        "n_bookmakers": len(h2h_odds),
        "odds_details": h2h_odds,
        "totals": {"over_25": over_25, "under_25": under_25},
    }


def ingest_odds_to_db(con, competitions: list[str] | None = None, verbose: bool = True) -> int:
    """
    Fetch odds from The Odds API and store them in match_extras for upcoming matches.

    Matches are joined by fuzzy team name matching + date proximity.
    """
    from footy.normalize import canonical_team_name
    from difflib import SequenceMatcher

    events = fetch_all_odds(competitions)
    if not events:
        if verbose:
            log.info("The Odds API: no events returned")
        return 0

    # Load upcoming matches from DB
    upcoming = con.execute("""
        SELECT match_id, home_team, away_team, utc_date
        FROM matches
        WHERE status IN ('SCHEDULED', 'TIMED')
        ORDER BY utc_date ASC
    """).fetchall()

    if not upcoming:
        if verbose:
            log.info("No upcoming matches in DB to match odds to")
        return 0

    def _norm(s: str) -> str:
        return (canonical_team_name(s) or s).lower().strip()

    def _sim(a: str, b: str) -> float:
        return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

    matched = 0
    for event in events:
        parsed = parse_odds_for_match(event)
        if not parsed["best_h"]:
            continue

        # Find best matching upcoming match
        best_match = None
        best_score = 0.0

        for mid, h, aw, dt in upcoming:
            score = (_sim(parsed["home_team"] or "", h) +
                     _sim(parsed["away_team"] or "", aw)) / 2
            if score > best_score and score > 0.65:
                best_score = score
                best_match = mid

        if best_match is None:
            continue

        # Upsert odds into match_extras
        try:
            existing = con.execute(
                "SELECT match_id FROM match_extras WHERE match_id = ?",
                [best_match]
            ).fetchone()

            if existing:
                con.execute("""
                    UPDATE match_extras SET
                        avgh = COALESCE(avgh, ?),
                        avgd = COALESCE(avgd, ?),
                        avga = COALESCE(avga, ?),
                        maxh = COALESCE(maxh, ?),
                        maxd = COALESCE(maxd, ?),
                        maxa = COALESCE(maxa, ?)
                    WHERE match_id = ?
                """, [
                    parsed["avg_h"], parsed["avg_d"], parsed["avg_a"],
                    parsed["best_h"], parsed["best_d"], parsed["best_a"],
                    best_match,
                ])
            else:
                con.execute("""
                    INSERT INTO match_extras (match_id, provider, avgh, avgd, avga,
                                              maxh, maxd, maxa)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    best_match, "the-odds-api",
                    parsed["avg_h"], parsed["avg_d"], parsed["avg_a"],
                    parsed["best_h"], parsed["best_d"], parsed["best_a"],
                ])
            matched += 1
        except Exception as e:
            log.warning("Failed to store odds for match %s: %s", best_match, e)

    if verbose:
        log.info("The Odds API: matched %d/%d events to DB matches", matched, len(events))
    return matched
