"""
Fantasy Premier League (FPL) API integration — injury data, player availability,
squad strength metrics for Premier League teams.

All endpoints are **public** (no API key needed).

Key data extracted:
    - Player injury/availability status (chance_of_playing_next_round)
    - Squad strength metrics (total team value, star player availability)
    - Fixture difficulty ratings
    - Minutes/form data for key player impact scoring
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from footy.providers.ratelimit import RateLimiter, TRANSIENT_ERRORS
from footy.normalize import canonical_team_name

log = logging.getLogger(__name__)

BASE = "https://fantasy.premierleague.com/api"
_rl = RateLimiter(max_calls=6, period_seconds=60)

# FPL team ID → canonical name mapping
_FPL_TEAM_MAP: dict[int, str] | None = None


def _get(path: str) -> dict | list | None:
    """GET request to FPL API with rate limiting."""
    _rl.wait()
    url = f"{BASE}{path}"
    for attempt in range(3):
        try:
            with httpx.Client(timeout=20.0, headers={
                "User-Agent": "footy-predictor/0.1"
            }) as c:
                r = c.get(url)
            if r.status_code == 429:
                import time
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                log.debug("FPL API %d: %s", r.status_code, path)
                return None
            return r.json()
        except TRANSIENT_ERRORS as e:
            log.warning("FPL API error (attempt %d): %s", attempt + 1, e)
            import time
            time.sleep(2)
    return None


def _ensure_team_map() -> dict[int, str]:
    """Load and cache the FPL team_id → canonical name mapping."""
    global _FPL_TEAM_MAP
    if _FPL_TEAM_MAP is not None:
        return _FPL_TEAM_MAP

    data = _get("/bootstrap-static/")
    if not data or "teams" not in data:
        log.warning("FPL: could not load bootstrap-static")
        _FPL_TEAM_MAP = {}
        return _FPL_TEAM_MAP

    _FPL_TEAM_MAP = {}
    for t in data["teams"]:
        name = canonical_team_name(t.get("name", ""))
        if name:
            _FPL_TEAM_MAP[t["id"]] = name

    log.debug("FPL: loaded %d team mappings", len(_FPL_TEAM_MAP))
    return _FPL_TEAM_MAP


def fetch_bootstrap() -> dict | None:
    """
    Fetch the master FPL data (teams, players, gameweeks).

    Returns the full bootstrap-static response with keys:
        events, teams, elements, element_types, ...
    """
    return _get("/bootstrap-static/")


def fetch_fixtures() -> list[dict]:
    """Fetch all FPL fixtures for the current season."""
    data = _get("/fixtures/")
    return data if isinstance(data, list) else []


def get_team_injury_report(team_name: str) -> dict:
    """
    Get injury/availability report for a team.

    Returns:
        {
            "team": str,
            "total_players": int,
            "available": int,
            "doubtful": int,       # chance_of_playing 26-75%
            "injured": int,        # chance_of_playing 0-25%
            "suspended": int,
            "injury_score": float, # 0-1, higher = more injuries (worse)
            "key_absences": [{name, position, status, chance, influence}],
            "squad_strength": float, # 0-1 relative playing strength available
        }
    """
    data = fetch_bootstrap()
    if not data:
        return {"team": team_name, "total_players": 0, "available": 0,
                "injury_score": 0.0, "squad_strength": 1.0,
                "key_absences": [], "doubtful": 0, "injured": 0, "suspended": 0}

    team_map = _ensure_team_map()
    # Find FPL team ID for this team
    canon = canonical_team_name(team_name)
    team_id = None
    for tid, tname in team_map.items():
        if tname == canon:
            team_id = tid
            break

    if team_id is None:
        return {"team": team_name, "total_players": 0, "available": 0,
                "injury_score": 0.0, "squad_strength": 1.0,
                "key_absences": [], "doubtful": 0, "injured": 0, "suspended": 0}

    # Filter players for this team
    elements = data.get("elements", [])
    team_players = [p for p in elements if p.get("team") == team_id]

    total = len(team_players)
    available = 0
    doubtful = 0
    injured = 0
    suspended = 0
    key_absences: list[dict] = []
    total_influence = 0.0
    available_influence = 0.0

    element_types = {et["id"]: et["singular_name"]
                     for et in data.get("element_types", [])}

    for p in team_players:
        chance = p.get("chance_of_playing_next_round")
        status = p.get("status", "a")  # a=available, d=doubtful, i=injured, s=suspended, u=unavailable
        influence = float(p.get("influence", "0") or "0")
        total_influence += influence

        if chance is None or chance >= 76:
            available += 1
            available_influence += influence
        elif 26 <= (chance or 0) <= 75:
            doubtful += 1
            available_influence += influence * (chance / 100.0)  # weighted contribution
        else:
            if status == "s":
                suspended += 1
            else:
                injured += 1

        # Track key absences (high-influence players who are doubtful/injured)
        if (chance is not None and chance < 76) or status in ("i", "s", "u"):
            pos = element_types.get(p.get("element_type", 0), "Unknown")
            key_absences.append({
                "name": f"{p.get('first_name', '')} {p.get('second_name', '')}".strip(),
                "position": pos,
                "status": status,
                "chance": chance,
                "influence": influence,
                "news": p.get("news", ""),
            })

    # Sort key absences by influence (most impactful first)
    key_absences.sort(key=lambda x: x.get("influence", 0), reverse=True)

    # Injury score: 0 = fully fit, 1 = all injured
    injury_score = 1.0 - (available / max(total, 1))
    # Squad strength: weighted by influence
    squad_strength = available_influence / max(total_influence, 1.0)

    return {
        "team": canon,
        "total_players": total,
        "available": available,
        "doubtful": doubtful,
        "injured": injured,
        "suspended": suspended,
        "injury_score": round(injury_score, 3),
        "squad_strength": round(squad_strength, 3),
        "key_absences": key_absences[:5],  # top 5 most impactful
    }


def get_all_teams_availability() -> dict[str, dict]:
    """
    Get availability reports for all PL teams in one API call.

    Returns dict mapping team_name → injury report.
    """
    data = fetch_bootstrap()
    if not data:
        return {}

    team_map = _ensure_team_map()
    elements = data.get("elements", [])
    element_types = {et["id"]: et["singular_name"]
                     for et in data.get("element_types", [])}

    # Group players by team
    by_team: dict[int, list[dict]] = {}
    for p in elements:
        tid = p.get("team")
        if tid:
            by_team.setdefault(tid, []).append(p)

    results: dict[str, dict] = {}
    for tid, players in by_team.items():
        tname = team_map.get(tid)
        if not tname:
            continue

        total = len(players)
        available = 0
        doubtful = 0
        injured = 0
        suspended = 0
        total_influence = 0.0
        available_influence = 0.0

        for p in players:
            chance = p.get("chance_of_playing_next_round")
            status = p.get("status", "a")
            influence = float(p.get("influence", "0") or "0")
            total_influence += influence

            if chance is None or chance >= 76:
                available += 1
                available_influence += influence
            elif 26 <= (chance or 0) <= 75:
                doubtful += 1
                available_influence += influence * (chance / 100.0)
            else:
                if status == "s":
                    suspended += 1
                else:
                    injured += 1

        injury_score = 1.0 - (available / max(total, 1))
        squad_strength = available_influence / max(total_influence, 1.0)

        results[tname] = {
            "team": tname,
            "total_players": total,
            "available": available,
            "doubtful": doubtful,
            "injured": injured,
            "suspended": suspended,
            "injury_score": round(injury_score, 3),
            "squad_strength": round(squad_strength, 3),
        }

    return results


def get_fixture_difficulty() -> dict[str, dict]:
    """
    Get FPL fixture difficulty rating for each team's upcoming matches.

    FPL assigns each fixture a difficulty 1-5 from each team's perspective.

    Returns dict mapping team_name → {
        "next_3_avg_difficulty": float,
        "next_6_avg_difficulty": float,
        "upcoming_fixtures": [{opponent, difficulty, is_home}],
    }
    """
    data = fetch_bootstrap()
    fixtures = fetch_fixtures()
    if not data or not fixtures:
        return {}

    team_map = _ensure_team_map()

    # Find current gameweek
    events = data.get("events", [])
    current_gw = 1
    for ev in events:
        if ev.get("is_current"):
            current_gw = ev["id"]
            break

    # Filter future fixtures
    future = [f for f in fixtures if not f.get("finished") and f.get("event")]
    future.sort(key=lambda f: (f.get("event", 999), f.get("kickoff_time", "")))

    # Build per-team difficulty
    team_fixtures: dict[str, list[dict]] = {}
    for f in future:
        home_id = f.get("team_h")
        away_id = f.get("team_a")
        h_diff = f.get("team_h_difficulty", 3)
        a_diff = f.get("team_a_difficulty", 3)

        home_name = team_map.get(home_id, "")
        away_name = team_map.get(away_id, "")

        if home_name:
            team_fixtures.setdefault(home_name, []).append({
                "opponent": away_name,
                "difficulty": h_diff,
                "is_home": True,
            })
        if away_name:
            team_fixtures.setdefault(away_name, []).append({
                "opponent": home_name,
                "difficulty": a_diff,
                "is_home": False,
            })

    results: dict[str, dict] = {}
    for team, fxs in team_fixtures.items():
        next_3 = fxs[:3]
        next_6 = fxs[:6]
        results[team] = {
            "next_3_avg_difficulty": (
                sum(f["difficulty"] for f in next_3) / len(next_3)
                if next_3 else 3.0
            ),
            "next_6_avg_difficulty": (
                sum(f["difficulty"] for f in next_6) / len(next_6)
                if next_6 else 3.0
            ),
            "upcoming_fixtures": fxs[:6],
        }

    return results
