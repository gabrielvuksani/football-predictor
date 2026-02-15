from __future__ import annotations

import json
from datetime import date, timedelta
import math

import httpx

from footy.config import settings
from footy.providers.ratelimit import RateLimiter
from footy.normalize import canonical_team_name

BASE = "https://api.football-data.org/v4"

_rl = RateLimiter(max_calls=10, period_seconds=60)

def _client():
    s = settings()
    return httpx.Client(
        headers={"X-Auth-Token": s.football_data_org_token},
        timeout=30.0
    )

def fetch_matches(date_from: date, date_to_exclusive: date) -> dict:
    _rl.wait()
    with _client() as c:
        r = c.get(
            f"{BASE}/matches",
            params={"dateFrom": str(date_from), "dateTo": str(date_to_exclusive)},
        )
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"football-data.org error {r.status_code} for {r.url}\nResponse: {r.text}"
            ) from e
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
            print(f"[fd.org] chunk {chunk_i}/{n_chunks}: {cur} → {end_inclusive}", flush=True)

        data = fetch_matches(cur, end_exclusive)
        got = 0
        for m in data.get("matches", []):
            mid = int(m.get("id"))
            if mid not in seen_ids:
                seen_ids.add(mid)
                all_matches.append(m)
                got += 1

        if verbose:
            print(f"[fd.org]   received {got} new matches", flush=True)

        cur = end_inclusive + timedelta(days=1)

    return all_matches

def normalize_match(m: dict) -> dict:
    score = (m.get("score") or {}).get("fullTime") or {}
    home_goals = score.get("home")
    away_goals = score.get("away")
    comp = (m.get("competition") or {}).get("code")
    season = (m.get("season") or {}).get("startDate", "")[:4]
    return {
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
