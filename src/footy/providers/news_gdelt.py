from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

from footy.providers.ratelimit import RateLimiter

BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

# Gentle pacing + retries
_rl = RateLimiter(max_calls=1, period_seconds=1)

def _fmt(dt: datetime) -> str:
    # GDELT expects YYYYMMDDHHMMSS in UTC
    return dt.strftime("%Y%m%d%H%M%S")

def fetch_team_news(team: str, days_back: int = 2, max_records: int = 10) -> pd.DataFrame:
    """
    Direct GDELT DOC 2.1 call with hard timeouts + retry/backoff.
    Returns a DataFrame of articles (url/title/seendate/domain/etc).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    params = {
        "query": team,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_records),
        "startdatetime": _fmt(start),
        "enddatetime": _fmt(end),
    }

    timeout = httpx.Timeout(20.0, connect=10.0)
    backoff = 1.5

    with httpx.Client(timeout=timeout, headers={"User-Agent": "footy-predictor/0.1"}) as c:
        for attempt in range(6):
            _rl.wait()
            try:
                r = c.get(BASE, params=params)
            except httpx.RequestError:
                time.sleep(backoff)
                backoff *= 2.0
                continue

            ct = (r.headers.get("content-type") or "").lower()

            # Rate limit / transient server issues
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2.0
                continue

            # Sometimes GDELT returns HTML even with 200; treat as transient
            if r.status_code == 200 and "application/json" not in ct:
                time.sleep(backoff)
                backoff *= 2.0
                continue

            r.raise_for_status()
            data = r.json()
            arts = data.get("articles", []) or []
            if not arts:
                return pd.DataFrame()
            return pd.DataFrame(arts)

    # After retries, give up cleanly (don’t hang the pipeline)
    return pd.DataFrame()
