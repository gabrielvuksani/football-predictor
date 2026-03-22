"""
Opta Analyst — scrape win-probability predictions from theanalyst.com.

**Disclaimer**: This module is intended for personal / research use only.
The data on theanalyst.com is owned by Stats Perform / Opta.  Usage must
comply with their Terms of Service.  Some content may be behind a paywall;
this scraper only attempts to read publicly-accessible pages.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

import duckdb
import httpx

from footy.providers.ratelimit import RateLimiter, TRANSIENT_ERRORS
from footy.team_mapping import get_canonical_name

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter: max 10 requests per 60 s
# ---------------------------------------------------------------------------
_LIMITER = RateLimiter(max_calls=10, period_seconds=60)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}

_BASE_URL = "https://theanalyst.com"

# Candidate paths where predictions may live — tried in order.
_PREDICTION_PATHS: list[str] = [
    "/football/predictions/",
    "/football/predictions/premier-league/",
    "/football/predictions/la-liga/",
    "/football/predictions/bundesliga/",
    "/football/predictions/serie-a/",
    "/football/predictions/ligue-1/",
]

# Map optional competition filter → path fragments we care about
_COMP_PATH_MAP: dict[str, list[str]] = {
    "PL": ["/premier-league"],
    "PD": ["/la-liga"],
    "BL1": ["/bundesliga"],
    "SA": ["/serie-a"],
    "FL1": ["/ligue-1"],
}

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def ensure_opta_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create the opta_predictions table if it doesn't exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS opta_predictions (
            match_key VARCHAR PRIMARY KEY,   -- "home_team__away_team__date"
            home_win  DOUBLE,
            draw      DOUBLE,
            away_win  DOUBLE,
            source_url VARCHAR,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def _match_key(home: str, away: str, date_str: str) -> str:
    return f"{home}__{away}__{date_str}"


# ---------------------------------------------------------------------------
# HTML parsing helpers (regex-based, no BS4 dependency)
# ---------------------------------------------------------------------------

# Pattern 1: JSON-like data embedded in the page
# e.g. "homeWin":0.45,"draw":0.25,"awayWin":0.30
_JSON_PROB_RE = re.compile(
    r'"homeWin"\s*:\s*([\d.]+)\s*,\s*'
    r'"draw"\s*:\s*([\d.]+)\s*,\s*'
    r'"awayWin"\s*:\s*([\d.]+)',
    re.IGNORECASE,
)

# Pattern 2: percentage text like "45%" near team names
_PCT_RE = re.compile(r'(\d{1,3}(?:\.\d+)?)\s*%')

# Pattern 3: team names — broad, picks up <span>/aria labels etc.
_TEAM_RE = re.compile(
    r'(?:home[-_]?team|away[-_]?team|team[-_]?name)["\s:>]+([^"<]{2,40})',
    re.IGNORECASE,
)

# Pattern 4: match card blocks (generic)
_MATCH_BLOCK_RE = re.compile(
    r'(?:class=["\'][^"\']*prediction[^"\']*["\']|'
    r'data-match|'
    r'class=["\'][^"\']*match[-_]?card[^"\']*["\'])'
    r'(.*?)'
    r'(?=class=["\'][^"\']*(?:prediction|match[-_]?card)|$)',
    re.IGNORECASE | re.DOTALL,
)

# Date pattern: YYYY-MM-DD
_DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')


def _extract_predictions_from_html(html: str, source_url: str) -> list[dict]:
    """
    Best-effort extraction of match predictions from page HTML.

    Returns a list of dicts with keys:
        home_team, away_team, date, home_win, draw, away_win, source_url
    """
    results: list[dict] = []

    # ------------------------------------------------------------------
    # Strategy A: look for structured JSON probability data
    # ------------------------------------------------------------------
    json_matches = _JSON_PROB_RE.findall(html)
    if json_matches:
        # Try to pair with team names
        teams = _TEAM_RE.findall(html)
        dates = _DATE_RE.findall(html)
        for i, (hw, dr, aw) in enumerate(json_matches):
            home = teams[i * 2] if i * 2 < len(teams) else f"Home_{i}"
            away = teams[i * 2 + 1] if i * 2 + 1 < len(teams) else f"Away_{i}"
            date_str = dates[i] if i < len(dates) else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            results.append({
                "home_team": _normalise_team(home),
                "away_team": _normalise_team(away),
                "date": date_str,
                "home_win": _safe_float(hw),
                "draw": _safe_float(dr),
                "away_win": _safe_float(aw),
                "source_url": source_url,
            })
        return results

    # ------------------------------------------------------------------
    # Strategy B: look for match-card blocks with percentages
    # ------------------------------------------------------------------
    blocks = _MATCH_BLOCK_RE.findall(html)
    for block in blocks:
        pcts = _PCT_RE.findall(block)
        teams = _TEAM_RE.findall(block)
        dates = _DATE_RE.findall(block)
        if len(pcts) >= 3 and len(teams) >= 2:
            date_str = dates[0] if dates else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            results.append({
                "home_team": _normalise_team(teams[0]),
                "away_team": _normalise_team(teams[1]),
                "date": date_str,
                "home_win": _safe_float(pcts[0]) / 100.0,
                "draw": _safe_float(pcts[1]) / 100.0,
                "away_win": _safe_float(pcts[2]) / 100.0,
                "source_url": source_url,
            })

    return results


def _normalise_team(raw: str) -> str:
    """Map scraped team name → canonical display name."""
    cleaned = raw.strip()
    canonical = get_canonical_name(cleaned, provider="opta")
    return canonical or cleaned


def _safe_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Scrape logic
# ---------------------------------------------------------------------------


def _fetch_page(url: str, client: httpx.Client) -> Optional[str]:
    """GET a page, respecting rate limits. Returns HTML or None."""
    _LIMITER.wait()
    try:
        resp = client.get(url, headers=_HEADERS, follow_redirects=True, timeout=15)
        if resp.status_code == 200:
            return resp.text
        log.warning("Opta page %s returned HTTP %d", url, resp.status_code)
    except (httpx.HTTPError, httpx.TimeoutException, httpx.ConnectError,
            httpx.RemoteProtocolError, ConnectionResetError, OSError) as exc:
        log.warning("Opta request failed for %s: %s", url, exc)
    return None


def _discover_prediction_links(index_html: str, base_url: str) -> list[str]:
    """
    From an index/listing page, find links to individual prediction pages.
    """
    # Look for hrefs containing "prediction" or "preview"
    href_re = re.compile(
        r'href=["\'](' + re.escape(base_url) + r'[^"\']*(?:prediction|preview)[^"\']*)["\']',
        re.IGNORECASE,
    )
    # Also relative hrefs
    rel_re = re.compile(
        r'href=["\'](/[^"\']*(?:prediction|preview)[^"\']*)["\']',
        re.IGNORECASE,
    )
    links = set(href_re.findall(index_html))
    for rel in rel_re.findall(index_html):
        links.add(base_url.rstrip("/") + rel)
    return sorted(links)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_opta_predictions(
    con: duckdb.DuckDBPyConnection,
    competition: Optional[str] = None,
) -> list[dict]:
    """
    Fetch Opta predictions for upcoming matches from theanalyst.com.

    Scrapes publicly-accessible prediction pages, extracts win probabilities,
    maps team names to canonical forms, and caches results in the DB.

    Args:
        con: DuckDB connection.
        competition: Optional competition code (e.g. "PL", "BL1") to
                     restrict which pages are scraped.

    Returns:
        List of prediction dicts with keys:
            match_key, home_team, away_team, date,
            home_win, draw, away_win, source_url
    """
    ensure_opta_table(con)

    # Decide which paths to try
    if competition and competition.upper() in _COMP_PATH_MAP:
        paths = [
            f"/football/predictions{frag}/"
            for frag in _COMP_PATH_MAP[competition.upper()]
        ]
    else:
        paths = list(_PREDICTION_PATHS)

    all_predictions: list[dict] = []

    with httpx.Client(http2=False) as client:
        for path in paths:
            url = _BASE_URL + path
            log.info("Scraping Opta predictions from %s", url)
            html = _fetch_page(url, client)
            if not html:
                continue

            # First try to extract directly from the listing page
            preds = _extract_predictions_from_html(html, url)

            # If the listing page has links to individual match predictions,
            # follow a handful of them (cap at 15 to stay polite).
            if not preds:
                sub_links = _discover_prediction_links(html, _BASE_URL)
                for link in sub_links[:15]:
                    sub_html = _fetch_page(link, client)
                    if sub_html:
                        preds.extend(
                            _extract_predictions_from_html(sub_html, link)
                        )

            all_predictions.extend(preds)

    # Upsert into DB
    for p in all_predictions:
        mk = _match_key(p["home_team"], p["away_team"], p["date"])
        p["match_key"] = mk
        try:
            con.execute(
                """
                INSERT INTO opta_predictions (match_key, home_win, draw, away_win, source_url, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (match_key) DO UPDATE SET
                    home_win   = excluded.home_win,
                    draw       = excluded.draw,
                    away_win   = excluded.away_win,
                    source_url = excluded.source_url,
                    scraped_at = excluded.scraped_at
                """,
                [mk, p["home_win"], p["draw"], p["away_win"], p["source_url"],
                 datetime.now(timezone.utc)],
            )
        except Exception as exc:
            log.warning("Failed to upsert opta prediction %s: %s", mk, exc)

    log.info("Scraped %d Opta predictions", len(all_predictions))
    return all_predictions


def get_cached_predictions(
    con: duckdb.DuckDBPyConnection,
) -> list[dict]:
    """Return all cached Opta predictions from the DB."""
    ensure_opta_table(con)
    try:
        rows = con.execute(
            "SELECT match_key, home_win, draw, away_win, source_url, scraped_at "
            "FROM opta_predictions ORDER BY scraped_at DESC"
        ).fetchall()
    except Exception:
        return []

    return [
        {
            "match_key": r[0],
            "home_win": r[1],
            "draw": r[2],
            "away_win": r[3],
            "source_url": r[4],
            "scraped_at": r[5],
        }
        for r in rows
    ]
