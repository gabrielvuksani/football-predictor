from __future__ import annotations

import os, time, json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import httpx


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _soft_norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    # common football suffix/prefix noise
    for tok in [" football club", " fc", " cf", " sc", " ac", " sv", " u19", " u21", " b", " ii"]:
        s = s.replace(tok, "")
    # punctuation-ish
    for ch in ["'", ".", ",", ":", ";", "(", ")", "[", "]", "{", "}", "-", "_", "’"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def _sim(a: str, b: str) -> float:
    a = _soft_norm(a)
    b = _soft_norm(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class AFClient:
    base: str
    key: str
    timeout: float = 30.0

    def _headers(self) -> Dict[str, str]:
        return {"x-apisports-key": self.key}

    def get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base.rstrip("/") + path
        # Retry on 429, mild backoff; respect headers when possible
        for attempt in range(1, 6):
            with httpx.Client(timeout=self.timeout, headers=self._headers()) as c:
                r = c.get(url, params=params)

            if r.status_code == 429:
                # rate limited: wait a bit (API documents per-minute caps)
                wait_s = min(60, 3 * attempt)
                time.sleep(wait_s)
                continue

            r.raise_for_status()

            # If minute remaining is 0, pause briefly to avoid next call failing
            rem = r.headers.get("X-RateLimit-Remaining") or r.headers.get("x-ratelimit-remaining")
            try:
                if rem is not None and int(rem) <= 0:
                    time.sleep(3)
            except Exception:
                pass

            data = r.json()
            return data

        raise RuntimeError("API-Football: too many 429 retries")


def client_from_env() -> AFClient:
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing env var API_FOOTBALL_KEY")
    base = os.getenv("API_FOOTBALL_BASE", "https://v3.football.api-sports.io").strip()
    return AFClient(base=base, key=key)


def ensure_tables(con) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS af_leagues (
            competition VARCHAR PRIMARY KEY,
            league_id INTEGER,
            season INTEGER,
            league_name VARCHAR,
            country VARCHAR,
            updated_at TIMESTAMP
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS af_fixture_map (
            match_id BIGINT PRIMARY KEY,
            fixture_id BIGINT,
            competition VARCHAR,
            league_id INTEGER,
            season INTEGER,
            fixture_date DATE,
            updated_at TIMESTAMP
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS af_context (
            match_id BIGINT PRIMARY KEY,
            fixture_id BIGINT,
            fetched_at TIMESTAMP,
            fixture_json VARCHAR,
            injuries_json VARCHAR,
            home_injuries INTEGER,
            away_injuries INTEGER,
            llm_model VARCHAR,
            llm_summary VARCHAR
        );
    """)


# Your project’s competition codes -> search hints (used to resolve league IDs)
_LEAGUE_HINTS: Dict[str, Tuple[str, str]] = {
    "PL": ("Premier League", "England"),
    "PD": ("La Liga", "Spain"),
    "SA": ("Serie A", "Italy"),
    "BL1": ("Bundesliga", "Germany"),
    "FL1": ("Ligue 1", "France"),
    "DED": ("Eredivisie", "Netherlands"),
}


def _season_year_for_date(d: datetime) -> int:
    # API-Football season parameter is the starting year (typical)
    y = d.year
    return y if d.month >= 7 else y - 1


def resolve_league(con, competition: str, for_date: datetime, verbose: bool = True) -> Tuple[int, int]:
    """
    Returns (league_id, season_year). Uses DB cache; otherwise queries /leagues?search=...
    """
    ensure_tables(con)
    season = _season_year_for_date(for_date)

    row = con.execute(
        "SELECT league_id, season, updated_at FROM af_leagues WHERE competition=?",
        [competition],
    ).fetchone()

    if row and row[0] is not None and int(row[1]) == int(season):
        return int(row[0]), int(row[1])

    hint = _LEAGUE_HINTS.get(competition)
    if not hint:
        raise RuntimeError(f"No league hint configured for competition={competition}")
    search_name, country = hint

    af = client_from_env()
    data = af.get("/leagues", {"search": search_name})

    best = None
    for item in data.get("response", []) or []:
        lg = item.get("league", {}) or {}
        ct = item.get("country", {}) or {}
        if ct.get("name") != country:
            continue
        # pick first "League" match
        if lg.get("type") == "League":
            best = item
            break
    if not best and (data.get("response") or []):
        best = (data.get("response") or [None])[0]

    if not best:
        raise RuntimeError(f"Could not resolve league for {competition} via /leagues search")

    league_id = int(best["league"]["id"])
    league_name = best["league"].get("name")
    ctry = best.get("country", {}).get("name")

    con.execute(
        "INSERT OR REPLACE INTO af_leagues VALUES (?, ?, ?, ?, ?, ?)",
        [competition, league_id, season, league_name, ctry, _utcnow()],
    )
    if verbose:
        print(f"[af] resolved {competition} -> league_id={league_id} season={season} ({league_name}, {ctry})", flush=True)
    return league_id, season


def fetch_fixtures_index(con, competition: str, date_from: datetime, date_to: datetime, verbose: bool = True) -> Dict[Tuple[str, str, str], int]:
    """
    Returns dict keyed by (YYYY-MM-DD, norm_home, norm_away) -> fixture_id
    """
    league_id, season = resolve_league(con, competition, date_from, verbose=verbose)

    af = client_from_env()
    params = {
        "league": league_id,
        "season": season,
        "from": date_from.date().isoformat(),
        "to": date_to.date().isoformat(),
    }
    data = af.get("/fixtures", params)
    idx: Dict[Tuple[str, str, str], int] = {}
    for fx in data.get("response", []) or []:
        fid = int(fx["fixture"]["id"])
        fdate = fx["fixture"]["date"]  # ISO
        # key by UTC date only
        d = fdate[:10]
        ht = fx.get("teams", {}).get("home", {}).get("name", "")
        at = fx.get("teams", {}).get("away", {}).get("name", "")
        idx[(d, _soft_norm(ht), _soft_norm(at))] = fid
    if verbose:
        print(f"[af] fixtures index {competition}: {len(idx)} fixtures", flush=True)
    return idx


def _best_fuzzy_fixture(target_home: str, target_away: str, candidates: List[Tuple[int, str, str]]) -> Optional[int]:
    """
    candidates: [(fixture_id, home_name, away_name), ...] same date+competition
    """
    best = (0.0, None)
    for fid, h, a in candidates:
        s = 0.5 * _sim(target_home, h) + 0.5 * _sim(target_away, a)
        if s > best[0]:
            best = (s, fid)
    if best[0] >= 0.80:
        return best[1]
    return None


def map_upcoming_matches(con, lookahead_days: int = 7, verbose: bool = True) -> int:
    """
    Fills af_fixture_map for upcoming matches in your matches table.
    """
    ensure_tables(con)
    now = _utcnow()
    d1 = now + timedelta(days=int(lookahead_days))

    up = con.execute(
        """
        SELECT match_id, competition, utc_date, home_team, away_team
        FROM matches
        WHERE status IN ('SCHEDULED','TIMED')
          AND utc_date >= ?
          AND utc_date < ?
        """,
        [now, d1],
    ).df()

    if up is None or up.empty:
        if verbose:
            print("[af] no upcoming matches to map", flush=True)
        return 0

    up["utc_date"] = up["utc_date"].astype("datetime64[ns]")
    up["date"] = up["utc_date"].dt.strftime("%Y-%m-%d")

    inserted = 0
    for comp, grp in up.groupby("competition"):
        if comp not in _LEAGUE_HINTS:
            continue
        date_from = now - timedelta(days=1)
        date_to = d1 + timedelta(days=1)

        idx = fetch_fixtures_index(con, comp, date_from, date_to, verbose=verbose)

        # also build candidates-by-date for fuzzy fallback
        cand_by_date: Dict[str, List[Tuple[int, str, str]]] = {}
        for (d, nh, na), fid in idx.items():
            cand_by_date.setdefault(d, []).append((fid, nh, na))

        for r in grp.itertuples(index=False):
            mid = int(r.match_id)
            d = str(r.date)
            h = str(r.home_team)
            a = str(r.away_team)

            key = (d, _soft_norm(h), _soft_norm(a))
            fid = idx.get(key)

            if fid is None:
                # fuzzy same day fallback
                cands = cand_by_date.get(d, [])
                fid = _best_fuzzy_fixture(h, a, cands)

            if fid is None:
                if verbose:
                    print(f"[af] unmatched: {comp} {d} | {h} vs {a}", flush=True)
                continue

            league_id, season = resolve_league(con, comp, now, verbose=False)
            con.execute(
                "INSERT OR REPLACE INTO af_fixture_map VALUES (?, ?, ?, ?, ?, ?, ?)",
                [mid, int(fid), comp, int(league_id), int(season), d, _utcnow()],
            )
            inserted += 1

    if verbose:
        print(f"[af] fixture_map upserted: {inserted}", flush=True)
    return inserted


def fetch_fixture_bundle(fixture_id: int) -> Dict[str, Any]:
    """
    Gets fixture details (lineups/stats/etc) using /fixtures?id=... .
    """
    af = client_from_env()
    return af.get("/fixtures", {"id": int(fixture_id)})


def fetch_injuries(fixture_id: int) -> Dict[str, Any]:
    """
    Injuries exist as a dedicated endpoint; we try /injuries?fixture=... .
    If your plan doesn’t include it, we gracefully return empty.
    """
    af = client_from_env()
    try:
        return af.get("/injuries", {"fixture": int(fixture_id)})
    except Exception:
        return {"response": []}


def upsert_context(con, stale_hours: int = 6, verbose: bool = True) -> int:
    """
    For mapped upcoming fixtures, store fixture_json + injuries_json with counts.
    """
    ensure_tables(con)
    now = _utcnow()
    stale_cut = now - timedelta(hours=int(stale_hours))

    rows = con.execute(
        """
        SELECT m.match_id, m.utc_date, fm.fixture_id
        FROM matches m
        JOIN af_fixture_map fm ON fm.match_id = m.match_id
        WHERE m.status IN ('SCHEDULED','TIMED')
          AND m.utc_date >= ?
          AND m.utc_date < ?
        """,
        [now - timedelta(days=1), now + timedelta(days=8)],
    ).df()

    if rows is None or rows.empty:
        if verbose:
            print("[af] no mapped upcoming fixtures to ingest", flush=True)
        return 0

    wrote = 0
    for r in rows.itertuples(index=False):
        mid = int(r.match_id)
        fid = int(r.fixture_id)

        existing = con.execute(
            "SELECT fetched_at FROM af_context WHERE match_id=?",
            [mid],
        ).fetchone()
        if existing and existing[0] is not None:
            try:
                if existing[0] >= stale_cut:
                    continue
            except Exception:
                pass

        bundle = fetch_fixture_bundle(fid)
        inj = fetch_injuries(fid)

        # count injuries by team (best-effort)
        home_inj = 0
        away_inj = 0
        try:
            resp = inj.get("response", []) or []
            # injuries response often includes team in each row
            # we just count total and split if we can
            teams = {}
            for it in resp:
                t = (it.get("team") or {}).get("id")
                teams.setdefault(t, 0)
                teams[t] += 1
            # if exactly two teams exist, assign deterministically
            if len(teams) == 2:
                vals = list(teams.values())
                home_inj, away_inj = int(vals[0]), int(vals[1])
            else:
                home_inj = int(len(resp))
                away_inj = 0
        except Exception:
            pass

        con.execute(
            "INSERT OR REPLACE INTO af_context VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                mid,
                fid,
                now,
                json.dumps(bundle),
                json.dumps(inj),
                int(home_inj),
                int(away_inj),
                None,
                None,
            ],
        )
        wrote += 1
        if verbose and wrote % 10 == 0:
            print(f"[af] context saved {wrote}/{len(rows)} ...", flush=True)

    if verbose:
        print(f"[af] context upserted: {wrote}", flush=True)
    return wrote
