from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

from footy.db import connect
from footy.normalize import canonical_team_name
from footy.providers.fdcuk_history import DIV_MAP
from footy.providers.fdcuk_fixtures import download_fixtures

def _num(v):
    try:
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return float(v)
    except Exception:
        return None

STOPWORDS = {
    "fc","cf","sc","ac","as","us","ss","rc","sv","afc",
    "club","racing","sporting","calcio",
    "futbol","fútbol","football",
    "de","del","da","di","la","le","los","las","der","den","the",
}

def _norm(s: str) -> str:
    s = canonical_team_name(s) or ""
    s = s.lower().strip()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in re.sub(r"\s+", " ", s).strip().split(" ") if t and t not in STOPWORDS]
    return " ".join(toks)

def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def ingest_upcoming_odds(verbose: bool = True) -> int:
    """
    Download fixtures.csv and attach odds to upcoming matches in matches table.

    Matching strategy:
      1) Exact match on (div, date, home_n, away_n)
      2) Exact match on (div, date±1, home_n, away_n)
      3) Exact match ignoring div on (date, home_n, away_n) (+ ±1 day)
      4) Fuzzy match within same (div, date±1): pick best combined similarity if confident
    """
    con = connect()
    fx = download_fixtures()

    need = {"Div", "Date", "HomeTeam", "AwayTeam"}
    if not need.issubset(fx.columns):
        raise RuntimeError(f"fixtures.csv missing columns: {sorted(need - set(fx.columns))}")

    # Parse date/time
    dt = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
    if "Time" in fx.columns:
        tm = fx["Time"].fillna("00:00").astype(str)
        dt = pd.to_datetime(fx["Date"].astype(str) + " " + tm, dayfirst=True, errors="coerce")

    fx = fx.assign(
        utc_date=dt,
        div=fx["Div"].astype(str),
        home=fx["HomeTeam"].map(canonical_team_name),
        away=fx["AwayTeam"].map(canonical_team_name),
    ).dropna(subset=["utc_date", "div", "home", "away"])

    fx["date_only"] = pd.to_datetime(fx["utc_date"]).dt.date
    fx["home_n"] = fx["home"].map(_norm)
    fx["away_n"] = fx["away"].map(_norm)

    fx["b365h"] = fx["B365H"].map(_num) if "B365H" in fx.columns else None
    fx["b365d"] = fx["B365D"].map(_num) if "B365D" in fx.columns else None
    fx["b365a"] = fx["B365A"].map(_num) if "B365A" in fx.columns else None

    # Build indexes
    idx_div = {}
    idx_any = {}
    bucket = {}  # (div, date) -> list of rows for fuzzy search

    for r in fx.itertuples(index=False):
        val = {"b365h": r.b365h, "b365d": r.b365d, "b365a": r.b365a, "div": r.div, "date": str(r.date_only)}
        idx_div[(r.div, r.date_only, r.home_n, r.away_n)] = val
        idx_any[(r.date_only, r.home_n, r.away_n)] = val
        bucket.setdefault((r.div, r.date_only), []).append((r.home_n, r.away_n, val))

    comp_to_div = {k: v.div for k, v in DIV_MAP.items()}

    up = con.execute(
        """SELECT match_id, utc_date, competition, home_team, away_team
           FROM matches
           WHERE status IN ('SCHEDULED','TIMED')
           ORDER BY utc_date ASC"""
    ).df()

    if up.empty:
        if verbose:
            print("[fixtures] no upcoming matches in DB", flush=True)
        return 0

    up["utc_date"] = pd.to_datetime(up["utc_date"], utc=True, errors="coerce")
    up = up.dropna(subset=["utc_date", "home_team", "away_team"])
    up["date_only"] = up["utc_date"].dt.date
    up["home_n"] = up["home_team"].map(_norm)
    up["away_n"] = up["away_team"].map(_norm)

    if verbose:
        print(f"[fixtures] fixtures rows={len(fx)} unique_div_keys={len(idx_div)} unique_any_keys={len(idx_any)}", flush=True)
        print("[fixtures] upcoming by competition:", flush=True)
        print(up.groupby("competition")["match_id"].count().to_string(), flush=True)

    matched = 0
    exact = div_pm = anydiv = anydiv_pm = fuzzy = 0
    unmatched_examples = []

    for r in up.itertuples(index=False):
        div = comp_to_div.get(r.competition)
        d0 = r.date_only
        cand_dates = [
            d0,
            (pd.Timestamp(d0) - pd.Timedelta(days=1)).date(),
            (pd.Timestamp(d0) + pd.Timedelta(days=1)).date(),
        ]

        found = None

        # 1/2 exact by div
        if div:
            for d in cand_dates:
                k = (div, d, r.home_n, r.away_n)
                if k in idx_div:
                    found = idx_div[k]
                    if d == d0: exact += 1
                    else: div_pm += 1
                    break

        # 3 exact ignoring div
        if found is None:
            for d in cand_dates:
                k = (d, r.home_n, r.away_n)
                if k in idx_any:
                    found = idx_any[k]
                    if d == d0: anydiv += 1
                    else: anydiv_pm += 1
                    break

        # 4 fuzzy within same div/date±1
        if found is None and div:
            best = None
            best_score = 0.0
            second = 0.0

            for d in cand_dates:
                rows = bucket.get((div, d), [])
                for h_n, a_n, val in rows:
                    s1 = _ratio(r.home_n, h_n)
                    s2 = _ratio(r.away_n, a_n)
                    score = (s1 + s2) / 2.0
                    if score > best_score:
                        second = best_score
                        best_score = score
                        best = val
                    elif score > second:
                        second = score

            # accept only if confident and not ambiguous
            if best is not None and best_score >= 0.82 and (best_score - second) >= 0.04:
                found = best
                fuzzy += 1

        if found is None:
            if len(unmatched_examples) < 12:
                unmatched_examples.append((r.competition, str(r.date_only), r.home_team, r.away_team, r.home_n, r.away_n))
            continue

        con.execute(
            """INSERT OR REPLACE INTO match_extras
               (match_id, provider, competition, season_code, div_code, b365h, b365d, b365a, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                int(r.match_id),
                "football-data.co.uk-fixtures",
                r.competition,
                None,
                found.get("div") or div,
                found.get("b365h"), found.get("b365d"), found.get("b365a"),
                json.dumps({"source": "fixtures.csv", "matched_date": found.get("date"), "match_type": "fuzzy" if found and found.get("date") else "exact"}, default=str),
            ]
        )
        matched += 1

    if verbose:
        print(f"[fixtures] attached odds rows={matched} (exact={exact}, div±1={div_pm}, anydiv_exact={anydiv}, anydiv±1={anydiv_pm}, fuzzy={fuzzy})", flush=True)
        if unmatched_examples:
            print("[fixtures] examples unmatched (competition, date, home, away, home_n, away_n):", flush=True)
            for ex in unmatched_examples:
                print("  -", ex, flush=True)

    return matched
