from __future__ import annotations

import json
import hashlib
import pandas as pd

from footy.db import connect
from footy.normalize import canonical_team_name
from footy.providers.fdcuk_history import DIV_MAP, season_codes_last_n, download_division_csv

def _match_id(season_code: str, competition: str, utc_date, home: str, away: str) -> int:
    # Must match ingest-history deterministic key
    key = f"fdcuk|{season_code}|{competition}|{utc_date.date()}|{home}|{away}"
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    u = int.from_bytes(h, byteorder="big", signed=False) & 0x7FFFFFFFFFFFFFFF
    return u if u != 0 else 1

def _num(v):
    try:
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return float(v)
    except Exception:
        return None

def ingest_extras_fdcuk(n_seasons: int = 8, verbose: bool = True) -> int:
    """
    Upserts match_extras rows keyed by deterministic match_id (same as ingest-history).
    Uses finished matches only (requires FTHG/FTAG present).
    """
    con = connect()
    seasons = season_codes_last_n(n_seasons, include_current=True)

    divs = list(DIV_MAP.values())
    total_files = len(seasons) * len(divs)

    inserted = 0
    file_i = 0

    wanted_cols = ["B365H","B365D","B365A","HS","AS","HST","AST","HC","AC","HY","AY","HR","AR"]

    for season_code in seasons:
        for d in divs:
            file_i += 1
            if verbose:
                print(f"[extras] [{file_i}/{total_files}] {season_code}/{d.div}.csv", flush=True)

            try:
                df = download_division_csv(season_code, d.div)
            except Exception as e:
                if verbose:
                    print(f"[extras]   download failed: {e}", flush=True)
                continue

            required = {"Date","HomeTeam","AwayTeam","FTHG","FTAG"}
            if not required.issubset(df.columns):
                if verbose:
                    print(f"[extras]   missing required cols; skipping ({sorted(required - set(df.columns))})", flush=True)
                continue

            # Parse datetime
            dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            if "Time" in df.columns:
                tm = df["Time"].fillna("00:00").astype(str)
                dt = pd.to_datetime(df["Date"].astype(str) + " " + tm, dayfirst=True, errors="coerce")

            home = df["HomeTeam"].map(canonical_team_name)
            away = df["AwayTeam"].map(canonical_team_name)

            # finished only
            mask = dt.notna() & home.notna() & away.notna() & df["FTHG"].notna() & df["FTAG"].notna()
            if not mask.any():
                if verbose:
                    print("[extras]   no finished rows in this file (yet)", flush=True)
                continue

            dfm = df.loc[mask].copy()
            dfm["utc_date"] = dt.loc[mask].astype("datetime64[ns]")
            dfm["home_team"] = home.loc[mask]
            dfm["away_team"] = away.loc[mask]

            before = inserted

            # Iterate using safe column names (no _dt/_home/_away attributes)
            for r in dfm.itertuples(index=False):
                row = r._asdict()
                utc_date = row["utc_date"]
                home_team = row["home_team"]
                away_team = row["away_team"]

                mid = _match_id(season_code, d.competition, utc_date, home_team, away_team)

                def g(col):  # safe getter
                    return row.get(col, None)

                raw = {c: row.get(c) for c in (["Date","Time","HomeTeam","AwayTeam","FTHG","FTAG"] + wanted_cols) if c in row}

                con.execute(
                    """INSERT OR REPLACE INTO match_extras
                       (match_id, provider, competition, season_code, div_code,
                        b365h, b365d, b365a,
                        hs, as_, hst, ast,
                        hc, ac, hy, ay, hr, ar,
                        raw_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        int(mid),
                        "football-data.co.uk",
                        d.competition,
                        season_code,
                        d.div,
                        _num(g("B365H")), _num(g("B365D")), _num(g("B365A")),
                        _num(g("HS")), _num(g("AS")), _num(g("HST")), _num(g("AST")),
                        _num(g("HC")), _num(g("AC")), _num(g("HY")), _num(g("AY")), _num(g("HR")), _num(g("AR")),
                        json.dumps(raw, default=str),
                    ]
                )
                inserted += 1

            if verbose:
                print(f"[extras]   upserted {inserted - before}", flush=True)

    if verbose:
        print(f"[extras] total upserted: {inserted}", flush=True)
    return inserted
