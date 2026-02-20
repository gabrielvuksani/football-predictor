from __future__ import annotations

import json
import hashlib
import logging
import pandas as pd

from footy.db import connect
from footy.normalize import canonical_team_name
from footy.providers.fdcuk_history import DIV_MAP, season_codes_last_n, download_division_csv
from footy.utils import safe_num as _num

log = logging.getLogger(__name__)

def _match_id(season_code: str, competition: str, utc_date, home: str, away: str) -> int:
    # Must match ingest-history deterministic key
    key = f"fdcuk|{season_code}|{competition}|{utc_date.date()}|{home}|{away}"
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    u = int.from_bytes(h, byteorder="big", signed=False) & 0x7FFFFFFFFFFFFFFF
    return u if u != 0 else 1

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

    # Extended column list: 1X2 odds (opening + closing + Pinnacle + avg + max),
    # O/U 2.5, Asian Handicap, half-time scores, basic stats
    wanted_cols = [
        # B365 opening
        "B365H","B365D","B365A",
        # B365 closing
        "B365CH","B365CD","B365CA",
        # Pinnacle (sharpest bookmaker)
        "PSH","PSD","PSA",
        # Market average & max
        "AvgH","AvgD","AvgA",
        "MaxH","MaxD","MaxA",
        # Over/Under 2.5
        "B365>2.5","B365<2.5",
        "Avg>2.5","Avg<2.5",
        "Max>2.5","Max<2.5",
        # Asian Handicap
        "B365AHH","B365AH","B365AHA",
        # Half-time (goals)
        "HTHG","HTAG",
        # Match stats
        "HS","AS","HST","AST","HC","AC","HY","AY","HR","AR",
    ]

    for season_code in seasons:
        for d in divs:
            file_i += 1
            if verbose:
                log.debug("[%d/%d] %s/%s.csv", file_i, total_files, season_code, d.div)

            try:
                df = download_division_csv(season_code, d.div)
            except Exception as e:
                if verbose:
                    log.warning("download failed: %s", e)
                continue

            required = {"Date","HomeTeam","AwayTeam","FTHG","FTAG"}
            if not required.issubset(df.columns):
                if verbose:
                    log.warning("missing required cols; skipping %s", sorted(required - set(df.columns)))
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
                    log.debug("no finished rows in this file (yet)")
                continue

            dfm = df.loc[mask].copy()
            dfm["utc_date"] = dt.loc[mask].astype("datetime64[ns]")
            dfm["home_team"] = home.loc[mask]
            dfm["away_team"] = away.loc[mask]

            before = inserted

            # Use iterrows() — itertuples()._asdict() mangles cols with >/<
            for _, row in dfm.iterrows():
                utc_date = row["utc_date"]
                home_team = row["home_team"]
                away_team = row["away_team"]

                mid = _match_id(season_code, d.competition, utc_date, home_team, away_team)

                def g(col):  # safe getter — preserves original column names
                    v = row.get(col)
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    return v

                raw = {}
                for c in (["Date","Time","HomeTeam","AwayTeam","FTHG","FTAG"] + wanted_cols):
                    if c in row.index:
                        v = row[c]
                        raw[c] = None if (isinstance(v, float) and pd.isna(v)) else v

                con.execute(
                    """INSERT OR REPLACE INTO match_extras
                       (match_id, provider, competition, season_code, div_code,
                        b365h, b365d, b365a,
                        b365ch, b365cd, b365ca,
                        psh, psd, psa,
                        avgh, avgd, avga,
                        maxh, maxd, maxa,
                        b365_o25, b365_u25,
                        avg_o25, avg_u25,
                        max_o25, max_u25,
                        b365ahh, b365ahha, b365ahaw,
                        hthg, htag,
                        hs, as_, hst, ast,
                        hc, ac, hy, ay, hr, ar,
                        raw_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        int(mid),
                        "football-data.co.uk",
                        d.competition,
                        season_code,
                        d.div,
                        _num(g("B365H")), _num(g("B365D")), _num(g("B365A")),
                        _num(g("B365CH")), _num(g("B365CD")), _num(g("B365CA")),
                        _num(g("PSH")), _num(g("PSD")), _num(g("PSA")),
                        _num(g("AvgH")), _num(g("AvgD")), _num(g("AvgA")),
                        _num(g("MaxH")), _num(g("MaxD")), _num(g("MaxA")),
                        _num(g("B365>2.5")), _num(g("B365<2.5")),
                        _num(g("Avg>2.5")), _num(g("Avg<2.5")),
                        _num(g("Max>2.5")), _num(g("Max<2.5")),
                        _num(g("B365AHH")), _num(g("B365AH")), _num(g("B365AHA")),
                        _num(g("HTHG")), _num(g("HTAG")),
                        _num(g("HS")), _num(g("AS")), _num(g("HST")), _num(g("AST")),
                        _num(g("HC")), _num(g("AC")), _num(g("HY")), _num(g("AY")), _num(g("HR")), _num(g("AR")),
                        json.dumps(raw, default=str),
                    ]
                )
                inserted += 1

            if verbose:
                log.debug("upserted %d", inserted - before)

    if verbose:
        log.info("total upserted: %d", inserted)
    return inserted
