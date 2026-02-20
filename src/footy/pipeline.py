from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, timedelta, datetime, timezone

import pandas as pd

from footy.db import connect
from footy.config import settings
from footy.providers.football_data_org import fetch_matches_range, normalize_match
from footy.providers.news_gdelt import fetch_team_news
from footy.providers.fdcuk_history import DIV_MAP, season_codes_last_n, download_division_csv
from footy.normalize import canonical_team_name
from footy.models import elo
from footy.models.poisson import fit_poisson
from footy.utils import outcome_label, score_prediction

log = logging.getLogger(__name__)

# Preferred provider for finished-match training after history is ingested
TRAIN_PROVIDER = "football-data.co.uk"

def reset_states(verbose: bool = True) -> None:
    """
    Use this once after ingesting large history, to rebuild Elo/Poisson cleanly.
    Does NOT delete matches/news.
    """
    con = connect()
    for table in ["predictions", "metrics", "elo_state", "elo_applied", "poisson_state"]:
        try:
            con.execute(f"DELETE FROM {table}")
        except Exception as e:
            log.debug("reset_states: %s — %s", table, e)
    if verbose:
        log.info("cleared predictions/metrics/elo_state/elo_applied/poisson_state")

def ingest_history_fdcuk(n_seasons: int = 8, verbose: bool = True) -> int:
    """
    Downloads football-data.co.uk historical season CSVs for tracked top leagues,
    inserts finished matches only (rows with FTHG/FTAG).
    """
    con = connect()
    s = settings()

    # Map tracked competitions -> football-data.co.uk division codes
    divs = []
    for comp in s.tracked_competitions:
        if comp in DIV_MAP:
            divs.append(DIV_MAP[comp])

    if not divs:
        raise RuntimeError("No tracked competitions map to football-data.co.uk divisions. Use PL,PD,SA,BL1,FL1.")

    seasons = season_codes_last_n(n_seasons, include_current=True)
    total_files = len(seasons) * len(divs)

    if verbose:
        log.info("seasons=%s", seasons)
        log.info("divisions=%s (files=%d)", [d.div for d in divs], total_files)

    inserted = 0
    file_i = 0
    for season_code in seasons:
        for d in divs:
            file_i += 1
            if verbose:
                log.debug("[%d/%d] downloading %s/%s.csv", file_i, total_files, season_code, d.div)

            try:
                df = download_division_csv(season_code, d.div)
            except Exception as e:
                if verbose:
                    log.warning("failed %s/%s: %s", season_code, d.div, e)
                continue

            # minimal column set
            # common columns: Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG
            cols = {c: c for c in df.columns}
            need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
            if not all(n in cols for n in need):
                if verbose:
                    log.warning("missing required columns; got=%s skipping", list(df.columns)[:12])
                continue

            # parse datetime
            # Date examples: 05/08/2022, 05/08/22, 2022-08-05 ; Time optional: 20:00
            date_col = df["Date"].astype(str)
            if "Time" in df.columns:
                date_col = date_col + " " + df["Time"].fillna("00:00").astype(str)
            dt = pd.to_datetime(date_col, dayfirst=True, format="mixed", errors="coerce")

            df2 = pd.DataFrame({
                "utc_date": dt,
                "home_team": df["HomeTeam"].map(canonical_team_name),
                "away_team": df["AwayTeam"].map(canonical_team_name),
                "home_goals": pd.to_numeric(df["FTHG"], errors="coerce"),
                "away_goals": pd.to_numeric(df["FTAG"], errors="coerce"),
            }).dropna(subset=["utc_date","home_team","away_team","home_goals","away_goals"])

            if df2.empty:
                if verbose:
                    log.debug("no finished rows in this file (yet)")
                continue

            # Deterministic 64-bit match_id via blake2b
            def make_id(row) -> int:
                key = f"fdcuk|{season_code}|{d.competition}|{row.utc_date.date()}|{row.home_team}|{row.away_team}"
                h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
                u = int.from_bytes(h, byteorder="big", signed=False)
                u = u & 0x7FFFFFFFFFFFFFFF  # fit signed BIGINT
                return u if u != 0 else 1

            raw_json = None  # keep null; can add full row later if needed

            before = inserted
            for r in df2.itertuples(index=False):
                mid = make_id(r)
                con.execute(
                    """INSERT OR REPLACE INTO matches
                       (match_id, provider, competition, season, utc_date, status,
                        home_team, away_team, home_goals, away_goals, raw_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        mid,
                        "football-data.co.uk",
                        d.competition,
                        int("20" + season_code[:2]),
                        r.utc_date.to_pydatetime(),
                        "FINISHED",
                        r.home_team,
                        r.away_team,
                        int(r.home_goals),
                        int(r.away_goals),
                        raw_json,
                    ]
                )
                inserted += 1

            if verbose:
                log.debug("inserted %d finished matches", inserted - before)

    if verbose:
        log.info("total inserted/updated: %d", inserted)
    return inserted

def _upsert_match_extras_from_api(con, match_id: int, competition: str | None, season: int | None, extras: dict):
    """Upsert unfolded data from football-data.org into match_extras."""
    if not extras:
        return
    # Build dynamic SET clause from available extras
    allowed = {
        "hthg", "htag", "hy", "ay", "hr", "ar",
        "formation_home", "formation_away", "lineup_home", "lineup_away",
    }
    to_set = {k: v for k, v in extras.items() if k in allowed and v is not None}
    if not to_set:
        return

    cols = ["match_id", "provider", "competition", "season_code"] + list(to_set.keys())
    placeholders = ", ".join(["?"] * len(cols))
    updates = ", ".join(f"{c}=excluded.{c}" for c in to_set.keys())
    vals = [match_id, "football-data.org", competition, str(season) if season else None] + list(to_set.values())

    sql = (
        f"INSERT INTO match_extras ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (match_id) DO UPDATE SET {updates}"
    )
    con.execute(sql, vals)


def ingest(days_back: int = 30, days_forward: int = 7, chunk_days: int = 10, verbose: bool = True) -> int:
    con = connect()
    d0 = date.today() - timedelta(days=days_back)
    d1 = date.today() + timedelta(days=days_forward)

    if verbose:
        log.info("fetching matches %s → %s (chunk_days=%d)", d0, d1, chunk_days)

    matches = fetch_matches_range(d0, d1, chunk_days=chunk_days, verbose=verbose)
    if verbose:
        log.info("fetched %d raw matches; inserting…", len(matches))

    s = settings()
    n = 0
    kept = 0

    for i, m in enumerate(matches, start=1):
        nm = normalize_match(m)

        if nm["competition"] and nm["competition"] not in s.tracked_competitions:
            continue

        kept += 1
        con.execute(
            """INSERT OR REPLACE INTO matches
               (match_id, provider, competition, season, utc_date, status, home_team, away_team, home_goals, away_goals, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                nm["match_id"], nm["provider"], nm["competition"], nm["season"],
                nm["utc_date"], nm["status"], nm["home_team"], nm["away_team"],
                nm["home_goals"], nm["away_goals"], nm["raw_json"]
            ],
        )

        # Persist unfolded extras (lineups, formations, bookings, HT scores)
        extras = nm.get("_extras", {})
        if extras:
            _upsert_match_extras_from_api(con, nm["match_id"], nm["competition"], nm["season"], extras)

        n += 1

        if verbose and i % 100 == 0:
            log.debug("processed %d/%d raw matches…", i, len(matches))

    if verbose:
        log.info("inserted/updated %d matches (kept=%d after competition filter)", n, kept)

    return n

def update_elo_from_finished(verbose: bool = True) -> int:
    """
    Applies Elo updates ONLY ONCE per finished match (tracks applied match_ids).
    """
    con = connect()
    rows = con.execute(
        """SELECT m.match_id, m.home_team, m.away_team, m.home_goals, m.away_goals
           FROM matches m
           LEFT JOIN elo_applied ea ON ea.match_id = m.match_id
           WHERE m.status='FINISHED'
             AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
             AND ea.match_id IS NULL
           ORDER BY m.utc_date ASC"""
    ).fetchall()

    if verbose:
        log.info("new finished matches to apply: %d", len(rows))

    count = 0
    for i, (mid, home, away, hg, ag) in enumerate(rows, start=1):
        elo.update_from_match(con, home, away, int(hg), int(ag))
        con.execute("INSERT OR IGNORE INTO elo_applied(match_id) VALUES (?)", [int(mid)])
        count += 1

        if verbose and (i % 500 == 0 or i == len(rows)):
            log.debug("applied %d/%d…", i, len(rows))

    return count

def refit_poisson(verbose: bool = True) -> dict:
    con = connect()
    df = con.execute(
        """SELECT home_team, away_team, home_goals, away_goals, utc_date
           FROM matches
           WHERE status='FINISHED' AND home_goals IS NOT NULL AND away_goals IS NOT NULL"""
    ).df()

    if verbose:
        log.info("fitting on finished matches: %d", len(df))

    state = fit_poisson(df)
    con.execute("INSERT OR REPLACE INTO poisson_state(key, value) VALUES ('state', ?)", [json.dumps(state)])

    if verbose:
        log.info("fit complete; teams=%d", len(state.get('teams', [])))

    return state

# ---------------------------------------------------------------------------
# Prediction scoring — saves individual prediction outcomes so the model
# can track its own performance over time and detect degradation.
# ---------------------------------------------------------------------------

def score_finished_predictions(verbose: bool = True) -> dict:
    """Score all finished predictions that haven't been scored yet.

    For each (match_id, model_version) pair where the match is FINISHED
    but no row exists in prediction_scores, computes:
      - outcome  (0=Home, 1=Draw, 2=Away)
      - logloss  (-log of the probability assigned to the actual outcome)
      - brier    (mean squared error across the probability vector)
      - correct  (True if highest probability matched the outcome)
      - goals_mae (MAE of predicted vs actual goals)
      - btts_correct (was BTTS prediction correct?)
      - ou25_correct (was Over 2.5 prediction correct?)
      - score_correct (was exact score correct?)
    """
    con = connect()
    rows = con.execute("""
        SELECT m.match_id, m.home_goals, m.away_goals,
               p.model_version, p.p_home, p.p_draw, p.p_away,
               p.eg_home, p.eg_away, p.notes
        FROM matches m
        JOIN predictions p ON m.match_id = p.match_id
        WHERE m.status = 'FINISHED'
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM prediction_scores ps
              WHERE ps.match_id = m.match_id AND ps.model_version = p.model_version
          )
        ORDER BY m.utc_date ASC
    """).fetchall()

    if not rows:
        if verbose:
            log.info("no new finished predictions to score")
        return {"scored": 0}

    scored = 0
    total_ll = 0.0
    total_correct = 0
    total_btts_correct = 0
    total_ou25_correct = 0
    total_score_correct = 0
    n_btts = 0
    n_ou25 = 0
    n_score = 0

    for match_id, hg, ag, model_version, ph, pd_, pa, eg_h, eg_a, notes in rows:
        outcome = outcome_label(int(hg), int(ag))
        probs = [float(ph), float(pd_), float(pa)]
        s = score_prediction(probs, outcome)
        ll = s["logloss"]
        brier = s["brier"]
        correct = s["correct"]

        # Goal prediction accuracy
        goals_mae = None
        eg_home_v = float(eg_h) if eg_h is not None else None
        eg_away_v = float(eg_a) if eg_a is not None else None
        if eg_home_v is not None and eg_away_v is not None:
            goals_mae = (abs(eg_home_v - hg) + abs(eg_away_v - ag)) / 2.0

        # Parse notes JSON for BTTS, O2.5, predicted score
        p_btts = None
        p_o25 = None
        pred_h = None
        pred_a = None
        btts_ok = None
        ou25_ok = None
        score_ok = None

        if notes:
            try:
                nj = json.loads(notes) if isinstance(notes, str) else {}
            except Exception:
                nj = {}

            p_btts_v = nj.get("btts")
            p_o25_v = nj.get("o25")
            pred_score = nj.get("predicted_score")

            # BTTS scoring
            if p_btts_v is not None:
                p_btts = float(p_btts_v)
                actual_btts = (hg > 0) and (ag > 0)
                predicted_btts = p_btts >= 0.5
                btts_ok = predicted_btts == actual_btts

            # Over 2.5 scoring
            if p_o25_v is not None:
                p_o25 = float(p_o25_v)
                actual_o25 = (hg + ag) > 2
                predicted_o25 = p_o25 >= 0.5
                ou25_ok = predicted_o25 == actual_o25

            # Exact score scoring
            if pred_score and isinstance(pred_score, (list, tuple)) and len(pred_score) == 2:
                pred_h = int(pred_score[0])
                pred_a = int(pred_score[1])
                score_ok = (pred_h == hg) and (pred_a == ag)

        con.execute("""
            INSERT OR REPLACE INTO prediction_scores
            (match_id, model_version, outcome, logloss, brier, correct,
             goals_mae, eg_home, eg_away,
             btts_correct, ou25_correct, score_correct,
             p_btts, p_o25, predicted_score_h, predicted_score_a)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [match_id, model_version, outcome, ll, brier, correct,
              goals_mae, eg_home_v, eg_away_v,
              btts_ok, ou25_ok, score_ok,
              p_btts, p_o25, pred_h, pred_a])
        scored += 1
        total_ll += ll
        total_correct += int(correct)
        if btts_ok is not None:
            total_btts_correct += int(btts_ok)
            n_btts += 1
        if ou25_ok is not None:
            total_ou25_correct += int(ou25_ok)
            n_ou25 += 1
        if score_ok is not None:
            total_score_correct += int(score_ok)
            n_score += 1

    # Update aggregate metrics table
    for mv in set(r[3] for r in rows):
        agg = con.execute("""
            SELECT COUNT(*), AVG(logloss), AVG(brier),
                   SUM(CASE WHEN correct THEN 1 ELSE 0 END)
            FROM prediction_scores WHERE model_version = ?
        """, [mv]).fetchone()
        if agg and agg[0] > 0:
            n, avg_ll, avg_br, n_correct = agg
            con.execute(
                "INSERT OR REPLACE INTO metrics(model_version, n_matches, logloss, brier, accuracy) VALUES (?, ?, ?, ?, ?)",
                [mv, int(n), float(avg_ll), float(avg_br), float(n_correct) / int(n)]
            )

    if verbose:
        avg_ll = total_ll / scored if scored else 0
        acc = total_correct / scored if scored else 0
        parts = [f"scored {scored} predictions — logloss={avg_ll:.4f} accuracy={acc:.1%}"]
        if n_btts:
            parts.append(f"BTTS={total_btts_correct / n_btts:.1%}({n_btts})")
        if n_ou25:
            parts.append(f"O2.5={total_ou25_correct / n_ou25:.1%}({n_ou25})")
        if n_score:
            parts.append(f"ExactScore={total_score_correct / n_score:.1%}({n_score})")
        log.info(" | ".join(parts))

    return {"scored": scored, "logloss": total_ll / scored if scored else 0,
            "accuracy": total_correct / scored if scored else 0,
            "btts_accuracy": total_btts_correct / n_btts if n_btts else None,
            "ou25_accuracy": total_ou25_correct / n_ou25 if n_ou25 else None,
            "score_accuracy": total_score_correct / n_score if n_score else None}


def ingest_news_for_teams(days_back: int = 2, max_records: int = 10) -> int:
    con = connect()
    s = settings()

    lookahead_cutoff = datetime.now(timezone.utc) + timedelta(days=s.lookahead_days)
    teams = con.execute(
        """SELECT DISTINCT home_team AS team FROM matches
              WHERE status IN ('SCHEDULED','TIMED')
                AND utc_date <= ?
            UNION
            SELECT DISTINCT away_team AS team FROM matches
              WHERE status IN ('SCHEDULED','TIMED')
                AND utc_date <= ?""",
        [lookahead_cutoff, lookahead_cutoff]
    ).fetchall()

    teams = [t for (t,) in teams if t]
    total = len(teams)
    if total == 0:
        log.warning("No upcoming teams found. Run: footy update")
        return 0

    def _parse_seendate(v):
        if v is None:
            return None
        if isinstance(v, str):
            s2 = v.strip()
            try:
                return datetime.strptime(s2, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                try:
                    return pd.to_datetime(s2, utc=True).to_pydatetime()
                except Exception:
                    return None
        try:
            return pd.to_datetime(v, utc=True).to_pydatetime()
        except Exception:
            return None

    cooldown_hours = 12
    now = datetime.now(timezone.utc)
    max_teams_per_run = min(12, total)

    n = 0
    fetched = 0
    for i, team in enumerate(teams, start=1):
        last = con.execute("SELECT MAX(fetched_at) FROM news WHERE team=?", [team]).fetchone()[0]
        if last is not None:
            try:
                last_dt = pd.to_datetime(last, utc=True).to_pydatetime()
                if (now - last_dt).total_seconds() < cooldown_hours * 3600:
                    continue
            except Exception:
                pass

        log.debug("[%d/%d] Fetching: %s", i, total, team)

        df = fetch_team_news(team, days_back=days_back, max_records=max_records)
        if df is None or df.empty:
            fetched += 1
            if fetched >= max_teams_per_run:
                break
            continue

        for _, r in df.iterrows():
            sd = _parse_seendate(r.get("seendate"))
            con.execute(
                """INSERT OR IGNORE INTO news(team, seendate, title, url, domain, tone, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [team, sd, r.get("title"), r.get("url"), r.get("domain"), None, "gdelt"]
            )
            n += 1

        fetched += 1
        if fetched >= max_teams_per_run:
            break

    return n
