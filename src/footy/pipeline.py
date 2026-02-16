from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import date, timedelta, datetime, timezone

import numpy as np
import pandas as pd

from footy.db import connect
from footy.config import settings
from footy.providers.football_data_org import fetch_matches_range, normalize_match
from footy.providers.news_gdelt import fetch_team_news
from footy.providers.fdcuk_history import DIV_MAP, season_codes_last_n, download_division_csv
from footy.normalize import canonical_team_name
from footy.models import elo
from footy.models.poisson import fit_poisson, expected_goals, outcome_probs
from footy.utils import outcome_label, compute_metrics

log = logging.getLogger(__name__)

MODEL_VERSION = "v1_elo_poisson"

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

def _features_for_match(con, poisson_state: dict, home: str, away: str):
    # base probs
    pE = elo.predict_probs(con, home, away)
    lam_h, lam_a = expected_goals(poisson_state, home, away)
    pP = outcome_probs(lam_h, lam_a)

    # rating diff (home advantage already baked into elo.predict_probs, but diff still useful)
    rh = elo.get_rating(con, home)
    ra = elo.get_rating(con, away)
    elo_diff = (rh - ra)

    X = [
        pE[0], pE[1], pE[2],
        pP[0], pP[1], pP[2],
        float(elo_diff),
        float(lam_h), float(lam_a), float(lam_h - lam_a),
    ]
    return X, pE, pP, lam_h, lam_a

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

def load_poisson() -> dict:
    con = connect()
    row = con.execute("SELECT value FROM poisson_state WHERE key='state'").fetchone()
    if not row:
        return {"teams": [], "attack": [], "defense": [], "home_adv": 0.0, "mu": 0.0}
    return json.loads(row[0])

def predict_upcoming(lookahead_days: int | None = None, verbose: bool = True) -> int:
    con = connect()
    s = settings()
    look = lookahead_days or s.lookahead_days

    cutoff_date = datetime.now(timezone.utc) + timedelta(days=look)
    df = con.execute(
        """SELECT match_id, home_team, away_team, utc_date, status
            FROM matches
            WHERE status IN ('SCHEDULED','TIMED')
              AND utc_date <= ?
            ORDER BY utc_date ASC""",
        [cutoff_date]
    ).df()

    if verbose:
        log.info("upcoming matches in next %d days: %d", look, len(df))

    poisson_state = load_poisson()

    n = 0
    for _, r in df.iterrows():
        mid = int(r["match_id"])
        home = r["home_team"]
        away = r["away_team"]

        X, pE, pP, lam_h, lam_a = _features_for_match(con, poisson_state, home, away)

        # --- v1 baseline (blend) ---
        p_home = 0.45 * pE[0] + 0.55 * pP[0]
        p_draw = 0.45 * pE[1] + 0.55 * pP[1]
        p_away = 0.45 * pE[2] + 0.55 * pP[2]
        ssum = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home / ssum, p_draw / ssum, p_away / ssum

        con.execute(
            """INSERT OR REPLACE INTO predictions
               (match_id, model_version, p_home, p_draw, p_away, eg_home, eg_away, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [mid, MODEL_VERSION, p_home, p_draw, p_away, lam_h, lam_a, "blend(elo,poisson)"]
        )
        n += 1

    return n

def backtest_metrics() -> dict:
    con = connect()
    df = con.execute(
        """SELECT p.match_id, p.p_home, p.p_draw, p.p_away,
                  m.home_goals, m.away_goals
           FROM predictions p
           JOIN matches m USING(match_id)
           WHERE p.model_version=? AND m.status='FINISHED'
             AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL""",
        [MODEL_VERSION]
    ).df()

    if df.empty:
        return {"n": 0, "logloss": None, "brier": None, "accuracy": None}

    P = df[["p_home", "p_draw", "p_away"]].to_numpy()
    y = np.array([outcome_label(int(hg), int(ag))
                  for hg, ag in zip(df["home_goals"], df["away_goals"])], dtype=int)

    m = compute_metrics(P, y)

    con.execute(
        "INSERT OR REPLACE INTO metrics(model_version, n_matches, logloss, brier, accuracy) VALUES (?, ?, ?, ?, ?)",
        [MODEL_VERSION, int(len(y)), m["logloss"], m["brier"], m["accuracy"]]
    )
    return {"n": int(len(y)), **m}


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
        outcome_prob = probs[outcome]
        predicted = max(range(3), key=lambda i: probs[i])

        ll = -math.log(max(outcome_prob, 1e-15))
        brier = sum((p - (1.0 if i == outcome else 0.0)) ** 2
                     for i, p in enumerate(probs))
        correct = predicted == outcome

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

def train_meta_model(days: int = 365, test_days: int = 28, verbose: bool = True) -> dict:
    """
    Train a multinomial logistic regression stacker (v2) WITHOUT mutating production Elo tables.
    Robust chronological split:
      - tries a "last test_days" split
      - if train/test too small, falls back to an 80/20 chronological split with minimum sizes
    """
    con = connect()
    history_cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    df = con.execute(
        """SELECT match_id, home_team, away_team, home_goals, away_goals, utc_date
              FROM matches
              WHERE status='FINISHED'
                AND home_goals IS NOT NULL AND away_goals IS NOT NULL
                AND utc_date >= ?
              ORDER BY utc_date ASC""",
        [history_cutoff]
    ).df()

    if df.empty or len(df) < 80:
        return {"error": "Not enough finished matches. Ingest more history (later we’ll add football-data.co.uk history to fix this permanently)."}

    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)

    min_train = 120
    min_test = 40

    # Attempt split by test_days
    cutoff = df["utc_date"].max() - pd.Timedelta(days=test_days)
    train = df[df["utc_date"] < cutoff].copy()
    test = df[df["utc_date"] >= cutoff].copy()

    # If too small, fall back to 80/20 split
    if len(train) < min_train or len(test) < min_test:
        split_idx = int(len(df) * 0.80)
        split_idx = max(min_train, min(split_idx, len(df) - min_test))
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        cutoff = test["utc_date"].min()

    if verbose:
        log.info("total finished: %d | train: %d | test: %d | cutoff: %s", len(df), len(train), len(test), cutoff)

    if len(train) < 50 or len(test) < 20:
        return {"error": "Still not enough data after fallback split. Increase finished match history."}

    # Fit Poisson on TRAIN only
    poisson_state = fit_poisson(train)

    # Elo simulated in-memory (dict) for proper sequential features
    DEFAULT_R = 1500.0
    ratings: dict[str, float] = {}

    def features_from_dict(home: str, away: str):
        pE = _elo_predict_dict(ratings, home, away)  # already defined below in pipeline.py
        lam_h, lam_a = expected_goals(poisson_state, home, away)
        pP = outcome_probs(lam_h, lam_a)
        elo_diff = float(ratings.get(home, DEFAULT_R) - ratings.get(away, DEFAULT_R))
        X = [
            pE[0], pE[1], pE[2],
            pP[0], pP[1], pP[2],
            elo_diff,
            float(lam_h), float(lam_a), float(lam_h - lam_a),
        ]
        return X

    # Build TRAIN features sequentially
    X_train, y_train = [], []
    for i, r in enumerate(train.itertuples(index=False), start=1):
        home = r.home_team; away = r.away_team
        hg = int(r.home_goals); ag = int(r.away_goals)

        X_train.append(features_from_dict(home, away))
        y_train.append(outcome_label(hg, ag))

        _elo_update_dict(ratings, home, away, hg, ag)

        if verbose and i % 100 == 0:
            log.debug("train features %d/%d…", i, len(train))

    import numpy as np
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=int)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler
    model = SkPipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=500, C=1.0)),
    ])
    model.fit(X_train, y_train)

    # Evaluate on TEST sequentially (ratings continue rolling forward)
    X_test, y_test = [], []
    for i, r in enumerate(test.itertuples(index=False), start=1):
        home = r.home_team; away = r.away_team
        hg = int(r.home_goals); ag = int(r.away_goals)

        X_test.append(features_from_dict(home, away))
        y_test.append(outcome_label(hg, ag))

        _elo_update_dict(ratings, home, away, hg, ag)

        if verbose and i % 50 == 0:
            log.debug("test features %d/%d…", i, len(test))

    X_test = np.array(X_test, dtype=float)
    y_test = np.array(y_test, dtype=int)

    P = model.predict_proba(X_test)
    m = compute_metrics(P, y_test)

    META_VERSION = "v2_meta_stack"
    import joblib
    from pathlib import Path
    model_path = Path("data/models") / f"{META_VERSION}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": [
        "pE_home","pE_draw","pE_away","pP_home","pP_draw","pP_away",
        "elo_diff","eg_home","eg_away","eg_diff",
    ]}, model_path)

    con.execute(
        "INSERT OR REPLACE INTO metrics(model_version, n_matches, logloss, brier, accuracy) VALUES (?, ?, ?, ?, ?)",
        [META_VERSION, int(len(y_test)), m["logloss"], m["brier"], m["accuracy"]]
    )

    out = {"n": int(len(y_test)), **m}
    if verbose:
        log.info("saved %s to data/models; result=%s", META_VERSION, out)
    return out

# --- Real test you can run immediately (time-split backtest) ---

# Use consolidated Elo core for all in-memory Elo operations
from footy.models.elo_core import elo_predict as _elo_predict_core, elo_update as _elo_update_core

def _elo_predict_dict(ratings: dict, home: str, away: str) -> tuple[float, float, float]:
    return _elo_predict_core(ratings, home, away, home_adv=60.0, draw_base=0.26)

def _elo_update_dict(ratings: dict, home: str, away: str, hg: int, ag: int):
    _elo_update_core(ratings, home, away, hg, ag, home_adv=60.0, k=20.0)

def backtest_time_split(days: int = 180, test_days: int = 14, verbose: bool = True) -> dict:
    """
    Proper chronological backtest:
    - train = older matches
    - test = most recent `test_days` of finished matches
    """
    con = connect()
    history_cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    df = con.execute(
        """SELECT home_team, away_team, home_goals, away_goals, utc_date
            FROM matches
            WHERE status='FINISHED'
              AND home_goals IS NOT NULL AND away_goals IS NOT NULL
              AND utc_date >= ?
            ORDER BY utc_date ASC""",
        [history_cutoff]
    ).df()

    if df.empty or len(df) < 50:
        return {"error": "Not enough finished matches in DB yet. Ingest more history or wait for results."}

    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)
    cutoff = df["utc_date"].max() - pd.Timedelta(days=test_days)

    train = df[df["utc_date"] < cutoff].copy()
    test = df[df["utc_date"] >= cutoff].copy()

    if verbose:
        log.info("total finished in window: %d", len(df))
        log.info("train: %d | test: %d | cutoff: %s", len(train), len(test), cutoff)

    if len(train) < 30 or len(test) < 10:
        return {"error": "Train/test split too small. Increase days or reduce test_days."}

    # Fit Poisson only on TRAIN
    poisson_state = fit_poisson(train)

    # Build Elo sequentially over TRAIN only
    ratings: dict[str, float] = {}
    for _, r in train.iterrows():
        _elo_update_dict(ratings, r["home_team"], r["away_team"], int(r["home_goals"]), int(r["away_goals"]))

    # Predict TEST
    P = []
    y = []
    for i, r in enumerate(test.itertuples(index=False), start=1):
        home = r.home_team
        away = r.away_team
        hg = int(r.home_goals)
        ag = int(r.away_goals)

        pE = _elo_predict_dict(ratings, home, away)
        lam_h, lam_a = expected_goals(poisson_state, home, away)
        pP = outcome_probs(lam_h, lam_a)

        p_home = 0.45 * pE[0] + 0.55 * pP[0]
        p_draw = 0.45 * pE[1] + 0.55 * pP[1]
        p_away = 0.45 * pE[2] + 0.55 * pP[2]
        ssum = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home / ssum, p_draw / ssum, p_away / ssum
        P.append([p_home, p_draw, p_away])

        y.append(outcome_label(hg, ag))

        if verbose and i % 25 == 0:
            log.debug("predicted %d/%d…", i, len(test))

        # Update Elo with the *observed* result as we roll forward (realistic sequential backtest)
        _elo_update_dict(ratings, home, away, hg, ag)

    P = np.array(P, dtype=float)
    y = np.array(y, dtype=int)

    result = {"n": int(len(y)), **compute_metrics(P, y)}
    if verbose:
        log.info("result: %s", result)
    return result
