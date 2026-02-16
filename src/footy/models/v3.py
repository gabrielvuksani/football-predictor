from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

from footy.normalize import canonical_team_name
from footy.config import settings
from footy.models.poisson import fit_poisson, expected_goals, outcome_probs
from footy.models.elo_core import (
    elo_expected as _elo_expected,
    elo_predict_with_diff as _elo_predict_with_diff,
    elo_update,
)

V3_MODEL_VERSION = "v3_gbdt_form"
MODEL_PATH = Path("data/models") / f"{V3_MODEL_VERSION}.joblib"

# ----------------------------
# helpers
# ----------------------------
@dataclass
class TeamState:
    last_date: pd.Timestamp | None = None
    pts5: list[int] = None
    pts10: list[int] = None
    gf5: list[int] = None
    ga5: list[int] = None
    gf10: list[int] = None
    ga10: list[int] = None
    shots_for5: list[float] = None
    shots_against5: list[float] = None
    corners_for5: list[float] = None
    corners_against5: list[float] = None

    def __post_init__(self):
        self.pts5 = self.pts5 or []
        self.pts10 = self.pts10 or []
        self.gf5 = self.gf5 or []
        self.ga5 = self.ga5 or []
        self.gf10 = self.gf10 or []
        self.ga10 = self.ga10 or []
        self.shots_for5 = self.shots_for5 or []
        self.shots_against5 = self.shots_against5 or []
        self.corners_for5 = self.corners_for5 or []
        self.corners_against5 = self.corners_against5 or []

def _push(buf: list, v, n: int):
    buf.append(v)
    if len(buf) > n:
        del buf[0]

def _mean(buf: list[float] | list[int]) -> float:
    return float(np.mean(buf)) if buf else 0.0

def _to_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None

def implied_probs_from_odds(h, d, a):
    h = _to_float(h); d = _to_float(d); a = _to_float(a)
    if not h or not d or not a:
        return (0.0, 0.0, 0.0, 0.0)
    ih, id_, ia = 1.0 / h, 1.0 / d, 1.0 / a
    s = ih + id_ + ia
    return (ih / s, id_ / s, ia / s, 1.0)

def elo_predict(ratings: dict[str, float], home: str, away: str):
    ph, pd_, pa, elo_diff = _elo_predict_with_diff(
        ratings, home, away, home_adv=60.0, draw_base=0.26)
    return (ph, pd_, pa, elo_diff)

FEATURES = [
    "comp_id",
    "elo_diff",
    "pE_home","pE_draw","pE_away",
    "lam_h","lam_a","lam_diff",
    "pP_home","pP_draw","pP_away",
    "rest_home","rest_away","rest_diff",
    "pts5_home","pts5_away","pts10_home","pts10_away",
    "gd5_home","gd5_away","gd10_home","gd10_away",
    "shots_for5_home","shots_for5_away","shots_against5_home","shots_against5_away",
    "corn_for5_home","corn_for5_away","corn_against5_home","corn_against5_away",
    "odds_p_home","odds_p_draw","odds_p_away","has_odds",
]

def _build_comp_map(df: pd.DataFrame) -> dict[str, int]:
    comps = sorted([c for c in df["competition"].dropna().unique().tolist() if c])
    return {c: i for i, c in enumerate(comps)}

def build_sequential_Xy(df: pd.DataFrame, poisson_state: dict, comp_map: dict[str, int]):
    """
    df must be sorted by utc_date asc.
    Returns X, y, dates where dates is numpy datetime64 array.
    """
    states: dict[str, TeamState] = {}
    ratings: dict[str, float] = {}

    X, y, dates = [], [], []

    for r in df.itertuples(index=False):
        dt = r.utc_date
        comp = r.competition or ""
        home = canonical_team_name(r.home_team)
        away = canonical_team_name(r.away_team)
        hg = int(r.home_goals)
        ag = int(r.away_goals)

        sh = states.get(home) or TeamState()
        sa = states.get(away) or TeamState()
        states[home] = sh
        states[away] = sa

        rest_h = 7.0 if sh.last_date is None else max(0.0, (dt - sh.last_date).total_seconds() / 86400.0)
        rest_a = 7.0 if sa.last_date is None else max(0.0, (dt - sa.last_date).total_seconds() / 86400.0)

        pE_h, pE_d, pE_a, elo_diff = elo_predict(ratings, home, away)
        lam_h, lam_a = expected_goals(poisson_state, home, away)
        pP_h, pP_d, pP_a = outcome_probs(lam_h, lam_a)

        odds_h, odds_d, odds_a, has_odds = implied_probs_from_odds(r.b365h, r.b365d, r.b365a)

        row = [
            float(comp_map.get(comp, -1)),
            float(elo_diff),
            float(pE_h), float(pE_d), float(pE_a),
            float(lam_h), float(lam_a), float(lam_h - lam_a),
            float(pP_h), float(pP_d), float(pP_a),
            float(rest_h), float(rest_a), float(rest_h - rest_a),
            _mean(sh.pts5), _mean(sa.pts5), _mean(sh.pts10), _mean(sa.pts10),
            _mean(sh.gf5) - _mean(sh.ga5), _mean(sa.gf5) - _mean(sa.ga5),
            _mean(sh.gf10) - _mean(sh.ga10), _mean(sa.gf10) - _mean(sa.ga10),
            _mean(sh.shots_for5), _mean(sa.shots_for5), _mean(sh.shots_against5), _mean(sa.shots_against5),
            _mean(sh.corners_for5), _mean(sa.corners_for5), _mean(sh.corners_against5), _mean(sa.corners_against5),
            float(odds_h), float(odds_d), float(odds_a), float(has_odds),
        ]
        X.append(row)

        if hg > ag: yy = 0
        elif hg == ag: yy = 1
        else: yy = 2
        y.append(yy)

        dates.append(dt.to_datetime64())  # IMPORTANT: datetime64 for safe masking

        # update AFTER features
        sh.last_date = dt
        sa.last_date = dt

        if hg > ag: ph_pts, pa_pts = 3, 0
        elif hg == ag: ph_pts, pa_pts = 1, 1
        else: ph_pts, pa_pts = 0, 3

        _push(sh.pts5, ph_pts, 5); _push(sa.pts5, pa_pts, 5)
        _push(sh.pts10, ph_pts, 10); _push(sa.pts10, pa_pts, 10)

        _push(sh.gf5, hg, 5); _push(sh.ga5, ag, 5)
        _push(sa.gf5, ag, 5); _push(sa.ga5, hg, 5)

        _push(sh.gf10, hg, 10); _push(sh.ga10, ag, 10)
        _push(sa.gf10, ag, 10); _push(sa.ga10, hg, 10)

        hs = _to_float(r.hs) or 0.0
        as_ = _to_float(r.as_) or 0.0
        hc = _to_float(r.hc) or 0.0
        ac = _to_float(r.ac) or 0.0

        _push(sh.shots_for5, hs, 5); _push(sh.shots_against5, as_, 5)
        _push(sa.shots_for5, as_, 5); _push(sa.shots_against5, hs, 5)

        _push(sh.corners_for5, hc, 5); _push(sh.corners_against5, ac, 5)
        _push(sa.corners_for5, ac, 5); _push(sa.corners_against5, hc, 5)

        elo_update(ratings, home, away, hg, ag)

    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(dates, dtype="datetime64[ns]")

def train_and_save(con, days: int = 3650, test_days: int = 28, verbose: bool = True) -> dict:
    df = con.execute(
        f"""SELECT m.utc_date, m.competition, m.home_team, m.away_team, m.home_goals, m.away_goals,
                   e.b365h, e.b365d, e.b365a, e.hs, e.as_, e.hc, e.ac
            FROM matches m
            LEFT JOIN match_extras e ON e.match_id=m.match_id
            WHERE m.status='FINISHED'
              AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
              AND m.utc_date >= (CURRENT_TIMESTAMP - INTERVAL {int(days)} DAY)
            ORDER BY m.utc_date ASC"""
    ).df()

    if df.empty or len(df) < 400:
        return {"error": f"Not enough finished matches for v3 (got {len(df)})."}

    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)

    cutoff = df["utc_date"].max() - pd.Timedelta(days=int(test_days))
    train_df = df[df["utc_date"] < cutoff]
    test_df = df[df["utc_date"] >= cutoff]
    if len(train_df) < 600 or len(test_df) < 200:
        split_idx = int(len(df) * 0.8)
        split_idx = max(600, min(split_idx, len(df) - 200))
        cutoff = df.iloc[split_idx]["utc_date"]

    if verbose:
        print(f"[v3] total={len(df)} train={int((df['utc_date'] < cutoff).sum())} test={int((df['utc_date'] >= cutoff).sum())} cutoff={cutoff}", flush=True)

    # poisson on TRAIN for eval
    poisson_train = fit_poisson(df[df["utc_date"] < cutoff].copy())

    comp_map = _build_comp_map(df)

    # Build sequential features once (state carries through)
    X_all, y_all, dates = build_sequential_Xy(df, poisson_train, comp_map)

    test_mask = dates >= cutoff.to_datetime64()
    Xtr, ytr = X_all[~test_mask], y_all[~test_mask]
    Xte, yte = X_all[test_mask], y_all[test_mask]

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=500,
        l2_regularization=0.2,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(Xtr, ytr)
    P = model.predict_proba(Xte)

    eps = 1e-12
    logloss = float(np.mean(-np.log(P[np.arange(len(yte)), yte] + eps)))
    Y = np.zeros_like(P); Y[np.arange(len(yte)), yte] = 1.0
    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
    acc = float(np.mean(np.argmax(P, axis=1) == yte))

    # Deploy refit on ALL data
    poisson_all = fit_poisson(df.copy())
    X_deploy, y_deploy, _ = build_sequential_Xy(df, poisson_all, comp_map)
    model.fit(X_deploy, y_deploy)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES, "comp_map": comp_map, "poisson_state": poisson_all},
        MODEL_PATH
    )

    # Save to metrics table if present
    try:
        con.execute(
            "INSERT OR REPLACE INTO metrics(model_version, n_matches, logloss, brier, accuracy) VALUES (?, ?, ?, ?, ?)",
            [V3_MODEL_VERSION, int(len(yte)), logloss, brier, acc]
        )
    except Exception:
        pass

    out = {"n_test": int(len(yte)), "logloss": logloss, "brier": brier, "accuracy": acc}
    if verbose:
        print(f"[v3] saved {V3_MODEL_VERSION} to {MODEL_PATH} | {out}", flush=True)
    return out

def load():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

def predict_upcoming(con, lookahead_days: int = 7) -> pd.DataFrame:
    obj = load()
    if obj is None:
        raise RuntimeError("v3 model not found. Run: footy train-v3")

    model = obj["model"]
    comp_map = obj["comp_map"]
    poisson_state = obj["poisson_state"]

    # roll states through all finished
    hist = con.execute(
        """SELECT m.utc_date, m.competition, m.home_team, m.away_team, m.home_goals, m.away_goals,
                  e.hs, e.as_, e.hc, e.ac
           FROM matches m
           LEFT JOIN match_extras e ON e.match_id=m.match_id
           WHERE m.status='FINISHED'
             AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
           ORDER BY m.utc_date ASC"""
    ).df()
    if hist.empty:
        raise RuntimeError("No finished matches to build rolling state from. Ingest history first.")

    hist["utc_date"] = pd.to_datetime(hist["utc_date"], utc=True)

    states: dict[str, TeamState] = {}
    ratings: dict[str, float] = {}

    for r in hist.itertuples(index=False):
        dt = r.utc_date
        home = canonical_team_name(r.home_team)
        away = canonical_team_name(r.away_team)
        hg = int(r.home_goals); ag = int(r.away_goals)

        sh = states.get(home) or TeamState()
        sa = states.get(away) or TeamState()
        states[home] = sh
        states[away] = sa

        sh.last_date = dt
        sa.last_date = dt

        if hg > ag: ph_pts, pa_pts = 3, 0
        elif hg == ag: ph_pts, pa_pts = 1, 1
        else: ph_pts, pa_pts = 0, 3

        _push(sh.pts5, ph_pts, 5); _push(sa.pts5, pa_pts, 5)
        _push(sh.pts10, ph_pts, 10); _push(sa.pts10, pa_pts, 10)

        _push(sh.gf5, hg, 5); _push(sh.ga5, ag, 5)
        _push(sa.gf5, ag, 5); _push(sa.ga5, hg, 5)

        _push(sh.gf10, hg, 10); _push(sh.ga10, ag, 10)
        _push(sa.gf10, ag, 10); _push(sa.ga10, hg, 10)

        hs = _to_float(r.hs) or 0.0
        as_ = _to_float(r.as_) or 0.0
        hc = _to_float(r.hc) or 0.0
        ac = _to_float(r.ac) or 0.0
        _push(sh.shots_for5, hs, 5); _push(sh.shots_against5, as_, 5)
        _push(sa.shots_for5, as_, 5); _push(sa.shots_against5, hs, 5)
        _push(sh.corners_for5, hc, 5); _push(sh.corners_against5, ac, 5)
        _push(sa.corners_for5, ac, 5); _push(sa.corners_against5, hc, 5)

        elo_update(ratings, home, away, hg, ag)

    s = settings()
    tracked = tuple(s.tracked_competitions or [])
    comp_filter = ""
    params = []
    if tracked:
        comp_filter = " AND m.competition IN (" + ",".join(["?"] * len(tracked)) + ")"
        params.extend(list(tracked))

    up = con.execute(
        f"""SELECT m.match_id, m.utc_date, m.competition, m.home_team, m.away_team,
                   e.b365h, e.b365d, e.b365a
            FROM matches m
            LEFT JOIN match_extras e ON e.match_id=m.match_id
            WHERE m.status IN ('SCHEDULED','TIMED')
              AND m.utc_date <= (CURRENT_TIMESTAMP + INTERVAL {int(lookahead_days)} DAY)
              {comp_filter}
            ORDER BY m.utc_date ASC""",
        params
    ).df()

    if up.empty:
        return up

    up["utc_date"] = pd.to_datetime(up["utc_date"], utc=True)

    X = []
    for r in up.itertuples(index=False):
        dt = r.utc_date
        comp = r.competition or ""
        home = canonical_team_name(r.home_team)
        away = canonical_team_name(r.away_team)

        sh = states.get(home) or TeamState()
        sa = states.get(away) or TeamState()

        rest_h = 7.0 if sh.last_date is None else max(0.0, (dt - sh.last_date).total_seconds() / 86400.0)
        rest_a = 7.0 if sa.last_date is None else max(0.0, (dt - sa.last_date).total_seconds() / 86400.0)

        pE_h, pE_d, pE_a, elo_diff = elo_predict(ratings, home, away)
        lam_h, lam_a = expected_goals(poisson_state, home, away)
        pP_h, pP_d, pP_a = outcome_probs(lam_h, lam_a)

        odds_h, odds_d, odds_a, has_odds = implied_probs_from_odds(r.b365h, r.b365d, r.b365a)

        row = [
            float(comp_map.get(comp, -1)),
            float(elo_diff),
            float(pE_h), float(pE_d), float(pE_a),
            float(lam_h), float(lam_a), float(lam_h - lam_a),
            float(pP_h), float(pP_d), float(pP_a),
            float(rest_h), float(rest_a), float(rest_h - rest_a),
            _mean(sh.pts5), _mean(sa.pts5), _mean(sh.pts10), _mean(sa.pts10),
            _mean(sh.gf5) - _mean(sh.ga5), _mean(sa.gf5) - _mean(sa.ga5),
            _mean(sh.gf10) - _mean(sh.ga10), _mean(sa.gf10) - _mean(sa.ga10),
            _mean(sh.shots_for5), _mean(sa.shots_for5), _mean(sh.shots_against5), _mean(sa.shots_against5),
            _mean(sh.corners_for5), _mean(sa.corners_for5), _mean(sh.corners_against5), _mean(sa.corners_against5),
            float(odds_h), float(odds_d), float(odds_a), float(has_odds),
        ]
        X.append(row)

    P = model.predict_proba(np.array(X, dtype=float))
    up["p_home"] = P[:, 0]
    up["p_draw"] = P[:, 1]
    up["p_away"] = P[:, 2]
    return up[["match_id","utc_date","competition","home_team","away_team","p_home","p_draw","p_away"]]
