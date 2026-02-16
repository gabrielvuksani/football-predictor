"""
Footy Predictor — FastAPI backend.

Serves:
    - REST JSON API at /api/*
    - Static files (CSS / JS) at /static/*
    - Single-page HTML frontend at /
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import duckdb

from footy.normalize import canonical_team_name
from footy.config import settings as get_settings
from footy.xg import compute_xg_advanced, learn_conversion_rates, ensure_xg_table

settings = get_settings()

WEB_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))

# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # no persistent DB connection — opened per-request instead

app = FastAPI(title="Footy Predictor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Response-Time header to all responses for performance monitoring."""
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time"] = f"{dt:.0f}ms"
    return response


def con() -> duckdb.DuckDBPyConnection:
    """Open a short-lived read-only DuckDB connection.

    Each API call gets its own connection that is released as soon as
    the caller's reference goes out of scope (CPython ref-counting).
    This means the shared file-lock is only held for the few milliseconds
    a query takes, so CLI write commands (go, refresh, nuke …) can
    acquire the exclusive lock between requests without blocking.

    Falls back to a brief retry if a write lock is active (e.g. during pipeline runs).
    """
    for attempt in range(3):
        try:
            return duckdb.connect(settings.db_path, read_only=True)
        except duckdb.IOException:
            if attempt < 2:
                import time as _t
                _t.sleep(0.5)
    # Last attempt — let it raise
    return duckdb.connect(settings.db_path, read_only=True)

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.get("/match/{match_id}", response_class=HTMLResponse)
async def match_page(request: Request, match_id: int):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------------------------
# API — Matches
# ---------------------------------------------------------------------------
@app.get("/api/matches")
async def api_matches(days: int = 7, model: str = "v8_council"):
    """Upcoming matches with predictions."""
    try:
        db = con()
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again shortly."}, 503)
    rows = db.execute(f"""
        SELECT m.match_id, m.utc_date, m.competition,
               m.home_team, m.away_team,
               p.p_home, p.p_draw, p.p_away,
               p.eg_home, p.eg_away
        FROM matches m
        LEFT JOIN predictions p ON p.match_id = m.match_id AND p.model_version = ?
        WHERE m.status IN ('SCHEDULED','TIMED')
          AND m.utc_date <= (CURRENT_TIMESTAMP + INTERVAL {int(days)} DAY)
        ORDER BY m.utc_date
    """, [model]).fetchall()

    matches = []
    for r in rows:
        matches.append({
            "match_id": r[0], "utc_date": str(r[1])[:16],
            "competition": r[2], "home_team": r[3], "away_team": r[4],
            "p_home": round(float(r[5]), 3) if r[5] else None,
            "p_draw": round(float(r[6]), 3) if r[6] else None,
            "p_away": round(float(r[7]), 3) if r[7] else None,
            "eg_home": round(float(r[8]), 2) if r[8] else None,
            "eg_away": round(float(r[9]), 2) if r[9] else None,
        })
    return {"matches": matches, "model": model}


@app.get("/api/matches/{match_id}")
async def api_match_detail(match_id: int, model: str = "v8_council"):
    """Detailed match data."""
    try:
        db = con()
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again shortly."}, 503)
    # Basic info + prediction
    row = db.execute("""
        SELECT m.match_id, m.utc_date, m.competition, m.home_team, m.away_team,
               m.status, m.home_goals, m.away_goals
        FROM matches m WHERE m.match_id = ?
    """, [match_id]).fetchone()
    if not row:
        return JSONResponse({"error": "Match not found"}, 404)

    pred_row = db.execute(
        "SELECT p_home, p_draw, p_away, eg_home, eg_away, notes FROM predictions WHERE match_id=? AND model_version=?",
        [match_id, model]
    ).fetchone()

    prediction = None
    poisson_stats = None
    if pred_row:
        prediction = {
            "p_home": round(float(pred_row[0]), 3),
            "p_draw": round(float(pred_row[1]), 3),
            "p_away": round(float(pred_row[2]), 3),
            "eg_home": round(float(pred_row[3]), 2) if pred_row[3] else None,
            "eg_away": round(float(pred_row[4]), 2) if pred_row[4] else None,
        }
        # Parse Poisson stats from notes JSON
        try:
            notes = json.loads(pred_row[5]) if pred_row[5] else {}
            if isinstance(notes, dict) and "btts" in notes:
                poisson_stats = {
                    "btts": notes.get("btts"),
                    "o25": notes.get("o25"),
                    "predicted_score": notes.get("predicted_score"),
                    "lambda_home": notes.get("lambda_home"),
                    "lambda_away": notes.get("lambda_away"),
                }
        except (json.JSONDecodeError, TypeError):
            pass

    # Odds
    odds_row = db.execute(
        "SELECT b365h, b365d, b365a FROM match_extras WHERE match_id=?",
        [match_id]
    ).fetchone()

    odds = None
    if odds_row and odds_row[0]:
        h, d, a = float(odds_row[0]), float(odds_row[1]), float(odds_row[2])
        ih, id_, ia = 1/h, 1/d, 1/a
        s = ih + id_ + ia
        odds = {
            "b365h": round(h, 2), "b365d": round(d, 2), "b365a": round(a, 2),
            "implied_h": round(ih/s, 3), "implied_d": round(id_/s, 3), "implied_a": round(ia/s, 3),
            "overround": round(s - 1, 3),
        }

    # Elo
    home_c = canonical_team_name(row[3])
    away_c = canonical_team_name(row[4])
    elo_h = db.execute("SELECT rating FROM elo_state WHERE team=?", [home_c]).fetchone()
    elo_a = db.execute("SELECT rating FROM elo_state WHERE team=?", [away_c]).fetchone()

    # Prediction score (if match is finished and was scored)
    score_row = db.execute(
        "SELECT outcome, logloss, brier, correct FROM prediction_scores WHERE match_id=? AND model_version=?",
        [match_id, model]
    ).fetchone()
    score = None
    if score_row:
        score = {
            "outcome": ["Home", "Draw", "Away"][score_row[0]],
            "logloss": round(float(score_row[1]), 4),
            "brier": round(float(score_row[2]), 4),
            "correct": bool(score_row[3]),
        }

    return {
        "match_id": row[0], "utc_date": str(row[1])[:16],
        "competition": row[2], "home_team": row[3], "away_team": row[4],
        "status": row[5], "home_goals": row[6], "away_goals": row[7],
        "prediction": prediction,
        "poisson": poisson_stats,
        "odds": odds,
        "elo": {
            "home": round(float(elo_h[0]), 0) if elo_h else None,
            "away": round(float(elo_a[0]), 0) if elo_a else None,
            "diff": round(float(elo_h[0] - elo_a[0]), 0) if elo_h and elo_a else None,
        },
        "score": score,
    }


# ---------------------------------------------------------------------------
# API — Expert Council Breakdown
# ---------------------------------------------------------------------------
@app.get("/api/matches/{match_id}/experts")
async def api_experts(match_id: int):
    """Expert council breakdown for a match."""
    try:
        db = con()
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again in a moment."}, 503)
    try:
        from footy.models.council import get_expert_breakdown
        result = get_expert_breakdown(db, match_id)
        if result is None:
            return JSONResponse({"error": "Could not compute expert breakdown"}, 404)
        return result
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again in a moment."}, 503)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ---------------------------------------------------------------------------
# API — H2H
# ---------------------------------------------------------------------------
@app.get("/api/matches/{match_id}/h2h")
async def api_h2h(match_id: int):
    """Head-to-head data for a match."""
    try:
        db = con()
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again shortly."}, 503)
    row = db.execute(
        "SELECT home_team, away_team FROM matches WHERE match_id=?", [match_id]
    ).fetchone()
    if not row:
        return JSONResponse({"error": "Match not found"}, 404)

    home = canonical_team_name(row[0])
    away = canonical_team_name(row[1])

    try:
        from footy.h2h import get_h2h_any_venue
        result = get_h2h_any_venue(con(), home, away, limit=10)

        # format recent matches for JSON serialization
        recent = []
        for m in result.get("recent_matches", []):
            recent.append({
                "utc_date": str(m.get("utc_date", ""))[:10],
                "home_team": m.get("home_team", ""),
                "away_team": m.get("away_team", ""),
                "home_goals": int(m.get("home_goals", 0)),
                "away_goals": int(m.get("away_goals", 0)),
            })

        stats = result.get("stats")
        if stats:
            # convert Timestamp to string
            stats = {k: (str(v) if hasattr(v, "isoformat") else v)
                     for k, v in stats.items()}

        return {"stats": stats, "recent_matches": recent}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ---------------------------------------------------------------------------
# API — Form (recent results streak)
# ---------------------------------------------------------------------------
@app.get("/api/matches/{match_id}/form")
async def api_form(match_id: int, n: int = 6):
    """Recent form (W/D/L streak + PPG) for both teams."""
    try:
        db = con()
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again shortly."}, 503)
    row = db.execute(
        "SELECT home_team, away_team FROM matches WHERE match_id=?", [match_id]
    ).fetchone()
    if not row:
        return JSONResponse({"error": "Match not found"}, 404)

    def get_form(team, limit=6):
        results = db.execute("""
            SELECT home_team, away_team, home_goals, away_goals
            FROM matches
            WHERE status = 'FINISHED'
              AND (home_team = ? OR away_team = ?)
            ORDER BY utc_date DESC
            LIMIT ?
        """, [team, team, limit]).fetchall()
        streak = []
        pts = 0
        for r in results:
            ht, at, hg, ag = r[0], r[1], r[2], r[3]
            if hg is None or ag is None:
                continue
            is_home = (ht == team)
            if is_home:
                if hg > ag: streak.append('W'); pts += 3
                elif hg == ag: streak.append('D'); pts += 1
                else: streak.append('L')
            else:
                if ag > hg: streak.append('W'); pts += 3
                elif ag == hg: streak.append('D'); pts += 1
                else: streak.append('L')
        ppg = pts / len(streak) if streak else None
        return streak, ppg

    hf, hppg = get_form(row[0], n)
    af, appg = get_form(row[1], n)

    return {
        "home_form": hf, "home_ppg": round(hppg, 2) if hppg else None,
        "away_form": af, "away_ppg": round(appg, 2) if appg else None,
    }


# ---------------------------------------------------------------------------
# API — xG
# ---------------------------------------------------------------------------
@app.get("/api/matches/{match_id}/xg")
async def api_xg(match_id: int):
    """xG data for a match."""
    try:
        db = con()
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again shortly."}, 503)
    row = db.execute(
        "SELECT home_team, away_team, competition FROM matches WHERE match_id=?", [match_id]
    ).fetchone()
    if not row:
        return JSONResponse({"error": "Match not found"}, 404)

    home_team, away_team, competition = row[0], row[1], row[2]

    # Check cache first
    cached = db.execute(
        "SELECT home_xg, away_xg, method, confidence FROM match_xg WHERE match_id=?",
        [match_id]
    ).fetchone()
    if cached:
        rates = {}
        try:
            rates = learn_conversion_rates(db)
            league_rates = rates.get(competition, rates.get("overall", {}))
        except Exception:
            league_rates = {"on_target": 0.10, "off_target": 0.02}
        return {
            "home_xg": round(float(cached[0]), 3) if cached[0] else None,
            "away_xg": round(float(cached[1]), 3) if cached[1] else None,
            "method": cached[2],
            "confidence": round(float(cached[3]), 3) if cached[3] else None,
            "rates": league_rates,
        }

    # Compute fresh
    try:
        result = compute_xg_advanced(db, home_team, away_team, competition)
        rates = result.get("details", {}).get("conversion_rates", {})
        return {
            "home_xg": result.get("home_xg"),
            "away_xg": result.get("away_xg"),
            "method": result.get("method"),
            "confidence": result.get("confidence"),
            "rates": {
                "on_target": rates.get("on_target", 0.10),
                "off_target": rates.get("off_target", 0.02),
            },
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ---------------------------------------------------------------------------
# API — Goal Patterns
# ---------------------------------------------------------------------------
@app.get("/api/matches/{match_id}/patterns")
async def api_patterns(match_id: int, n: int = 10):
    """Goal scoring patterns for both teams."""
    try:
        db = con()
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again shortly."}, 503)
    row = db.execute(
        "SELECT home_team, away_team FROM matches WHERE match_id=?", [match_id]
    ).fetchone()
    if not row:
        return JSONResponse({"error": "Match not found"}, 404)

    def _team_patterns(team: str, limit: int = 10) -> dict:
        matches = db.execute("""
            SELECT m.home_team, m.away_team, m.home_goals, m.away_goals,
                   me.hthg, me.htag
            FROM matches m
            LEFT JOIN match_extras me ON me.match_id = m.match_id
            WHERE m.status = 'FINISHED'
              AND (m.home_team = ? OR m.away_team = ?)
              AND m.home_goals IS NOT NULL
            ORDER BY m.utc_date DESC
            LIMIT ?
        """, [team, team, limit]).fetchall()

        if not matches:
            return {"first_goal_rate": None, "comeback_rate": None,
                    "btts_rate": None, "avg_goals_scored": None,
                    "avg_goals_conceded": None, "n": 0}

        scored_first = 0
        comebacks = 0
        btts = 0
        total_scored = 0
        total_conceded = 0
        ht_count = 0

        for m in matches:
            ht, at, hg, ag = m[0], m[1], m[2], m[3]
            hthg, htag = m[4], m[5]
            is_home = (ht == team)
            gf = int(hg) if is_home else int(ag)
            ga = int(ag) if is_home else int(hg)
            total_scored += gf
            total_conceded += ga

            # BTTS
            if gf > 0 and ga > 0:
                btts += 1

            # First goal / comeback (from half-time data)
            if hthg is not None and htag is not None:
                ht_count += 1
                ht_gf = int(hthg) if is_home else int(htag)
                ht_ga = int(htag) if is_home else int(hthg)
                if ht_gf > ht_ga:
                    scored_first += 1
                elif ht_gf < ht_ga and gf >= ga:
                    comebacks += 1

        n_matches = len(matches)
        return {
            "first_goal_rate": round(scored_first / ht_count, 3) if ht_count else None,
            "comeback_rate": round(comebacks / ht_count, 3) if ht_count else None,
            "btts_rate": round(btts / n_matches, 3),
            "avg_goals_scored": round(total_scored / n_matches, 2),
            "avg_goals_conceded": round(total_conceded / n_matches, 2),
            "n": n_matches,
        }

    return {
        "home": _team_patterns(row[0], n),
        "away": _team_patterns(row[1], n),
    }


# ---------------------------------------------------------------------------
# API — AI Narrative
# ---------------------------------------------------------------------------
@app.get("/api/matches/{match_id}/ai")
async def api_ai_narrative(match_id: int):
    """Generate AI narrative for a match using Ollama + expert breakdown."""
    try:
        from footy.models.council import get_expert_breakdown
        from footy.llm.ollama_client import chat

        bd = get_expert_breakdown(con(), match_id)
        if not bd:
            return {"narrative": None, "available": False}

        # Build structured prompt for Ollama
        experts_text = ""
        for name, data in bd["experts"].items():
            p = data["probs"]
            experts_text += (
                f"- {name.upper()} expert (confidence {data['confidence']:.0%}): "
                f"Home {p['home']:.0%}, Draw {p['draw']:.0%}, Away {p['away']:.0%}\n"
            )

        # Get the council's final prediction
        pred = con().execute(
            "SELECT p_home, p_draw, p_away FROM predictions "
            "WHERE match_id=? AND model_version='v8_council'",
            [match_id]
        ).fetchone()
        final = ""
        if pred:
            final = f"\nFinal Council prediction: Home {pred[0]:.0%}, Draw {pred[1]:.0%}, Away {pred[2]:.0%}"

        prompt = f"""Analyze this football match: {bd['home_team']} vs {bd['away_team']} ({bd['competition']}).

Expert council opinions:
{experts_text}{final}

Write a concise, insightful 3-4 sentence match preview. Highlight where experts agree/disagree 
and what that means. Be specific about why certain experts are confident or cautious.
Do not use bullet points. Write in a punchy, analytical style."""

        response = chat([{"role": "user", "content": prompt}])
        return {
            "narrative": response if response else None,
            "available": bool(response),
            "experts": bd["experts"],
        }
    except Exception as e:
        return {"narrative": None, "available": False, "error": str(e)}


# ---------------------------------------------------------------------------
# API — Insights
# ---------------------------------------------------------------------------
@app.get("/api/insights/value-bets")
async def api_value_bets(min_edge: float = 0.05):
    """Value bet scanner."""
    try:
        db = con()
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again shortly."}, 503)
    rows = db.execute("""
        SELECT m.match_id, m.utc_date, m.competition, m.home_team, m.away_team,
               p.p_home, p.p_draw, p.p_away,
               e.b365h, e.b365d, e.b365a
        FROM matches m
        JOIN predictions p ON p.match_id = m.match_id AND p.model_version = 'v8_council'
        LEFT JOIN match_extras e ON e.match_id = m.match_id
        WHERE m.status IN ('SCHEDULED','TIMED')
        ORDER BY m.utc_date
    """).fetchall()

    bets = []
    for r in rows:
        if not r[8] or not r[9] or not r[10]:
            continue
        for label, p_model, odds_val in [
            ("Home", float(r[5]), float(r[8])),
            ("Draw", float(r[6]), float(r[9])),
            ("Away", float(r[7]), float(r[10])),
        ]:
            if odds_val <= 1:
                continue
            implied = 1 / odds_val
            edge = p_model - implied
            if edge >= min_edge:
                bets.append({
                    "match_id": r[0], "date": str(r[1])[:10],
                    "competition": r[2],
                    "home_team": r[3], "away_team": r[4],
                    "bet": label, "odds": round(odds_val, 2),
                    "model_prob": round(p_model, 3),
                    "implied_prob": round(implied, 3),
                    "edge": round(edge, 3),
                })

    bets.sort(key=lambda x: -x["edge"])
    # Add Kelly criterion
    for b in bets:
        k = (b["model_prob"] * b["odds"] - 1) / (b["odds"] - 1) if b["odds"] > 1 else 0
        b["kelly"] = round(max(0, k), 3)
    return {"bets": bets[:30]}


# ---------------------------------------------------------------------------
# API — League Table
# ---------------------------------------------------------------------------
@app.get("/api/league-table/{competition}")
async def api_league_table(competition: str, season: str = None):
    """Current league standings built from finished matches."""
    try:
        db = con()
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again shortly."}, 503)

    # Determine season
    if season is None:
        season_row = db.execute("""
            SELECT DISTINCT season_code FROM match_extras
            WHERE competition = ? AND season_code IS NOT NULL
            ORDER BY season_code DESC LIMIT 1
        """, [competition]).fetchone()
        season = season_row[0] if season_row else None

    # Get finished matches for this competition/season
    if season:
        matches = db.execute("""
            SELECT m.home_team, m.away_team, m.home_goals, m.away_goals
            FROM matches m
            JOIN match_extras me ON me.match_id = m.match_id
            WHERE m.status = 'FINISHED'
              AND m.competition = ?
              AND me.season_code = ?
              AND m.home_goals IS NOT NULL
        """, [competition, season]).fetchall()
    else:
        matches = db.execute("""
            SELECT home_team, away_team, home_goals, away_goals
            FROM matches
            WHERE status = 'FINISHED'
              AND competition = ?
              AND home_goals IS NOT NULL
        """, [competition]).fetchall()

    if not matches:
        return JSONResponse({"error": f"No finished matches for {competition}"}, 404)

    # Build table
    table: dict = {}
    for ht, at, hg, ag in matches:
        hg, ag = int(hg), int(ag)
        for team in (ht, at):
            if team not in table:
                table[team] = {"team": team, "p": 0, "w": 0, "d": 0, "l": 0,
                                "gf": 0, "ga": 0, "gd": 0, "pts": 0}

        table[ht]["p"] += 1
        table[at]["p"] += 1
        table[ht]["gf"] += hg
        table[ht]["ga"] += ag
        table[at]["gf"] += ag
        table[at]["ga"] += hg

        if hg > ag:
            table[ht]["w"] += 1; table[ht]["pts"] += 3
            table[at]["l"] += 1
        elif hg == ag:
            table[ht]["d"] += 1; table[ht]["pts"] += 1
            table[at]["d"] += 1; table[at]["pts"] += 1
        else:
            table[at]["w"] += 1; table[at]["pts"] += 3
            table[ht]["l"] += 1

    for t in table.values():
        t["gd"] = t["gf"] - t["ga"]

    standings = sorted(table.values(), key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))
    for i, t in enumerate(standings, 1):
        t["pos"] = i

    return {"competition": competition, "season": season, "standings": standings}


# ---------------------------------------------------------------------------
# API — Stats
# ---------------------------------------------------------------------------
@app.get("/api/stats")
async def api_stats():
    """Database and model statistics."""
    try:
        db = con()
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again shortly."}, 503)
    total = db.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    finished = db.execute("SELECT COUNT(*) FROM matches WHERE status='FINISHED'").fetchone()[0]
    upcoming = db.execute("SELECT COUNT(*) FROM matches WHERE status IN ('TIMED','SCHEDULED')").fetchone()[0]
    teams = db.execute("SELECT COUNT(DISTINCT home_team) FROM matches").fetchone()[0]
    preds = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    try:
        h2h_pairs = db.execute("SELECT COUNT(*) FROM h2h_stats").fetchone()[0]
    except Exception:
        h2h_pairs = 0

    comps = db.execute("""
        SELECT competition, COUNT(*) n
        FROM matches WHERE status='FINISHED'
        GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()

    models = db.execute("""
        SELECT model_version, COUNT(*) n
        FROM predictions GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()

    return {
        "total_matches": total, "finished": finished,
        "upcoming": upcoming, "teams": teams, "predictions": preds,
        "h2h_pairs": h2h_pairs,
        "competitions": [{"code": c[0], "count": c[1]} for c in comps],
        "models": [{"version": m[0], "predictions": m[1]} for m in models],
    }


# ---------------------------------------------------------------------------
# API — Performance tracking
# ---------------------------------------------------------------------------
@app.get("/api/performance")
async def api_performance(model: str = "v8_council"):
    """Model performance metrics — accuracy, logloss, brier, calibration."""
    try:
        db = con()
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again shortly."}, 503)

    # Aggregate metrics
    metrics_row = db.execute(
        "SELECT n_matches, logloss, brier, accuracy FROM metrics WHERE model_version = ?",
        [model]
    ).fetchone()

    metrics = None
    if metrics_row:
        metrics = {
            "n_scored": metrics_row[0],
            "logloss": round(float(metrics_row[1]), 4) if metrics_row[1] else None,
            "brier": round(float(metrics_row[2]), 4) if metrics_row[2] else None,
            "accuracy": round(float(metrics_row[3]), 4) if metrics_row[3] else None,
        }

    # Recent scored predictions with outcomes
    recent = db.execute("""
        SELECT m.utc_date, m.home_team, m.away_team, m.home_goals, m.away_goals,
               p.p_home, p.p_draw, p.p_away,
               ps.outcome, ps.logloss, ps.correct, m.competition
        FROM prediction_scores ps
        JOIN predictions p ON ps.match_id = p.match_id AND ps.model_version = p.model_version
        JOIN matches m ON ps.match_id = m.match_id
        WHERE ps.model_version = ?
        ORDER BY m.utc_date DESC
        LIMIT 50
    """, [model]).fetchall()

    scored_matches = []
    for r in recent:
        scored_matches.append({
            "date": str(r[0])[:10],
            "home_team": r[1], "away_team": r[2],
            "home_goals": r[3], "away_goals": r[4],
            "p_home": round(float(r[5]), 3), "p_draw": round(float(r[6]), 3), "p_away": round(float(r[7]), 3),
            "outcome": ["Home", "Draw", "Away"][r[8]],
            "logloss": round(float(r[9]), 4),
            "correct": bool(r[10]),
            "competition": r[11],
        })

    # Accuracy by competition
    by_comp = db.execute("""
        SELECT m.competition, COUNT(*) as n,
               SUM(CASE WHEN ps.correct THEN 1 ELSE 0 END) as correct,
               AVG(ps.logloss) as avg_ll
        FROM prediction_scores ps
        JOIN matches m ON ps.match_id = m.match_id
        WHERE ps.model_version = ?
        GROUP BY m.competition
        ORDER BY n DESC
    """, [model]).fetchall()

    comp_perf = []
    for r in by_comp:
        comp_perf.append({
            "competition": r[0], "n": r[1],
            "accuracy": round(float(r[2]) / r[1], 3) if r[1] else 0,
            "logloss": round(float(r[3]), 4) if r[3] else None,
        })

    # Calibration: predicted probability buckets vs actual outcomes
    cal_rows = db.execute("""
        SELECT p.p_home, p.p_draw, p.p_away, ps.outcome
        FROM prediction_scores ps
        JOIN predictions p ON ps.match_id = p.match_id AND ps.model_version = p.model_version
        WHERE ps.model_version = ?
    """, [model]).fetchall()

    buckets = {i: {"predicted_sum": 0.0, "actual_sum": 0, "count": 0}
               for i in range(10)}  # 0-10%, 10-20%, ..., 90-100%
    for r in cal_rows:
        # Each row contributes 3 data points (one per outcome)
        for prob, is_actual in [
            (float(r[0]), 1 if r[3] == 0 else 0),  # home
            (float(r[1]), 1 if r[3] == 1 else 0),  # draw
            (float(r[2]), 1 if r[3] == 2 else 0),  # away
        ]:
            idx = min(int(prob * 10), 9)
            buckets[idx]["predicted_sum"] += prob
            buckets[idx]["actual_sum"] += is_actual
            buckets[idx]["count"] += 1

    calibration = []
    for i in range(10):
        b = buckets[i]
        calibration.append({
            "bucket": f"{i*10}-{(i+1)*10}%",
            "avg_predicted": round(b["predicted_sum"] / b["count"], 4) if b["count"] else None,
            "avg_actual": round(b["actual_sum"] / b["count"], 4) if b["count"] else None,
            "count": b["count"],
        })

    return {
        "model": model,
        "metrics": metrics,
        "recent": scored_matches,
        "by_competition": comp_perf,
        "calibration": calibration,
    }


# ---------------------------------------------------------------------------
# API — Competitions
# ---------------------------------------------------------------------------
@app.get("/api/competitions")
async def api_competitions():
    """List all competitions with metadata."""
    try:
        db = con()
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again shortly."}, 503)
    rows = db.execute("""
        SELECT m.competition,
               COUNT(*) AS match_count,
               COUNT(DISTINCT m.home_team) AS team_count,
               MAX(me.season_code) AS current_season
        FROM matches m
        LEFT JOIN match_extras me ON me.match_id = m.match_id
        GROUP BY m.competition
        ORDER BY match_count DESC
    """).fetchall()

    # Competition name mapping (common codes)
    _NAMES = {
        "PL": "Premier League", "BL1": "Bundesliga", "SA": "Serie A",
        "PD": "La Liga", "FL1": "Ligue 1", "ELC": "Championship",
        "DED": "Eredivisie", "PPL": "Primeira Liga", "CL": "Champions League",
        "EC": "European Championship", "WC": "World Cup",
    }

    competitions = []
    for r in rows:
        code = r[0]
        competitions.append({
            "code": code,
            "name": _NAMES.get(code, code),
            "match_count": r[1],
            "team_count": r[2],
            "current_season": r[3],
        })
    return {"competitions": competitions}


@app.get("/api/last-updated")
async def api_last_updated():
    """When predictions were last generated."""
    try:
        db = con()
    except duckdb.IOException:
        return JSONResponse({"error": "Database busy — pipeline running. Try again shortly."}, 503)
    row = db.execute(
        "SELECT MAX(created_at) FROM predictions WHERE model_version = 'v8_council'"
    ).fetchone()
    return {"last_updated": str(row[0])[:19] if row and row[0] else None}
