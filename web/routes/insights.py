"""Insights endpoints: value bets, BTTS/O2.5, accumulators, form tables, accuracy, review."""
from __future__ import annotations

import json
from collections import defaultdict

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from . import con, safe_error, validate_competition, VALID_COMPETITIONS

router = APIRouter(prefix="/api/insights", tags=["insights"])


@router.get("/value-bets")
async def api_value_bets(min_edge: float = Query(0.03, ge=0, le=0.5)):
    """Get identified value bets with edges above threshold."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        rows = db.execute("""
            SELECT m.match_id, m.home_team, m.away_team, m.competition, m.utc_date,
                   p.p_home, p.p_draw, p.p_away, p.confidence,
                   COALESCE(e.psh, e.b365h) AS odds_h,
                   COALESCE(e.psd, e.b365d) AS odds_d,
                   COALESCE(e.psa, e.b365a) AS odds_a
            FROM matches m
            JOIN predictions p ON m.match_id = p.match_id
            LEFT JOIN match_extras e ON m.match_id = e.match_id
            WHERE m.status = 'UPCOMING'
              AND p.model_version = 'v12_analyst'
            ORDER BY m.utc_date ASC
            LIMIT 50
        """).fetchall()

        bets = []
        for r in rows:
            odds_h = float(r[9]) if r[9] else None
            odds_d = float(r[10]) if r[10] else None
            odds_a = float(r[11]) if r[11] else None
            confidence = float(r[8]) if r[8] is not None else 0.0

            for outcome, prob, odds in [
                ("home", float(r[5]), odds_h),
                ("draw", float(r[6]), odds_d),
                ("away", float(r[7]), odds_a),
            ]:
                if odds is None or odds <= 1.0:
                    continue
                implied_prob = 1.0 / odds
                edge = prob - implied_prob
                if edge >= min_edge and edge > 0.05 and confidence > 0.50:
                    bets.append({
                        "match_id": int(r[0]),
                        "home_team": r[1], "away_team": r[2],
                        "competition": r[3],
                        "date": str(r[4])[:10],
                        "bet": outcome,
                        "model_prob": round(prob, 4),
                        "implied_prob": round(implied_prob, 4),
                        "odds": round(odds, 2),
                        "edge": round(edge, 4),
                        "confidence": round(confidence, 3),
                    })

        bets.sort(key=lambda b: b["edge"], reverse=True)
        return {"bets": bets}
    except Exception as e:
        return safe_error(e, "value bets")


@router.get("/btts-ou")
async def api_btts_ou():
    """Get BTTS and Over/Under 2.5 predictions grouped by likelihood."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        rows = db.execute("""
            SELECT m.match_id, m.home_team, m.away_team, m.competition, p.notes
            FROM matches m
            JOIN predictions p ON m.match_id = p.match_id
            WHERE m.status = 'UPCOMING' AND p.model_version = 'v12_analyst'
            ORDER BY m.utc_date ASC
        """).fetchall()

        btts_likely, btts_unlikely, over25, under25 = [], [], [], []

        for r in rows:
            notes = json.loads(r[4]) if r[4] else {}
            btts = notes.get("btts")
            o25 = notes.get("o25")
            base = {"match_id": int(r[0]), "home_team": r[1], "away_team": r[2], "competition": r[3]}

            if btts is not None:
                bv = float(btts)
                if bv > 0.55:
                    btts_likely.append({**base, "btts_prob": bv})
                elif bv < 0.35:
                    btts_unlikely.append({**base, "btts_prob": bv})

            if o25 is not None:
                ov = float(o25)
                if ov > 0.55:
                    over25.append({**base, "o25_prob": ov})
                elif ov < 0.35:
                    under25.append({**base, "under25_prob": ov})

        return {"btts_likely": btts_likely, "btts_unlikely": btts_unlikely, "over25": over25, "under25": under25}
    except Exception as e:
        return safe_error(e, "btts/ou")


@router.get("/accumulators")
async def api_accumulators():
    """Get suggested accumulators based on model consensus."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        rows = db.execute("""
            SELECT m.match_id, m.home_team, m.away_team, m.competition,
                   p.p_home, p.p_draw, p.p_away
            FROM matches m
            JOIN predictions p ON m.match_id = p.match_id
            WHERE m.status = 'UPCOMING' AND p.model_version = 'v12_analyst'
              AND (p.p_home > 0.55 OR p.p_away > 0.55 OR p.p_draw > 0.5)
            ORDER BY GREATEST(p.p_home, p.p_draw, p.p_away) DESC LIMIT 20
        """).fetchall()

        accumulators = []
        if len(rows) >= 3:
            for i in range(0, len(rows) - 2, 3):
                legs = []
                odds = 1.0
                for j in range(3):
                    r = rows[i + j]
                    strongest = max(float(r[4]), float(r[5]), float(r[6]))
                    outcome = "home" if float(r[4]) == strongest else "draw" if float(r[5]) == strongest else "away"
                    odds *= (1 / strongest) if strongest > 0 else 2.0
                    legs.append({
                        "match_id": int(r[0]), "home_team": r[1], "away_team": r[2],
                        "competition": r[3], "outcome": outcome, "probability": strongest,
                    })
                accumulators.append({"legs": legs, "odds": odds})

        return {"accumulators": accumulators}
    except Exception as e:
        return safe_error(e, "accumulators")


@router.get("/form-table/{competition}")
async def api_form_table(competition: str):
    """Get form table for a specific competition."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        rows = db.execute("""
            SELECT home_team, away_team, home_goals, away_goals, utc_date
            FROM matches WHERE competition = $1 AND status = 'FINISHED' AND home_goals IS NOT NULL
            ORDER BY utc_date DESC LIMIT 500
        """, [competition]).fetchall()

        team_results = defaultdict(list)
        for r in rows:
            ht, at, hg, ag = r[0], r[1], int(r[2]), int(r[3])
            if len(team_results[ht]) < 5:
                team_results[ht].append(("W" if hg > ag else "D" if hg == ag else "L", hg, ag))
            if len(team_results[at]) < 5:
                team_results[at].append(("W" if ag > hg else "D" if ag == hg else "L", ag, hg))

        table = []
        for team, results in sorted(team_results.items()):
            w = sum(1 for r in results if r[0] == "W")
            d = sum(1 for r in results if r[0] == "D")
            l = sum(1 for r in results if r[0] == "L")
            table.append({
                "team": team, "played": len(results),
                "wins": w, "draws": d, "losses": l,
                "goals_for": sum(r[1] for r in results),
                "goals_against": sum(r[2] for r in results),
                "form": "".join(r[0] for r in results),
                "points": w * 3 + d,
            })
        table.sort(key=lambda x: x["points"], reverse=True)
        return {"table": table}
    except Exception as e:
        return safe_error(e, "form table")


@router.get("/accuracy")
async def api_accuracy(days_back: int = Query(30, ge=7, le=365)):
    """Get model accuracy metrics over a time period."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        row = db.execute(f"""
            SELECT COUNT(*) as total,
                   COUNT(CASE WHEN ps.correct = TRUE THEN 1 END) as correct,
                   AVG(ps.logloss) as avg_log_loss
            FROM prediction_scores ps
            JOIN matches m ON ps.match_id = m.match_id
            WHERE ps.model_version = 'v12_analyst'
              AND m.status = 'FINISHED' AND m.home_goals IS NOT NULL
              AND m.utc_date > NOW() - INTERVAL {days_back} DAY
        """).fetchone()

        total = int(row[0]) if row[0] else 0
        correct = int(row[1]) if row[1] else 0
        return {
            "accuracy": (correct / total) if total > 0 else 0,
            "total_predictions": total,
            "correct_predictions": correct,
            "log_loss": round(float(row[2]) if row[2] is not None else 1.0, 4),
            "period_days": days_back,
        }
    except Exception as e:
        return safe_error(e, "accuracy")


@router.get("/round-preview/{competition}")
async def api_round_preview(competition: str):
    """Get a preview of the upcoming round in a competition."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        if validate_competition(competition) is None:
            return JSONResponse({"error": "Invalid competition"}, 400)

        row = db.execute("""
            SELECT COUNT(*), AVG(GREATEST(p.p_home, p.p_draw, p.p_away))
            FROM matches m
            LEFT JOIN predictions p ON m.match_id = p.match_id AND p.model_version = 'v12_analyst'
            WHERE m.competition = $1 AND m.status = 'UPCOMING'
              AND m.utc_date < NOW() + INTERVAL 7 DAY
        """, [competition]).fetchone()

        return {
            "competition": competition,
            "upcoming_matches": int(row[0]) if row[0] else 0,
            "avg_confidence": float(row[1]) if row[1] is not None else 0.5,
        }
    except Exception as e:
        return safe_error(e, "round preview")


@router.get("/post-match-review")
async def api_post_match_review(days_back: int = Query(7, ge=1, le=30)):
    """Get review of recent match results and prediction accuracy."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        row = db.execute(f"""
            SELECT COUNT(*),
                   CASE WHEN COUNT(*) > 0
                        THEN COUNT(CASE WHEN ps.correct = TRUE THEN 1 END) * 100.0 / COUNT(*)
                        ELSE 0 END
            FROM prediction_scores ps
            JOIN matches m ON ps.match_id = m.match_id
            WHERE ps.model_version = 'v12_analyst'
              AND m.status = 'FINISHED' AND m.home_goals IS NOT NULL
              AND m.utc_date > NOW() - INTERVAL {days_back} DAY
        """).fetchone()

        total = int(row[0]) if row[0] else 0
        accuracy = float(row[1]) if row[1] is not None else 0
        return {
            "summary": f"{total} matches reviewed in last {days_back} days",
            "accuracy": round(accuracy, 1),
            "period_days": days_back,
        }
    except Exception as e:
        return safe_error(e, "post-match review")
