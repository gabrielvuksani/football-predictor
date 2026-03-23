"""Prediction endpoints: Bayesian, unified, export, and upset alerts."""
from __future__ import annotations

import csv
import io
import json
import math

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from . import con, safe_error, validate_model, parse_notes

router = APIRouter(prefix="/api", tags=["predictions"])

# Bayesian engine cache
_BAYESIAN_ENGINE_CACHE: dict[str, object] = {
    "signature": None,
    "generated_at": 0.0,
    "engine": None,
}
_BAYESIAN_ENGINE_TTL_SECONDS = 300.0


@router.get("/bayesian/predict/{match_id}")
async def api_bayesian_predict(match_id: int):
    """Get Bayesian prediction for a specific match."""
    try:
        db = con()
        row = db.execute(
            "SELECT home_team, away_team, competition FROM matches WHERE match_id = $1",
            [match_id]
        ).fetchone()
        if not row:
            return JSONResponse({"error": "Match not found"}, 404)
        home_team, away_team, comp = row
    except Exception as e:
        return safe_error(e, "bayesian prediction")

    try:
        pred = db.execute(
            "SELECT p_home, p_draw, p_away, eg_home, eg_away FROM predictions WHERE match_id = $1 AND model_version = 'v13_oracle'",
            [match_id]
        ).fetchone()

        if pred and pred[0] is not None:
            p_home, p_draw, p_away = float(pred[0]), float(pred[1]), float(pred[2])
            eg_h = float(pred[3]) if pred[3] is not None else p_home * 2.5
            eg_a = float(pred[4]) if pred[4] is not None else p_away * 2.5
        else:
            p_home, p_draw, p_away = 0.45, 0.25, 0.30
            eg_h, eg_a = 1.5, 1.2
    except Exception:
        p_home, p_draw, p_away = 0.45, 0.25, 0.30
        eg_h, eg_a = 1.5, 1.2

    # Compute score probabilities from Poisson model instead of hardcoding
    top_scores = _poisson_score_probs(eg_h, eg_a, top_n=5)

    return {
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "top_scores": top_scores,
    }


@router.get("/unified-prediction/{match_id}")
async def api_unified_prediction(match_id: int):
    """Get unified prediction combining all models for a match."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        match = db.execute("SELECT match_id FROM matches WHERE match_id = $1", [match_id]).fetchone()
        if not match:
            return JSONResponse({"error": "Match not found"}, 404)

        pred = db.execute(
            "SELECT p_home, p_draw, p_away, eg_home, eg_away, notes FROM predictions WHERE match_id = $1 AND model_version = 'v13_oracle'",
            [match_id]
        ).fetchone()

        if not pred or pred[0] is None:
            return JSONResponse({"error": "No prediction found for this match"}, 404)

        p_home, p_draw, p_away = float(pred[0]), float(pred[1]), float(pred[2])
        notes = parse_notes(pred[5])

        # Build component breakdown from expert outputs
        component_breakdown = {}
        expert_prefixes = {
            "elo": ("elo_home", "elo_draw", "elo_away"),
            "poisson": ("poi_home", "poi_draw", "poi_away"),
            "dixon_coles": ("dc_home", "dc_draw", "dc_away"),
            "bivariate_poisson": ("bp_home", "bp_draw", "bp_away"),
            "copula": ("cop_home", "cop_draw", "cop_away"),
            "zip_model": ("zip_home", "zip_draw", "zip_away"),
            "bt_model": ("bt_home", "bt_draw", "bt_away"),
            "pi_rating": ("pi_home", "pi_draw", "pi_away"),
            "glicko2": ("gl_home", "gl_draw", "gl_away"),
        }
        for expert_name, (hk, dk, ak) in expert_prefixes.items():
            if hk in notes and dk in notes:
                try:
                    eh = float(notes[hk])
                    ed = float(notes[dk])
                    ea = float(notes.get(ak, max(0.0, 1.0 - eh - ed)))
                    component_breakdown[expert_name] = [round(eh, 4), round(ed, 4), round(ea, 4)]
                except (TypeError, ValueError):
                    pass

        # Build score probabilities from notes or compute
        score_probs = []
        score_dist = notes.get("score_dist") or notes.get("score_probs")
        if isinstance(score_dist, dict):
            sorted_scores = sorted(score_dist.items(), key=lambda x: float(x[1]), reverse=True)[:5]
            score_probs = [{"score": s, "probability": round(float(p), 4)} for s, p in sorted_scores]
        else:
            eg_h = float(pred[3]) if pred[3] is not None else p_home * 2.5
            eg_a = float(pred[4]) if pred[4] is not None else p_away * 2.5
            score_probs = _poisson_score_probs(eg_h, eg_a, top_n=5)

        return {
            "match_id": match_id,
            "n_models": len(component_breakdown) if component_breakdown else 1,
            "bayesian_cached": bool(_BAYESIAN_ENGINE_CACHE.get("engine")),
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "score_probabilities": score_probs,
            "component_breakdown": component_breakdown,
        }
    except Exception as e:
        return safe_error(e, "unified prediction")


@router.get("/upset-alerts")
async def api_upset_alerts(min_risk: float = Query(0.3, ge=0, le=1)):
    """Top upcoming matches ranked by upset probability."""
    try:
        db = con()
        # Use UPCOMING status consistently (not SCHEDULED/TIMED)
        rows = db.execute("""
            SELECT m.match_id, m.home_team, m.away_team, m.competition,
                   m.utc_date, p.p_home, p.p_draw, p.p_away, p.notes
            FROM matches m
            JOIN predictions p ON m.match_id = p.match_id
            WHERE m.status = 'UPCOMING'
              AND p.model_version = 'v13_oracle'
              AND p.p_home IS NOT NULL
            ORDER BY m.utc_date ASC LIMIT 100
        """).fetchall()
    except Exception as e:
        return safe_error(e, "upset alerts")

    alerts = []
    for row in rows:
        match_id, home, away, comp, utc_date, ph, pd_, pa, notes = row
        probs = [float(ph), float(pd_), float(pa)]
        favourite_prob = max(probs)
        upset_risk = max(0.0, 1.0 - favourite_prob * 1.3)

        notes_dict = parse_notes(notes)
        upset_score = float(notes_dict.get("upset_risk", upset_risk))

        if upset_score >= min_risk:
            reasons = []
            if notes_dict.get("xgr_overperf_h", 0) > 0.3:
                reasons.append("Home team overperforming xG — regression likely")
            if notes_dict.get("inj_diff", 0) > 0.5:
                reasons.append("Significant injury advantage for away team")
            if notes_dict.get("mot_motivation_diff", 0) < -0.2:
                reasons.append("Away team more motivated")
            if notes_dict.get("rot_congestion_diff", 0) > 0.3:
                reasons.append("Home team fixture congestion")
            if not reasons:
                reasons.append("Model disagreement with market odds")

            alerts.append({
                "match_id": int(match_id),
                "home_team": home, "away_team": away,
                "competition": comp,
                "utc_date": str(utc_date),
                "p_home": round(float(ph), 4),
                "p_draw": round(float(pd_), 4),
                "p_away": round(float(pa), 4),
                "upset_risk": round(upset_score, 3),
                "upset_level": "high" if upset_score > 0.6 else ("medium" if upset_score > 0.4 else "low"),
                "reasons": reasons,
            })

    alerts.sort(key=lambda x: x["upset_risk"], reverse=True)
    return {"alerts": alerts[:20], "total": len(alerts)}


@router.get("/export/predictions")
async def api_export_predictions(
    model: str = Query("v13_oracle"),
    days: int = Query(14, ge=1, le=90),
    status: str = Query("ALL", regex="^(UPCOMING|FINISHED|ALL)$"),
):
    """Export predictions as CSV with optional filtering."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        model = validate_model(model)
        interval_clause = f"NOW() - INTERVAL {days} DAY"

        query = f"""
            SELECT m.match_id, m.home_team, m.away_team, m.competition,
                   m.utc_date, p.p_home, p.p_draw, p.p_away,
                   p.eg_home, p.eg_away, p.notes
            FROM predictions p JOIN matches m ON p.match_id = m.match_id
            WHERE p.model_version = $1 AND m.utc_date > {interval_clause}
        """
        params = [model]
        if status != "ALL":
            if status in {"UPCOMING", "FINISHED"}:
                query += " AND m.status = $2"
                params.append(status)
        query += " ORDER BY m.utc_date DESC"

        rows = db.execute(query, params).fetchall()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Match ID", "Home Team", "Away Team", "Competition", "Date",
            "Home Win %", "Draw %", "Away Win %", "BTTS %", "Over 2.5 %", "xG Home", "xG Away"
        ])
        for row in rows:
            notes = json.loads(row[10]) if row[10] else {}
            btts, o25 = notes.get("btts"), notes.get("o25")
            writer.writerow([
                row[0], row[1], row[2], row[3], str(row[4])[:10],
                f"{float(row[5])*100:.1f}" if row[5] else "",
                f"{float(row[6])*100:.1f}" if row[6] else "",
                f"{float(row[7])*100:.1f}" if row[7] else "",
                f"{float(btts)*100:.1f}" if btts is not None else "",
                f"{float(o25)*100:.1f}" if o25 is not None else "",
                f"{float(row[8]):.2f}" if row[8] else "",
                f"{float(row[9]):.2f}" if row[9] else "",
            ])

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )
    except Exception as e:
        return safe_error(e, "export predictions")


def _poisson_score_probs(lambda_h: float, lambda_a: float, top_n: int = 5) -> list[dict]:
    """Compute most likely scorelines from independent Poisson model."""
    score_map = {}
    for gh in range(7):
        for ga in range(7):
            prob = (
                math.exp(-lambda_h) * lambda_h**gh / math.factorial(gh)
                * math.exp(-lambda_a) * lambda_a**ga / math.factorial(ga)
            )
            score_map[f"{gh}-{ga}"] = prob
    top = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"score": s, "probability": round(p, 4)} for s, p in top]
