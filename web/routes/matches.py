"""Match listing, detail, expert, H2H, form, xG, and pattern endpoints."""
from __future__ import annotations

import json

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from footy.normalize import canonical_team_name
from . import con, safe_error, validate_model, parse_notes

router = APIRouter(prefix="/api", tags=["matches"])


@router.get("/matches")
async def api_matches(
    days: int = Query(14, ge=1, le=90),
    model: str = Query("v13_oracle"),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
):
    """Get upcoming matches with predictions, with pagination support."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        model = validate_model(model)
        offset = (page - 1) * limit
        interval_clause = f"NOW() + INTERVAL {days} DAY"

        total_count_result = db.execute(f"""
            SELECT COUNT(*) FROM matches m
            WHERE m.status = 'UPCOMING'
              AND m.utc_date < {interval_clause}
        """).fetchone()
        total_count = total_count_result[0] if total_count_result else 0

        rows = db.execute(f"""
            SELECT m.match_id, m.home_team, m.away_team, m.competition,
                   m.utc_date, p.p_home, p.p_draw, p.p_away,
                   p.eg_home, p.eg_away, p.notes
            FROM matches m
            LEFT JOIN predictions p ON m.match_id = p.match_id
                                    AND p.model_version = $1
            WHERE m.status = 'UPCOMING'
              AND m.utc_date < {interval_clause}
            ORDER BY m.utc_date ASC
            LIMIT $2 OFFSET $3
        """, [model, limit, offset]).fetchall()

        matches = []
        for r in rows:
            match = {
                "match_id": int(r[0]),
                "home_team": r[1],
                "away_team": r[2],
                "competition": r[3],
                "utc_date": str(r[4])[:19] if r[4] else None,
            }
            if r[5] is not None:
                notes = parse_notes(r[10])
                match.update({
                    "p_home": float(r[5]),
                    "p_draw": float(r[6]),
                    "p_away": float(r[7]),
                    "btts": float(notes["btts"]) if notes.get("btts") is not None else None,
                    "o25": float(notes["o25"]) if notes.get("o25") is not None else None,
                    "eg_home": float(r[8]) if r[8] is not None else None,
                    "eg_away": float(r[9]) if r[9] is not None else None,
                })
            matches.append(match)

        return {
            "matches": matches,
            "page": page,
            "limit": limit,
            "total_count": total_count,
            "has_more": (offset + limit) < total_count,
        }
    except Exception as e:
        return safe_error(e, "matches list")


@router.get("/matches/{match_id}")
async def api_match_detail(match_id: int, model: str = Query("v13_oracle")):
    """Get detailed match information with full prediction breakdown."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        model = validate_model(model)
        m = db.execute(
            "SELECT match_id, home_team, away_team, competition, utc_date, status, home_goals, away_goals FROM matches WHERE match_id = $1",
            [match_id]
        ).fetchone()

        if not m:
            return JSONResponse({"error": "Match not found"}, 404)

        pred_elo = db.execute("""
            SELECT p.p_home, p.p_draw, p.p_away, p.eg_home, p.eg_away, p.notes,
                   eh.rating as home_elo, ea.rating as away_elo
            FROM predictions p
            LEFT JOIN elo_state eh ON eh.team = $2
            LEFT JOIN elo_state ea ON ea.team = $3
            WHERE p.match_id = $1 AND p.model_version = $4
        """, [match_id, m[1], m[2], model]).fetchone()

        odds_row = db.execute(
            "SELECT b365h, b365d, b365a FROM match_extras WHERE match_id = $1",
            [match_id]
        ).fetchone()

        result = {
            "match_id": m[0],
            "home_team": m[1],
            "away_team": m[2],
            "competition": m[3],
            "utc_date": str(m[4])[:19] if m[4] else None,
            "status": m[5],
            "score": None,
        }

        if m[5] == "FINISHED" and m[6] is not None and m[7] is not None:
            result["score"] = {"home": int(m[6]), "away": int(m[7])}

        if pred_elo and pred_elo[0] is not None:
            notes = parse_notes(pred_elo[5])
            result["prediction"] = {
                "p_home": float(pred_elo[0]),
                "p_draw": float(pred_elo[1]),
                "p_away": float(pred_elo[2]),
                "btts": float(notes["btts"]) if notes.get("btts") is not None else None,
                "o25": float(notes["o25"]) if notes.get("o25") is not None else None,
                "eg_home": float(pred_elo[3]) if pred_elo[3] is not None else None,
                "eg_away": float(pred_elo[4]) if pred_elo[4] is not None else None,
            }

        if odds_row and odds_row[0]:
            result["odds"] = {
                "home": float(odds_row[0]),
                "draw": float(odds_row[1]),
                "away": float(odds_row[2]),
            }
        else:
            result["odds"] = None

        result["elo"] = {
            "home": float(pred_elo[6]) if pred_elo and pred_elo[6] else None,
            "away": float(pred_elo[7]) if pred_elo and pred_elo[7] else None,
        }

        result["poisson"] = {
            "home": float(pred_elo[3]) if pred_elo and pred_elo[3] is not None else 1.5,
            "away": float(pred_elo[4]) if pred_elo and pred_elo[4] is not None else 1.2,
        }

        return result
    except Exception as e:
        return safe_error(e, "match detail")


@router.get("/matches/{match_id}/models")
async def api_match_models(match_id: int):
    """Return predictions from multiple model variants for comparison."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        match = db.execute("SELECT match_id FROM matches WHERE match_id = $1", [match_id]).fetchone()
        if not match:
            return JSONResponse({"error": "Match not found"}, 404)

        rows = db.execute("""
            SELECT model_version, p_home, p_draw, p_away
            FROM predictions WHERE match_id = $1 AND p_home IS NOT NULL
            ORDER BY model_version ASC
        """, [match_id]).fetchall()

        return {"models": [
            {
                "model": r[0],
                "p_home": round(float(r[1]), 4) if r[1] is not None else None,
                "p_draw": round(float(r[2]), 4) if r[2] is not None else None,
                "p_away": round(float(r[3]), 4) if r[3] is not None else None,
            }
            for r in rows
        ]}
    except Exception as e:
        return safe_error(e, "match models")


@router.get("/matches/{match_id}/experts")
async def api_match_experts(match_id: int):
    """Get expert predictions for a specific match."""
    try:
        db = con()
    except Exception:
        return {"experts": {}}

    try:
        row = db.execute(
            "SELECT breakdown_json FROM expert_cache WHERE match_id = $1 LIMIT 1",
            [match_id]
        ).fetchone()

        if row and row[0]:
            data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            experts_raw = data.get("experts", {})
            experts = {}
            for name, info in experts_raw.items():
                probs = info.get("probs", {})
                experts[name] = {
                    "probs": {
                        "home": round(float(probs.get("home", 0.33)), 4),
                        "draw": round(float(probs.get("draw", 0.33)), 4),
                        "away": round(float(probs.get("away", 0.34)), 4),
                    },
                    "confidence": round(float(info.get("confidence", 0.5)), 4),
                }
            return {"experts": experts}

        # Fallback: parse from prediction notes
        pred = db.execute(
            "SELECT notes FROM predictions WHERE match_id = $1 AND notes IS NOT NULL LIMIT 1",
            [match_id]
        ).fetchone()

        if pred and pred[0]:
            notes = json.loads(pred[0]) if isinstance(pred[0], str) else pred[0]
            expert_data = notes.get("experts", {})
            experts = {}
            for name, info in expert_data.items():
                if isinstance(info, dict):
                    experts[name] = {
                        "probs": {
                            "home": round(float(info.get("home", 0.33)), 4),
                            "draw": round(float(info.get("draw", 0.33)), 4),
                            "away": round(float(info.get("away", 0.34)), 4),
                        },
                        "confidence": round(float(info.get("confidence", 0.5)), 4),
                    }
            return {"experts": experts}

        return {"experts": {}}
    except Exception:
        return {"experts": {}}


@router.get("/matches/{match_id}/h2h")
async def api_match_h2h(match_id: int):
    """Get head-to-head history between two teams."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        m = db.execute("SELECT home_team, away_team FROM matches WHERE match_id = $1", [match_id]).fetchone()
        if not m:
            return JSONResponse({"error": "Match not found"}, 404)

        home_team = canonical_team_name(m[0])
        away_team = canonical_team_name(m[1])

        rows = db.execute("""
            SELECT home_team, away_team, home_goals, away_goals, utc_date, status
            FROM matches
            WHERE ((home_team = $1 AND away_team = $2) OR (home_team = $2 AND away_team = $1))
              AND status = 'FINISHED'
            ORDER BY utc_date DESC LIMIT 10
        """, [home_team, away_team]).fetchall()

        h2h = []
        home_wins, away_wins, draws = 0, 0, 0
        for r in rows:
            h2h.append({
                "home_team": r[0], "away_team": r[1],
                "home_goals": int(r[2]) if r[2] is not None else None,
                "away_goals": int(r[3]) if r[3] is not None else None,
                "date": str(r[4])[:10] if r[4] else None,
            })
            if r[2] is not None and r[3] is not None:
                if r[2] > r[3]:
                    home_wins += 1
                elif r[3] > r[2]:
                    away_wins += 1
                else:
                    draws += 1

        return {"h2h": h2h, "summary": {"home_wins": home_wins, "away_wins": away_wins, "draws": draws}}
    except Exception as e:
        return safe_error(e, "h2h")


@router.get("/matches/{match_id}/form")
async def api_match_form(match_id: int):
    """Get recent form for both teams."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        m = db.execute("SELECT home_team, away_team FROM matches WHERE match_id = $1", [match_id]).fetchone()
        if not m:
            return JSONResponse({"error": "Match not found"}, 404)

        home_team = canonical_team_name(m[0])
        away_team = canonical_team_name(m[1])

        def _get_form(team):
            rows = db.execute("""
                SELECT utc_date, home_team, away_team, home_goals, away_goals
                FROM matches WHERE (home_team = $1 OR away_team = $1) AND status = 'FINISHED' AND home_goals IS NOT NULL
                ORDER BY utc_date DESC LIMIT 10
            """, [team]).fetchall()
            out = []
            for r in rows:
                ht, at = r[1], r[2]
                hg, ag = int(r[3]), int(r[4])
                is_home = (canonical_team_name(ht) == team)
                result = ("W" if hg > ag else ("D" if hg == ag else "L")) if is_home else ("W" if ag > hg else ("D" if ag == hg else "L"))
                out.append({
                    "date": str(r[0])[:10] if r[0] else None,
                    "opponent": at if is_home else ht,
                    "home_goals": hg, "away_goals": ag,
                    "result": result,
                    "venue": "home" if is_home else "away",
                })
            return out

        return {"home_form": _get_form(home_team), "away_form": _get_form(away_team)}
    except Exception as e:
        return safe_error(e, "form")


@router.get("/matches/{match_id}/xg")
async def api_match_xg(match_id: int):
    """Get expected goals data for a match."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        row = db.execute("SELECT eg_home, eg_away FROM predictions WHERE match_id = $1 LIMIT 1", [match_id]).fetchone()
        if not row:
            return {"home_xg": None, "away_xg": None}
        return {
            "home_xg": float(row[0]) if row[0] is not None else None,
            "away_xg": float(row[1]) if row[1] is not None else None,
        }
    except Exception as e:
        return safe_error(e, "xg")


@router.get("/matches/{match_id}/patterns")
async def api_match_patterns(match_id: int):
    """Get statistical patterns and trends for a match."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        m = db.execute("SELECT home_team, away_team FROM matches WHERE match_id = $1", [match_id]).fetchone()
        if not m:
            return JSONResponse({"error": "Match not found"}, 404)

        home_team = canonical_team_name(m[0])
        away_team = canonical_team_name(m[1])

        h2h = db.execute("""
            SELECT home_goals, away_goals FROM matches
            WHERE ((home_team = $1 AND away_team = $2) OR (home_team = $2 AND away_team = $1))
              AND status = 'FINISHED'
            ORDER BY utc_date DESC LIMIT 20
        """, [home_team, away_team]).fetchall()

        patterns = {}
        trends = {}

        if h2h:
            total_games = len(h2h)
            goals_scored = sum(r[0] + r[1] for r in h2h if r[0] is not None and r[1] is not None)
            avg_goals = goals_scored / total_games if total_games > 0 else 0
            patterns["avg_goals_per_game"] = round(avg_goals, 2)
            patterns["total_games"] = total_games

            o25_games = sum(1 for r in h2h if r[0] is not None and r[1] is not None and r[0] + r[1] > 2.5)
            patterns["over_2_5_percentage"] = round(o25_games / total_games * 100, 1) if total_games > 0 else 0

            recent = h2h[:5]
            recent_o25 = sum(1 for r in recent if r[0] is not None and r[1] is not None and r[0] + r[1] > 2.5)
            trends["recent_over_2_5"] = round(recent_o25 / 5 * 100, 1) if recent else 0
            trends["trend_direction"] = "increasing" if patterns["over_2_5_percentage"] > 50 else "decreasing"
            strength = "high" if total_games >= 10 else "moderate" if total_games >= 5 else "low"
        else:
            strength = "low"

        return {"patterns": patterns, "trends": trends, "statistical_strength": strength}
    except Exception as e:
        return safe_error(e, "patterns")


@router.get("/matches/{match_id}/ai")
async def api_match_ai(match_id: int):
    """Generate AI narrative analysis for a match."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        m = db.execute(
            "SELECT home_team, away_team, competition FROM matches WHERE match_id = $1", [match_id]
        ).fetchone()
        if not m:
            return JSONResponse({"error": "Match not found"}, 404)

        p = db.execute(
            "SELECT p_home, p_draw, p_away, eg_home, eg_away FROM predictions WHERE match_id = $1 LIMIT 1",
            [match_id]
        ).fetchone()

        if not p or p[0] is None:
            return {"narrative": "No prediction data available for AI analysis."}

        ph, pd, pa = float(p[0]), float(p[1]), float(p[2])
        eg_h, eg_a = float(p[3]) if p[3] else 0, float(p[4]) if p[4] else 0
        max_prob = max(ph, pd, pa)
        conf_level = "high" if max_prob > 0.55 else "moderate" if max_prob > 0.42 else "balanced"

        if eg_h > eg_a + 0.5:
            goal_analysis = f"an attacking setup favoring {m[0]}"
        elif eg_a > eg_h + 0.5:
            goal_analysis = f"an attacking setup favoring {m[1]}"
        else:
            goal_analysis = "balanced attacking intent"

        narrative = (
            f"{m[0]} vs {m[1]} ({m[2]})\n\n"
            f"Model Forecast: {m[0]} {round(ph*100)}%, Draw {round(pd*100)}%, {m[1]} {round(pa*100)}%\n\n"
            f"Expected Goals: {m[0]} {eg_h:.2f} xG, {m[1]} {eg_a:.2f} xG\n\n"
            f"This match carries {conf_level} predictive confidence. "
            f"The expected goal differential suggests {goal_analysis}.\n\n"
            f"Key considerations: Recent form trends, head-to-head historical patterns, "
            f"and squad availability will be crucial factors."
        )
        return {"narrative": narrative}
    except Exception as e:
        return safe_error(e, "ai analysis")
