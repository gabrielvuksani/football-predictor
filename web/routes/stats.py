"""Statistics, league tables, performance, and dashboard endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from . import con, safe_error, validate_model, validate_competition

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats/dashboard")
async def api_dashboard_stats():
    """Summary dashboard statistics with caching support."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        total_preds = db.execute(
            "SELECT COUNT(*) FROM predictions WHERE model_version = 'v13_oracle'"
        ).fetchone()[0] or 0

        avg_conf = db.execute(
            "SELECT AVG(GREATEST(p_home, p_draw, p_away)) FROM predictions WHERE model_version = 'v13_oracle' AND p_home IS NOT NULL"
        ).fetchone()[0] or 0.5

        top_pred = db.execute("""
            SELECT m.home_team, m.away_team, p.p_home, p.p_draw, p.p_away
            FROM predictions p JOIN matches m ON p.match_id = m.match_id
            WHERE p.model_version = 'v13_oracle' AND p.p_home IS NOT NULL
            ORDER BY GREATEST(p.p_home, p.p_draw, p.p_away) DESC LIMIT 1
        """).fetchone()

        top_match = None
        if top_pred:
            top_match = {
                "home_team": top_pred[0], "away_team": top_pred[1],
                "p_home": round(float(top_pred[2]), 3),
                "p_draw": round(float(top_pred[3]), 3),
                "p_away": round(float(top_pred[4]), 3),
            }

        accuracy_row = db.execute("""
            SELECT CASE WHEN COUNT(*) > 0
                        THEN COUNT(CASE WHEN ps.correct = TRUE THEN 1 END) * 100.0 / COUNT(*)
                        ELSE 0 END
            FROM prediction_scores ps JOIN matches m ON ps.match_id = m.match_id
            WHERE ps.model_version = 'v13_oracle' AND m.status = 'FINISHED'
              AND m.home_goals IS NOT NULL AND m.utc_date > NOW() - INTERVAL 30 DAY
        """).fetchone()
        accuracy = accuracy_row[0] if accuracy_row and accuracy_row[0] else 0

        recent_correct = db.execute("""
            SELECT COUNT(CASE WHEN ps.correct = TRUE THEN 1 END), COUNT(*)
            FROM prediction_scores ps JOIN matches m ON ps.match_id = m.match_id
            WHERE ps.model_version = 'v13_oracle' AND m.status = 'FINISHED'
              AND m.home_goals IS NOT NULL AND m.utc_date > NOW() - INTERVAL 7 DAY
        """).fetchone()
        recent_acc = (recent_correct[0] or 0) / max(1, recent_correct[1] or 1)

        return {
            "total_predictions": total_preds,
            "avg_confidence": round(float(avg_conf), 3),
            "accuracy_30d": round(float(accuracy), 1),
            "accuracy_7d": round(recent_acc * 100, 1),
            "top_prediction": top_match,
            "cached": True,
        }
    except Exception as e:
        return safe_error(e, "dashboard stats")


@router.get("/stats")
async def api_stats():
    """Get general statistics about the system."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        total = db.execute("SELECT COUNT(*) FROM matches WHERE status != 'CANCELLED'").fetchone()[0] or 0
        upcoming = db.execute("SELECT COUNT(*) FROM matches WHERE status = 'UPCOMING'").fetchone()[0] or 0
        finished = db.execute("SELECT COUNT(*) FROM matches WHERE status = 'FINISHED'").fetchone()[0] or 0

        top_pred = db.execute("""
            SELECT m.home_team, m.away_team, MAX(GREATEST(p.p_home, p.p_draw, p.p_away))
            FROM matches m JOIN predictions p ON m.match_id = p.match_id
            WHERE p.model_version = 'v13_oracle'
            GROUP BY m.match_id, m.home_team, m.away_team
            ORDER BY MAX(GREATEST(p.p_home, p.p_draw, p.p_away)) DESC LIMIT 1
        """).fetchone()

        avg_conf_row = db.execute("""
            SELECT AVG(confidence), COUNT(*)
            FROM predictions WHERE model_version = 'v13_oracle' AND confidence IS NOT NULL
        """).fetchone()
        avg_confidence = round(float(avg_conf_row[0]), 4) if avg_conf_row and avg_conf_row[0] is not None else 0.0
        n_predictions = int(avg_conf_row[1]) if avg_conf_row and avg_conf_row[1] else 0

        teams_row = db.execute("""
            SELECT COUNT(DISTINCT team) FROM (
                SELECT home_team AS team FROM matches UNION SELECT away_team AS team FROM matches
            )
        """).fetchone()
        n_teams = int(teams_row[0]) if teams_row and teams_row[0] else 0

        return {
            "total_matches": total, "upcoming": upcoming, "finished": finished,
            "strongest_prediction": f"{top_pred[0]} vs {top_pred[1]}" if top_pred else None,
            "avg_confidence": avg_confidence, "teams": n_teams, "predictions": n_predictions,
        }
    except Exception as e:
        return safe_error(e, "stats")


@router.get("/performance")
async def api_performance(model: str = Query("v13_oracle")):
    """Get model performance metrics."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        model = validate_model(model)
        row = db.execute("""
            SELECT COUNT(*),
                   CASE WHEN COUNT(*) > 0
                        THEN COUNT(CASE WHEN ps.correct = TRUE THEN 1 END) * 100.0 / COUNT(*)
                        ELSE 0 END,
                   AVG(ps.logloss)
            FROM prediction_scores ps JOIN matches m ON ps.match_id = m.match_id
            WHERE ps.model_version = $1 AND m.status = 'FINISHED' AND m.home_goals IS NOT NULL
        """, [model]).fetchone()

        accuracy = float(row[1]) if row[1] is not None else 0
        return {
            "model": model,
            "metrics": {
                "accuracy": round(accuracy, 1),
                "total_predictions": int(row[0]) if row[0] else 0,
                "log_loss": round(float(row[2]) if row[2] is not None else 1.0, 4),
            },
            "calibration": {}, "by_competition": {}, "recent": {},
        }
    except Exception as e:
        return safe_error(e, "performance")


@router.get("/league-table/{competition}")
async def api_league_table(competition: str):
    """Get current league standings for a competition."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        if validate_competition(competition) is None:
            return JSONResponse({"error": "Invalid competition"}, 400)

        rows = db.execute("""
            WITH season_matches AS (
                SELECT home_team, away_team, home_goals, away_goals
                FROM matches
                WHERE competition = $1 AND status = 'FINISHED' AND home_goals IS NOT NULL
                  AND utc_date >= (SELECT MAX(utc_date) - INTERVAL 365 DAY FROM matches WHERE competition = $1 AND status = 'FINISHED')
            ),
            home AS (
                SELECT home_team AS team, COUNT(*) AS p,
                       SUM(CASE WHEN home_goals > away_goals THEN 1 ELSE 0 END) AS w,
                       SUM(CASE WHEN home_goals = away_goals THEN 1 ELSE 0 END) AS d,
                       SUM(CASE WHEN home_goals < away_goals THEN 1 ELSE 0 END) AS l,
                       SUM(CASE WHEN home_goals > away_goals THEN 3 WHEN home_goals = away_goals THEN 1 ELSE 0 END) AS pts,
                       SUM(home_goals) AS gf, SUM(away_goals) AS ga
                FROM season_matches GROUP BY home_team
            ),
            away AS (
                SELECT away_team AS team, COUNT(*) AS p,
                       SUM(CASE WHEN away_goals > home_goals THEN 1 ELSE 0 END) AS w,
                       SUM(CASE WHEN away_goals = home_goals THEN 1 ELSE 0 END) AS d,
                       SUM(CASE WHEN away_goals < home_goals THEN 1 ELSE 0 END) AS l,
                       SUM(CASE WHEN away_goals > home_goals THEN 3 WHEN away_goals = home_goals THEN 1 ELSE 0 END) AS pts,
                       SUM(away_goals) AS gf, SUM(home_goals) AS ga
                FROM season_matches GROUP BY away_team
            )
            SELECT COALESCE(h.team, a.team) AS team_name,
                   COALESCE(h.p,0)+COALESCE(a.p,0) AS played,
                   COALESCE(h.w,0)+COALESCE(a.w,0) AS won,
                   COALESCE(h.d,0)+COALESCE(a.d,0) AS drawn,
                   COALESCE(h.l,0)+COALESCE(a.l,0) AS lost,
                   COALESCE(h.gf,0)+COALESCE(a.gf,0) AS goals_for,
                   COALESCE(h.ga,0)+COALESCE(a.ga,0) AS goals_against,
                   (COALESCE(h.gf,0)+COALESCE(a.gf,0))-(COALESCE(h.ga,0)+COALESCE(a.ga,0)) AS goal_difference,
                   COALESCE(h.pts,0)+COALESCE(a.pts,0) AS points
            FROM home h FULL OUTER JOIN away a ON h.team = a.team
            ORDER BY points DESC, goal_difference DESC, goals_for DESC
        """, [competition]).fetchall()

        standings = []
        for idx, r in enumerate(rows):
            standings.append({
                "position": idx + 1, "team_id": None,
                "team_name": r[0],
                "played": int(r[1] or 0), "won": int(r[2] or 0),
                "drawn": int(r[3] or 0), "lost": int(r[4] or 0),
                "goals_for": int(r[5] or 0), "goals_against": int(r[6] or 0),
                "goal_difference": int(r[7] or 0), "points": int(r[8] or 0),
            })
        return {"standings": standings}
    except Exception as e:
        return safe_error(e, "league table")
