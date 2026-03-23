"""Advanced endpoints: Season simulation, team profile, streaks, prediction history."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from footy.normalize import canonical_team_name
from . import con, safe_error, validate_competition

router = APIRouter(prefix="/api", tags=["advanced"])


@router.get("/season-simulation/{comp}")
async def api_season_simulation(comp: str):
    """Monte Carlo season simulation with Poisson-distributed goals."""
    import numpy as np

    if validate_competition(comp) is None:
        return JSONResponse({"error": "Invalid competition"}, 400)

    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        standings_rows = db.execute("""
            WITH season_matches AS (
                SELECT home_team, away_team, home_goals, away_goals
                FROM matches
                WHERE competition = $1 AND status = 'FINISHED' AND home_goals IS NOT NULL
                  AND utc_date >= (SELECT MAX(utc_date) - INTERVAL 365 DAY FROM matches WHERE competition = $1 AND status = 'FINISHED')
            ),
            home AS (
                SELECT home_team AS team, COUNT(*) AS p,
                       SUM(CASE WHEN home_goals > away_goals THEN 3 WHEN home_goals = away_goals THEN 1 ELSE 0 END) AS pts,
                       SUM(home_goals) AS gf, SUM(away_goals) AS ga
                FROM season_matches GROUP BY home_team
            ),
            away AS (
                SELECT away_team AS team, COUNT(*) AS p,
                       SUM(CASE WHEN away_goals > home_goals THEN 3 WHEN away_goals = home_goals THEN 1 ELSE 0 END) AS pts,
                       SUM(away_goals) AS gf, SUM(home_goals) AS ga
                FROM season_matches GROUP BY away_team
            )
            SELECT COALESCE(h.team, a.team) AS team_name,
                   COALESCE(h.pts,0)+COALESCE(a.pts,0) AS points,
                   COALESCE(h.gf,0)+COALESCE(a.gf,0) AS goals_for,
                   COALESCE(h.ga,0)+COALESCE(a.ga,0) AS goals_against,
                   COALESCE(h.p,0)+COALESCE(a.p,0) AS played
            FROM home h FULL OUTER JOIN away a ON h.team = a.team
            ORDER BY points DESC, (COALESCE(h.gf,0)+COALESCE(a.gf,0)-COALESCE(h.ga,0)-COALESCE(a.ga,0)) DESC
        """, [comp]).fetchall()

        if not standings_rows:
            return {"competition": comp, "teams": []}

        teams = [r[0] for r in standings_rows]
        current_pts = {r[0]: int(r[1] or 0) for r in standings_rows}
        gf = {r[0]: int(r[2] or 0) for r in standings_rows}
        ga = {r[0]: int(r[3] or 0) for r in standings_rows}
        played = {r[0]: int(r[4] or 0) for r in standings_rows}

        remaining = db.execute("""
            SELECT home_team, away_team FROM matches
            WHERE competition = $1 AND status IN ('UPCOMING', 'SCHEDULED')
        """, [comp]).fetchall()

        total_played = sum(played.values()) / 2 or 1
        avg_goals = (sum(gf.values())) / (total_played * 2) if total_played > 0 else 1.3

        def team_attack(t):
            p = played.get(t, 1) or 1
            return (gf.get(t, 0) / p) / avg_goals if avg_goals > 0 else 1.0

        def team_defense(t):
            p = played.get(t, 1) or 1
            return (ga.get(t, 0) / p) / avg_goals if avg_goals > 0 else 1.0

        n_sims = 1000
        title_count = {t: 0 for t in teams}
        top4_count = {t: 0 for t in teams}
        releg_count = {t: 0 for t in teams}
        n_teams = len(teams)
        releg_zone = max(n_teams - 3, 0)

        for _ in range(n_sims):
            sim_pts = dict(current_pts)
            for fixture in remaining:
                ht, at = fixture[0], fixture[1]
                if ht not in sim_pts or at not in sim_pts:
                    continue
                lam_h = avg_goals * team_attack(ht) * team_defense(at) * 1.1
                lam_a = avg_goals * team_attack(at) * team_defense(ht)
                gh = int(np.random.poisson(max(lam_h, 0.1)))
                ga_ = int(np.random.poisson(max(lam_a, 0.1)))
                if gh > ga_:
                    sim_pts[ht] += 3
                elif gh == ga_:
                    sim_pts[ht] += 1
                    sim_pts[at] += 1
                else:
                    sim_pts[at] += 3

            ranked = sorted(sim_pts.keys(), key=lambda t: sim_pts[t], reverse=True)
            title_count[ranked[0]] += 1
            for t in ranked[:4]:
                top4_count[t] += 1
            if releg_zone < n_teams:
                for t in ranked[releg_zone:]:
                    releg_count[t] += 1

        pos_map = {r[0]: idx + 1 for idx, r in enumerate(standings_rows)}
        result_teams = []
        for t in teams:
            result_teams.append({
                "team": t,
                "p_title": round(title_count[t] / n_sims, 4),
                "p_top4": round(top4_count[t] / n_sims, 4),
                "p_relegation": round(releg_count[t] / n_sims, 4),
                "current_pts": current_pts[t],
                "current_pos": pos_map.get(t, 0),
            })
        result_teams.sort(key=lambda x: x["current_pos"])
        return {"competition": comp, "teams": result_teams}
    except Exception as e:
        return safe_error(e, "season simulation")


@router.get("/team/{team_name}/profile")
async def api_team_profile(team_name: str):
    """Comprehensive team profile with Elo, form, recent matches, and upcoming fixtures."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        canonical = canonical_team_name(team_name)

        elo_row = db.execute("SELECT rating FROM elo_state WHERE team = $1", [canonical]).fetchone()
        elo = round(float(elo_row[0]), 1) if elo_row and elo_row[0] is not None else 1500.0

        recent_rows = db.execute("""
            SELECT match_id, home_team, away_team, competition, utc_date, home_goals, away_goals, status
            FROM matches WHERE (home_team = $1 OR away_team = $1) AND status = 'FINISHED' AND home_goals IS NOT NULL
            ORDER BY utc_date DESC LIMIT 20
        """, [canonical]).fetchall()

        recent_matches = []
        form_results = []
        goals_scored = 0
        goals_conceded = 0

        for r in recent_rows:
            is_home = r[1] == canonical
            hs = int(r[5]) if r[5] is not None else 0
            as_ = int(r[6]) if r[6] is not None else 0
            if is_home:
                goals_scored += hs
                goals_conceded += as_
                result = "W" if hs > as_ else ("D" if hs == as_ else "L")
            else:
                goals_scored += as_
                goals_conceded += hs
                result = "W" if as_ > hs else ("D" if as_ == hs else "L")

            if len(form_results) < 5:
                form_results.append(result)
            recent_matches.append({
                "match_id": int(r[0]), "home_team": r[1], "away_team": r[2],
                "competition": r[3], "date": str(r[4])[:10],
                "home_score": hs, "away_score": as_, "result": result,
            })

        upcoming_rows = db.execute("""
            SELECT match_id, home_team, away_team, competition, utc_date
            FROM matches WHERE (home_team = $1 OR away_team = $1) AND status IN ('UPCOMING', 'SCHEDULED')
            ORDER BY utc_date ASC LIMIT 10
        """, [canonical]).fetchall()

        upcoming = [
            {"match_id": int(r[0]), "home_team": r[1], "away_team": r[2],
             "competition": r[3], "date": str(r[4])[:10]}
            for r in upcoming_rows
        ]

        return {
            "team": canonical, "elo": elo,
            "form": "".join(form_results),
            "recent_matches": recent_matches,
            "upcoming": upcoming,
            "stats": {"goals_scored": goals_scored, "goals_conceded": goals_conceded},
        }
    except Exception as e:
        return safe_error(e, "team profile")


@router.get("/streaks")
async def api_streaks():
    """Current winning, losing, and unbeaten streaks across all teams."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        rows = db.execute("""
            SELECT home_team, away_team, home_goals, away_goals
            FROM matches WHERE status = 'FINISHED' AND home_goals IS NOT NULL
            ORDER BY utc_date DESC LIMIT 2000
        """).fetchall()

        win_streak: dict[str, int] = {}
        lose_streak: dict[str, int] = {}
        unbeaten_streak: dict[str, int] = {}
        win_active: dict[str, bool] = {}
        lose_active: dict[str, bool] = {}
        unbeaten_active: dict[str, bool] = {}
        all_teams: set[str] = set()

        for r in rows:
            ht, at = r[0], r[1]
            hs = int(r[2]) if r[2] is not None else 0
            as_ = int(r[3]) if r[3] is not None else 0

            for team in (ht, at):
                if team not in all_teams:
                    all_teams.add(team)
                    win_streak[team] = 0
                    lose_streak[team] = 0
                    unbeaten_streak[team] = 0
                    win_active[team] = True
                    lose_active[team] = True
                    unbeaten_active[team] = True

            h_result = "W" if hs > as_ else ("D" if hs == as_ else "L")
            a_result = "W" if as_ > hs else ("D" if as_ == hs else "L")

            for team, result in [(ht, h_result), (at, a_result)]:
                if win_active[team]:
                    if result == "W":
                        win_streak[team] += 1
                    else:
                        win_active[team] = False
                if lose_active[team]:
                    if result == "L":
                        lose_streak[team] += 1
                    else:
                        lose_active[team] = False
                if unbeaten_active[team]:
                    if result in ("W", "D"):
                        unbeaten_streak[team] += 1
                    else:
                        unbeaten_active[team] = False

        return {
            "winning": sorted([{"team": t, "streak": s} for t, s in win_streak.items() if s > 0],
                              key=lambda x: x["streak"], reverse=True)[:10],
            "losing": sorted([{"team": t, "streak": s} for t, s in lose_streak.items() if s > 0],
                             key=lambda x: x["streak"], reverse=True)[:10],
            "unbeaten": sorted([{"team": t, "streak": s} for t, s in unbeaten_streak.items() if s > 0],
                               key=lambda x: x["streak"], reverse=True)[:10],
        }
    except Exception as e:
        return safe_error(e, "streaks")


@router.get("/predictions/history")
async def api_predictions_history():
    """Historical prediction accuracy over time grouped by ISO week."""
    try:
        db = con()
    except Exception:
        return JSONResponse({"error": "Database busy"}, 503)

    try:
        rows = db.execute("""
            SELECT CONCAT(EXTRACT(ISOYEAR FROM m.utc_date)::INT, '-W',
                          LPAD(EXTRACT(WEEK FROM m.utc_date)::INT::VARCHAR, 2, '0')) AS week_label,
                   COUNT(*) AS n_matches,
                   COUNT(CASE WHEN ps.correct = TRUE THEN 1 END) * 1.0 / COUNT(*) AS accuracy,
                   AVG(ps.logloss) AS avg_logloss,
                   AVG(ps.brier) AS avg_brier
            FROM prediction_scores ps
            JOIN matches m ON ps.match_id = m.match_id
            WHERE ps.model_version = 'v13_oracle' AND m.status = 'FINISHED'
            GROUP BY week_label ORDER BY week_label ASC
        """).fetchall()

        return {"weeks": [
            {
                "week": r[0],
                "n_matches": int(r[1]) if r[1] else 0,
                "accuracy": round(float(r[2]), 4) if r[2] is not None else 0.0,
                "logloss": round(float(r[3]), 4) if r[3] is not None else 1.0,
                "brier": round(float(r[4]), 4) if r[4] is not None else 0.25,
            }
            for r in rows
        ]}
    except Exception as e:
        return safe_error(e, "predictions history")
