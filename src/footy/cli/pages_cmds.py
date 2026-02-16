"""Export static JSON snapshots for GitHub Pages deployment.

Usage:
    footy pages export           # Export all data to docs/api/
    footy pages export --out /tmp/docs  # Custom output directory
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import typer

from footy.cli._shared import console

log = logging.getLogger(__name__)

app = typer.Typer(help="GitHub Pages static site management.")


def _dump(path: Path, data: dict | list) -> None:
    """Write JSON to file, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str), encoding="utf-8")


@app.command()
def export(
    out: str = typer.Option(
        "docs", help="Output directory for the static site."
    ),
    days: int = typer.Option(14, help="Days of matches to include."),
):
    """Export live database to static JSON files for GitHub Pages.

    Snapshots the current predictions, value bets, league tables,
    performance metrics, and per-match details into docs/api/*.json
    so the static frontend can serve them without a backend.
    """
    import duckdb
    from footy.config import settings

    root = Path(out)
    api = root / "api"

    # Ensure api dir exists (keep other docs/ files untouched)
    api.mkdir(parents=True, exist_ok=True)

    # Clean stale match detail files before re-export
    matches_dir = api / "matches"
    if matches_dir.exists():
        shutil.rmtree(matches_dir)
        log.debug("Cleaned stale match detail directory")

    cfg = settings()
    db = duckdb.connect(cfg.db_path, read_only=True)
    console.print("[cyan]Exporting static JSON for GitHub Pages…[/cyan]")

    # ── 1. Matches list ──────────────────────────────────────────────────
    rows = db.execute(f"""
        SELECT m.match_id, m.home_team, m.away_team, m.utc_date,
               m.competition, m.status,
               p.p_home, p.p_draw, p.p_away,
               p.eg_home, p.eg_away,
               e.b365h, e.b365d, e.b365a,
               p.model_version
        FROM matches m
        LEFT JOIN predictions p
          ON m.match_id = p.match_id AND p.model_version = 'v8_council'
        LEFT JOIN match_extras e ON m.match_id = e.match_id
        WHERE m.status IN ('SCHEDULED', 'TIMED')
          AND m.utc_date >= CURRENT_DATE - INTERVAL {int(days)} DAY
          AND m.utc_date <= CURRENT_DATE + INTERVAL 30 DAY
        ORDER BY m.utc_date
    """).fetchall()

    cols = [
        "match_id", "home_team", "away_team", "utc_date",
        "competition", "status",
        "p_home", "p_draw", "p_away", "eg_home", "eg_away",
        "b365h", "b365d", "b365a", "model_version",
    ]
    matches = [dict(zip(cols, r)) for r in rows]
    _dump(api / "matches.json", {"matches": matches, "model": "v8_council"})
    console.print(f"  matches.json — {len(matches)} matches")

    # ── 2. Per-match detail + experts + h2h + form ────────────────────
    match_ids = [m["match_id"] for m in matches]
    for mid in match_ids:
        _export_match_detail(db, api, mid)
    console.print(f"  match details — {len(match_ids)} matches")

    # ── 3. Value bets ─────────────────────────────────────────────────
    bets = _build_value_bets(matches)
    _dump(api / "value-bets.json", {"bets": bets})
    console.print(f"  value-bets.json — {len(bets)} bets")

    # ── 4. League tables ──────────────────────────────────────────────
    comps = db.execute(
        "SELECT DISTINCT competition FROM matches WHERE status IN ('SCHEDULED', 'TIMED')"
    ).fetchall()
    comp_list = sorted(set(c[0] for c in comps))
    for comp in comp_list:
        standings = _build_league_table(db, comp)
        _dump(api / "league-table" / f"{comp}.json", {"standings": standings})
    console.print(f"  league-table/ — {len(comp_list)} leagues")

    # ── 5. Stats ──────────────────────────────────────────────────────
    stats = _build_stats(db)
    _dump(api / "stats.json", stats)
    console.print("  stats.json")

    # ── 6. Performance ────────────────────────────────────────────────
    perf = _build_performance(db)
    _dump(api / "performance.json", perf)
    console.print("  performance.json")

    # ── 7. Competitions ───────────────────────────────────────────────
    _dump(api / "competitions.json", {"competitions": comp_list})
    console.print("  competitions.json")

    # ── 8. Last updated ───────────────────────────────────────────────
    from datetime import datetime, timezone

    _dump(api / "last-updated.json", {
        "last_updated": datetime.now(timezone.utc).isoformat(),
    })
    console.print("  last-updated.json")

    db.close()
    console.print(f"\n[green]✓ Static export complete → {root.resolve()}[/green]")
    console.print("  Push to GitHub and enable Pages (source: GitHub Actions) to deploy.")


# ─── Helpers ──────────────────────────────────────────────────────────────

def _export_match_detail(db, api: Path, match_id: int) -> None:
    """Export detail, experts, h2h, form for a single match.

    Matches the exact JSON structure that the live API returns so the
    static frontend (docs/app.js) renders identically.
    """
    from footy.normalize import canonical_team_name

    row = db.execute("""
        SELECT m.match_id, m.home_team, m.away_team, m.utc_date,
               m.competition, m.status, m.home_goals, m.away_goals
        FROM matches m
        WHERE m.match_id = ?
    """, [match_id]).fetchone()
    if not row:
        return

    detail = {
        "match_id": row[0],
        "home_team": row[1],
        "away_team": row[2],
        "utc_date": str(row[3])[:16],
        "competition": row[4],
        "status": row[5],
        "home_goals": row[6],
        "away_goals": row[7],
    }

    # Prediction
    pred_row = db.execute(
        "SELECT p_home, p_draw, p_away, eg_home, eg_away, notes "
        "FROM predictions WHERE match_id=? AND model_version='v8_council'",
        [match_id],
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

    detail["prediction"] = prediction
    detail["poisson"] = poisson_stats

    # Odds (match live API structure with implied probabilities)
    odds_row = db.execute(
        "SELECT b365h, b365d, b365a FROM match_extras WHERE match_id=?",
        [match_id],
    ).fetchone()
    odds = None
    if odds_row and odds_row[0]:
        h, d, a = float(odds_row[0]), float(odds_row[1]), float(odds_row[2])
        if h > 0 and d > 0 and a > 0:
            ih, id_, ia = 1 / h, 1 / d, 1 / a
            s = ih + id_ + ia
            odds = {
                "b365h": round(h, 2), "b365d": round(d, 2), "b365a": round(a, 2),
                "implied_h": round(ih / s, 3), "implied_d": round(id_ / s, 3),
                "implied_a": round(ia / s, 3),
                "overround": round(s - 1, 3),
            }
    detail["odds"] = odds

    # Elo ratings
    home_c = canonical_team_name(row[1])
    away_c = canonical_team_name(row[2])
    elo_h = db.execute("SELECT rating FROM elo_state WHERE team=?", [home_c]).fetchone()
    elo_a = db.execute("SELECT rating FROM elo_state WHERE team=?", [away_c]).fetchone()
    detail["elo"] = {
        "home": round(float(elo_h[0]), 0) if elo_h else None,
        "away": round(float(elo_a[0]), 0) if elo_a else None,
        "diff": round(float(elo_h[0] - elo_a[0]), 0) if elo_h and elo_a else None,
    }

    # Prediction score (if match is finished and was scored)
    score_row = db.execute(
        "SELECT outcome, logloss, brier, correct FROM prediction_scores "
        "WHERE match_id=? AND model_version='v8_council'",
        [match_id],
    ).fetchone()
    score = None
    if score_row:
        score = {
            "outcome": ["Home", "Draw", "Away"][score_row[0]],
            "logloss": round(float(score_row[1]), 4),
            "brier": round(float(score_row[2]), 4),
            "correct": bool(score_row[3]),
        }
    detail["score"] = score

    _dump(api / "matches" / f"{match_id}.json", detail)

    # ── Experts (from expert_cache) ───────────────────────────────────
    try:
        cache_row = db.execute(
            "SELECT breakdown_json FROM expert_cache WHERE match_id = ?",
            [match_id],
        ).fetchone()
        if cache_row:
            experts = json.loads(cache_row[0]) if isinstance(cache_row[0], str) else cache_row[0]
            _dump(api / "matches" / str(match_id) / "experts.json", experts)
    except Exception as e:
        log.warning("Failed to export experts for match %s: %s", match_id, e)

    # ── H2H ───────────────────────────────────────────────────────────
    home = detail["home_team"]
    away = detail["away_team"]
    try:
        h2h_row = db.execute("""
            SELECT total_matches, team_a_wins, team_b_wins, draws,
                   team_a, team_b,
                   team_a_avg_goals_for, team_a_avg_goals_against,
                   team_b_avg_goals_for, team_b_avg_goals_against
            FROM h2h_stats
            WHERE (team_a = ? AND team_b = ?)
               OR (team_a = ? AND team_b = ?)
            LIMIT 1
        """, [home, away, away, home]).fetchone()
        if h2h_row:
            stats = {
                "total_matches": h2h_row[0],
                "team_a_wins": h2h_row[1],
                "team_b_wins": h2h_row[2],
                "draws": h2h_row[3],
                "team_a": h2h_row[4],
                "team_b": h2h_row[5],
                "team_a_avg_goals_for": h2h_row[6],
                "team_a_avg_goals_against": h2h_row[7],
                "team_b_avg_goals_for": h2h_row[8],
                "team_b_avg_goals_against": h2h_row[9],
            }
            # Recent matches
            recent_rows = db.execute("""
                SELECT utc_date, home_team, away_team, home_goals, away_goals
                FROM matches
                WHERE status = 'FINISHED'
                  AND ((home_team = ? AND away_team = ?)
                    OR (home_team = ? AND away_team = ?))
                ORDER BY utc_date DESC LIMIT 10
            """, [home, away, away, home]).fetchall()
            recent_matches = [
                {
                    "utc_date": str(r[0])[:10],
                    "home_team": r[1],
                    "away_team": r[2],
                    "home_goals": r[3],
                    "away_goals": r[4],
                }
                for r in recent_rows
            ]
            _dump(api / "matches" / str(match_id) / "h2h.json", {
                "stats": stats,
                "recent_matches": recent_matches,
            })
    except Exception as e:
        log.warning("Failed to export H2H for match %s: %s", match_id, e)

    # ── Form ──────────────────────────────────────────────────────────
    try:
        form = {}
        for team, form_key, ppg_key in [
            (home, "home_form", "home_ppg"),
            (away, "away_form", "away_ppg"),
        ]:
            last_n = db.execute("""
                SELECT home_team, away_team, home_goals, away_goals
                FROM matches
                WHERE status = 'FINISHED'
                  AND (home_team = ? OR away_team = ?)
                ORDER BY utc_date DESC LIMIT 6
            """, [team, team]).fetchall()
            results = []
            for ht, at, hg, ag in last_n:
                if team == ht:
                    res = "W" if hg > ag else "D" if hg == ag else "L"
                else:
                    res = "W" if ag > hg else "D" if hg == ag else "L"
                results.append(res)
            pts = sum(3 if r == "W" else 1 if r == "D" else 0 for r in results)
            form[form_key] = results
            form[ppg_key] = round(pts / max(len(results), 1), 2)
        _dump(api / "matches" / str(match_id) / "form.json", form)
    except Exception as e:
        log.warning("Failed to export form for match %s: %s", match_id, e)


def _build_value_bets(matches: list[dict], min_edge: float = 0.03) -> list[dict]:
    """Identify value bets from the matches list."""
    bets = []
    for m in matches:
        if m.get("p_home") is None:
            continue
        for outcome, prob_key, odds_key, label in [
            ("home", "p_home", "b365h", "Home"),
            ("draw", "p_draw", "b365d", "Draw"),
            ("away", "p_away", "b365a", "Away"),
        ]:
            prob = m[prob_key]
            odds = m.get(odds_key)
            if not odds or odds <= 1 or prob is None:
                continue
            implied = 1 / odds
            edge = prob - implied
            if edge >= min_edge:
                bets.append({
                    "match_id": m["match_id"],
                    "home_team": m["home_team"],
                    "away_team": m["away_team"],
                    "competition": m["competition"],
                    "date": str(m["utc_date"])[:10] if m.get("utc_date") else "",
                    "bet": label,
                    "model_prob": round(prob, 4),
                    "odds": round(odds, 2),
                    "edge": round(edge, 4),
                })
    bets.sort(key=lambda b: b["edge"], reverse=True)
    return bets


def _build_league_table(db, comp: str) -> list[dict]:
    """Build league standings from finished matches in the current season."""
    rows = db.execute("""
        SELECT home_team, away_team, home_goals, away_goals
        FROM matches
        WHERE competition = ? AND status = 'FINISHED'
          AND season = (SELECT MAX(season) FROM matches WHERE competition = ?)
    """, [comp, comp]).fetchall()

    teams: dict[str, dict] = {}
    for ht, at, hg, ag in rows:
        for team in [ht, at]:
            if team not in teams:
                teams[team] = {"team": team, "p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0}
        # Home
        teams[ht]["p"] += 1
        teams[ht]["gf"] += hg
        teams[ht]["ga"] += ag
        # Away
        teams[at]["p"] += 1
        teams[at]["gf"] += ag
        teams[at]["ga"] += hg
        if hg > ag:
            teams[ht]["w"] += 1; teams[ht]["pts"] += 3
            teams[at]["l"] += 1
        elif hg < ag:
            teams[at]["w"] += 1; teams[at]["pts"] += 3
            teams[ht]["l"] += 1
        else:
            teams[ht]["d"] += 1; teams[ht]["pts"] += 1
            teams[at]["d"] += 1; teams[at]["pts"] += 1

    table = sorted(teams.values(), key=lambda t: (-t["pts"], -(t["gf"] - t["ga"]), -t["gf"]))
    for i, t in enumerate(table, 1):
        t["pos"] = i
        t["gd"] = t["gf"] - t["ga"]
    return table


def _build_stats(db) -> dict:
    """Build database statistics summary matching live API /api/stats shape."""
    try:
        row = db.execute("""
            SELECT
                (SELECT COUNT(*) FROM matches) AS total,
                (SELECT COUNT(*) FROM matches WHERE status = 'FINISHED') AS finished,
                (SELECT COUNT(*) FROM matches WHERE status IN ('SCHEDULED', 'TIMED')) AS upcoming,
                (SELECT COUNT(*) FROM predictions) AS predictions,
                (SELECT COUNT(DISTINCT competition) FROM matches) AS leagues,
                (SELECT COUNT(DISTINCT home_team) FROM matches) AS teams
        """).fetchone()
        return {
            "total_matches": row[0],
            "finished": row[1],
            "upcoming": row[2],
            "predictions": row[3],
            "leagues": row[4],
            "teams": row[5],
        }
    except Exception as e:
        log.warning("Failed to build stats: %s", e)
        return {}


def _build_performance(db) -> dict:
    """Build model performance metrics matching live API /api/performance shape."""
    model = "v8_council"
    try:
        # Aggregate metrics
        metrics_row = db.execute(
            "SELECT n_matches, logloss, brier, accuracy FROM metrics WHERE model_version = ?",
            [model],
        ).fetchone()
        metrics = None
        if metrics_row:
            metrics = {
                "n_scored": metrics_row[0],
                "logloss": round(float(metrics_row[1]), 4) if metrics_row[1] else None,
                "brier": round(float(metrics_row[2]), 4) if metrics_row[2] else None,
                "accuracy": round(float(metrics_row[3]), 4) if metrics_row[3] else None,
            }

        # Recent scored predictions
        recent_rows = db.execute("""
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
        recent = []
        for r in recent_rows:
            recent.append({
                "date": str(r[0])[:10],
                "home_team": r[1], "away_team": r[2],
                "home_goals": r[3], "away_goals": r[4],
                "p_home": round(float(r[5]), 3), "p_draw": round(float(r[6]), 3),
                "p_away": round(float(r[7]), 3),
                "outcome": ["Home", "Draw", "Away"][r[8]],
                "logloss": round(float(r[9]), 4),
                "correct": bool(r[10]),
                "competition": r[11],
            })

        # By competition
        by_comp_rows = db.execute("""
            SELECT m.competition, COUNT(*) as n,
                   SUM(CASE WHEN ps.correct THEN 1 ELSE 0 END) as correct,
                   AVG(ps.logloss) as avg_ll
            FROM prediction_scores ps
            JOIN matches m ON ps.match_id = m.match_id
            WHERE ps.model_version = ?
            GROUP BY m.competition
            ORDER BY n DESC
        """, [model]).fetchall()
        by_competition = []
        for r in by_comp_rows:
            by_competition.append({
                "competition": r[0], "n": r[1],
                "accuracy": round(float(r[2]) / r[1], 3) if r[1] else 0,
                "logloss": round(float(r[3]), 4) if r[3] else None,
            })

        # Calibration
        cal_rows = db.execute("""
            SELECT p.p_home, p.p_draw, p.p_away, ps.outcome
            FROM prediction_scores ps
            JOIN predictions p ON ps.match_id = p.match_id AND ps.model_version = p.model_version
            WHERE ps.model_version = ?
        """, [model]).fetchall()
        buckets = {i: {"predicted_sum": 0.0, "actual_sum": 0, "count": 0}
                   for i in range(10)}
        for r in cal_rows:
            for prob, is_actual in [
                (float(r[0]), 1 if r[3] == 0 else 0),
                (float(r[1]), 1 if r[3] == 1 else 0),
                (float(r[2]), 1 if r[3] == 2 else 0),
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
            "recent": recent,
            "by_competition": by_competition,
            "calibration": calibration,
        }
    except Exception as e:
        log.warning("Failed to build performance: %s", e)
        return {
            "model": model,
            "metrics": None,
            "recent": [],
            "by_competition": [],
            "calibration": [],
        }
