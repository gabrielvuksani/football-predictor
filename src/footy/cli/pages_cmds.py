"""Export static JSON snapshots for GitHub Pages deployment.

Usage:
    footy pages export           # Export all data to docs/api/
    footy pages export --out /tmp/docs  # Custom output directory
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer

from footy.cli._shared import console

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
        WHERE m.status = 'TIMED'
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
        "SELECT DISTINCT competition FROM matches WHERE status = 'TIMED'"
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
    console.print(f"  Push to GitHub and enable Pages (source: GitHub Actions) to deploy.")


# ─── Helpers ──────────────────────────────────────────────────────────────

def _export_match_detail(db, api: Path, match_id: int) -> None:
    """Export detail, experts, h2h, form for a single match."""
    row = db.execute("""
        SELECT m.match_id, m.home_team, m.away_team, m.utc_date,
               m.competition, m.status, m.home_goals, m.away_goals,
               p.p_home, p.p_draw, p.p_away, p.eg_home, p.eg_away,
               e.b365h, e.b365d, e.b365a,
               e.hs, e.as_, e.hst, e.ast
        FROM matches m
        LEFT JOIN predictions p
          ON m.match_id = p.match_id AND p.model_version = 'v8_council'
        LEFT JOIN match_extras e ON m.match_id = e.match_id
        WHERE m.match_id = ?
    """, [match_id]).fetchone()
    if not row:
        return

    detail_cols = [
        "match_id", "home_team", "away_team", "utc_date",
        "competition", "status", "home_goals", "away_goals",
        "p_home", "p_draw", "p_away", "eg_home", "eg_away",
        "b365h", "b365d", "b365a", "hs", "as_", "hst", "ast",
    ]
    detail = dict(zip(detail_cols, row))
    detail["prediction"] = {
        "p_home": detail.pop("p_home"),
        "p_draw": detail.pop("p_draw"),
        "p_away": detail.pop("p_away"),
        "eg_home": detail.pop("eg_home"),
        "eg_away": detail.pop("eg_away"),
    }
    detail["odds"] = {
        "b365h": detail.pop("b365h"),
        "b365d": detail.pop("b365d"),
        "b365a": detail.pop("b365a"),
    }
    _dump(api / "matches" / f"{match_id}.json", detail)

    # Experts (from expert_cache if available)
    try:
        cache_row = db.execute(
            "SELECT breakdown FROM expert_cache WHERE match_id = ?",
            [match_id],
        ).fetchone()
        if cache_row:
            experts = json.loads(cache_row[0]) if isinstance(cache_row[0], str) else cache_row[0]
            _dump(api / "matches" / str(match_id) / "experts.json", experts)
    except Exception:
        pass

    # H2H
    home = detail.get("home_team", "")
    away = detail.get("away_team", "")
    try:
        h2h_row = db.execute("""
            SELECT total_matches, home_wins, away_wins, draws,
                   home_goals_avg, away_goals_avg
            FROM h2h_stats
            WHERE team_a = ? AND team_b = ?
               OR team_a = ? AND team_b = ?
            LIMIT 1
        """, [home, away, away, home]).fetchone()
        if h2h_row:
            h2h = dict(zip(
                ["total_matches", "home_wins", "away_wins", "draws",
                 "home_goals_avg", "away_goals_avg"],
                h2h_row
            ))
            # Recent matches
            recent = db.execute("""
                SELECT utc_date, home_team, away_team, home_goals, away_goals
                FROM matches
                WHERE status = 'FINISHED'
                  AND ((home_team = ? AND away_team = ?)
                    OR (home_team = ? AND away_team = ?))
                ORDER BY utc_date DESC LIMIT 10
            """, [home, away, away, home]).fetchall()
            h2h["recent"] = [
                dict(zip(["date", "home", "away", "hg", "ag"], r))
                for r in recent
            ]
            _dump(api / "matches" / str(match_id) / "h2h.json", h2h)
    except Exception:
        pass

    # Form
    try:
        form = {}
        for team, key in [(home, "home"), (away, "away")]:
            last5 = db.execute("""
                SELECT home_team, away_team, home_goals, away_goals
                FROM matches
                WHERE status = 'FINISHED'
                  AND (home_team = ? OR away_team = ?)
                ORDER BY utc_date DESC LIMIT 5
            """, [team, team]).fetchall()
            results = []
            for ht, at, hg, ag in last5:
                if team == ht:
                    res = "W" if hg > ag else "D" if hg == ag else "L"
                else:
                    res = "W" if ag > hg else "D" if hg == ag else "L"
                results.append(res)
            pts = sum(3 if r == "W" else 1 if r == "D" else 0 for r in results)
            form[key] = {
                "team": team,
                "results": results,
                "ppg": round(pts / max(len(results), 1), 2),
            }
        _dump(api / "matches" / str(match_id) / "form.json", form)
    except Exception:
        pass


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
    """Build database statistics summary."""
    try:
        total = db.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        finished = db.execute("SELECT COUNT(*) FROM matches WHERE status = 'FINISHED'").fetchone()[0]
        upcoming = db.execute("SELECT COUNT(*) FROM matches WHERE status = 'TIMED'").fetchone()[0]
        predictions = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        leagues = db.execute("SELECT COUNT(DISTINCT competition) FROM matches").fetchone()[0]
        teams = db.execute("SELECT COUNT(DISTINCT home_team) FROM matches").fetchone()[0]
        return {
            "total_matches": total,
            "finished": finished,
            "upcoming": upcoming,
            "predictions": predictions,
            "leagues": leagues,
            "teams": teams,
        }
    except Exception:
        return {}


def _build_performance(db) -> dict:
    """Build model performance metrics."""
    try:
        rows = db.execute("""
            SELECT model_version,
                   COUNT(*) as n,
                   AVG(logloss) as avg_logloss,
                   AVG(brier) as avg_brier,
                   AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) as accuracy
            FROM prediction_scores
            GROUP BY model_version
            ORDER BY avg_logloss
        """).fetchall()
        models = []
        for mv, n, ll, br, acc in rows:
            models.append({
                "model": mv,
                "n": n,
                "logloss": round(ll, 4) if ll else None,
                "brier": round(br, 4) if br else None,
                "accuracy": round(acc, 4) if acc else None,
            })
        return {"models": models}
    except Exception:
        return {"models": []}
