from __future__ import annotations
import time
import typer
from rich import print
from footy import pipeline
from footy.models.council import train_and_save as council_train, predict_upcoming as council_predict
from footy.fixtures_odds import ingest_upcoming_odds
from footy.extras import ingest_extras_fdcuk

app = typer.Typer(add_completion=False)

@app.command()
def go(skip_history: bool = False):
    """Master command: full pipeline from ingest to predictions.
    
    Runs: history → ingest → extras → odds → train (Elo+Poisson) →
          train-council → predict-council → score → compute-h2h + xG
    
    v7 council is the unified model with 6 experts + meta-learner.
    """
    from footy.db import connect
    t0 = time.perf_counter()

    steps = [
        ("1/8", "Ingest history (fdcuk)"),
        ("2/8", "Ingest fixtures (API)"),
        ("3/8", "Ingest extras + odds"),
        ("4/8", "Train Elo + Poisson"),
        ("5/8", "Train council model"),
        ("6/8", "Predict council"),
        ("7/8", "Score finished predictions"),
        ("8/8", "Compute H2H + xG"),
    ]

    def step(i, msg):
        print(f"[cyan]Step {steps[i][0]}[/cyan] {msg}", flush=True)

    if not skip_history:
        step(0, "Ingesting history from football-data.co.uk...")
        try:
            n_hist = pipeline.ingest_history_fdcuk(n_seasons=25, verbose=True)
            print(f"  History: {n_hist} matches loaded/updated")
        except Exception as e:
            print(f"[yellow]History warning:[/yellow] {e}")
    else:
        print("[yellow]Step 1/8[/yellow] Skipping history (--skip-history)")

    step(1, "Ingesting fixtures from API (last 365 days + 7 ahead)...")
    try:
        pipeline.ingest(days_back=365, verbose=True)
    except Exception as e:
        print(f"[yellow]Ingest warning:[/yellow] {e}")

    step(2, "Ingesting extras + odds...")
    try:
        ingest_extras_fdcuk(verbose=True)
    except Exception as e:
        print(f"[yellow]Extras warning:[/yellow] {e}")
    try:
        ingest_upcoming_odds(verbose=True)
    except Exception as e:
        print(f"[yellow]Odds warning:[/yellow] {e}")

    step(3, "Training Elo + Poisson...")
    n_elo = pipeline.update_elo_from_finished(verbose=True)
    state = pipeline.refit_poisson(verbose=True)
    print(f"  Elo: {n_elo} matches | Poisson: {len(state.get('teams', []))} teams")

    step(4, "Training council model (6 experts + meta-learner)...")
    con = connect()
    rc = council_train(con, eval_days=365, verbose=True)
    print(f"  council: {rc}")

    step(5, "Predicting council...")
    nc = council_predict(con, verbose=True)
    print(f"  council: {nc} predictions")

    step(6, "Scoring finished predictions...")
    try:
        sc = pipeline.score_finished_predictions(verbose=True)
        print(f"  scored: {sc}")
    except Exception as e:
        print(f"[yellow]Scoring warning:[/yellow] {e}")

    step(7, "Computing H2H + xG...")
    try:
        from footy.h2h import recompute_h2h_stats
        recompute_h2h_stats(con, verbose=True)
    except Exception as e:
        print(f"[yellow]H2H warning:[/yellow] {e}")
    try:
        from footy.xg import backfill_xg_for_finished_matches
        backfill_xg_for_finished_matches(con, verbose=True)
    except Exception as e:
        print(f"[yellow]xG warning:[/yellow] {e}")

    dt = time.perf_counter() - t0
    print(f"\n[green bold]Pipeline complete in {dt:.0f}s[/green bold]")


@app.command()
def refresh():
    """Quick daily update: ingest recent → extras → odds → retrain council → predict → H2H.

    Faster than `go` — skips history download. Use this for daily cron jobs.
    """
    from footy.db import connect
    t0 = time.perf_counter()

    print("[cyan]Step 1/7[/cyan] Ingest recent fixtures...", flush=True)
    try:
        pipeline.ingest(days_back=30, verbose=True)
    except Exception as e:
        print(f"[yellow]Ingest warning:[/yellow] {e}")

    print("[cyan]Step 2/7[/cyan] Extras + odds...", flush=True)
    try:
        ingest_extras_fdcuk(verbose=True)
    except Exception as e:
        print(f"[yellow]Extras warning:[/yellow] {e}")
    try:
        ingest_upcoming_odds(verbose=True)
    except Exception as e:
        print(f"[yellow]Odds warning:[/yellow] {e}")

    print("[cyan]Step 3/7[/cyan] Train Elo + Poisson...", flush=True)
    n_elo = pipeline.update_elo_from_finished(verbose=True)
    pipeline.refit_poisson(verbose=True)
    print(f"  Elo: {n_elo} matches")

    print("[cyan]Step 4/7[/cyan] Retrain council...", flush=True)
    con = connect()
    rc = council_train(con, eval_days=365, verbose=True)
    print(f"  council: {rc}")

    print("[cyan]Step 5/7[/cyan] Predict council...", flush=True)
    nc = council_predict(con, lookahead_days=14, verbose=True)
    print(f"  council: {nc} predictions")

    print("[cyan]Step 6/7[/cyan] Score finished predictions...", flush=True)
    try:
        sc = pipeline.score_finished_predictions(verbose=True)
        print(f"  scored: {sc}")
    except Exception as e:
        print(f"[yellow]Scoring warning:[/yellow] {e}")

    print("[cyan]Step 7/7[/cyan] Compute H2H...", flush=True)
    try:
        from footy.h2h import recompute_h2h_stats
        recompute_h2h_stats(con, verbose=True)
    except Exception as e:
        print(f"[yellow]H2H warning:[/yellow] {e}")

    dt = time.perf_counter() - t0
    print(f"\n[green bold]Refresh complete in {dt:.0f}s[/green bold]")


@app.command()
def matchday():
    """Weekend preview: refresh + AI preview for all leagues.

    Runs a full refresh then generates AI previews for every league.
    Perfect to run Friday evening or Saturday morning.
    """
    from footy.db import connect
    t0 = time.perf_counter()

    # run refresh first
    print("[bold cyan]== REFRESH ==[/bold cyan]", flush=True)
    refresh()

    # AI preview for each league
    print("\n[bold cyan]== MATCHDAY PREVIEW ==[/bold cyan]", flush=True)
    try:
        from footy.llm.insights import league_round_summary
        from rich.panel import Panel
        from rich.console import Console
        console = Console()

        for code in ["PL", "PD", "SA", "BL1", "FL1"]:
            try:
                summary = league_round_summary(code)
                if summary.get("matches", 0) > 0:
                    console.print(Panel(
                        f"[bold]{summary.get('competition', code)} Round Summary[/bold] "
                        f"({summary.get('matches', 0)} matches)\n\n"
                        f"{summary.get('summary', '')}\n\n"
                        f"[cyan]Headline pick:[/cyan] {summary.get('headline_pick', 'N/A')}\n\n"
                        + "\n".join(
                            f"  {p['home']} vs {p['away']} — {p['pred']} ({p['probs']})"
                            for p in summary.get('predictions', [])
                        ),
                        title=f"[bold]{code}[/bold]",
                    ))
            except Exception as e:
                print(f"[yellow]{code} preview skipped:[/yellow] {e}")
    except ImportError:
        print("[yellow]AI preview unavailable (Ollama not configured)[/yellow]")

    dt = time.perf_counter() - t0
    print(f"\n[green bold]Matchday prep complete in {dt:.0f}s[/green bold]")


@app.command()
def nuke():
    """Nuclear option: delete the entire database and rebuild from scratch.

    Removes the DB file completely, recreates fresh schema,
    then runs full `go` pipeline.
    Use when something is fundamentally broken.
    """
    import os
    from footy.config import settings as get_settings
    t0 = time.perf_counter()
    print("[bold red]== NUKE: Full rebuild ==[/bold red]", flush=True)

    # Delete the DB file entirely for a truly clean slate
    s = get_settings()
    db_path = s.db_path
    for f in [db_path, db_path + ".wal"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"[red]Deleted {f}[/red]", flush=True)

    print("[red]Resetting all model states...[/red]", flush=True)
    pipeline.reset_states(verbose=True)

    print("[cyan]Running full pipeline from scratch...[/cyan]", flush=True)
    go(skip_history=False)

    dt = time.perf_counter() - t0
    print(f"\n[green bold]Nuke complete in {dt:.0f}s[/green bold]")


# ============================================================================
# AI-powered Commands (Ollama)
# ============================================================================

@app.command()
def ai_preview(match_id: int = None, league: str = None):
    """AI-generated pre-match preview.
    
    Provide --match-id for a single match, or --league (PL/PD/SA/BL1) for a round summary.
    """
    from footy.llm.insights import preview_match, league_round_summary
    from rich.panel import Panel
    from rich.console import Console
    console = Console()

    if match_id:
        pv = preview_match(match_id)
        if pv.get("error"):
            print(f"[red]{pv['error']}[/red]")
            return
        console.print(Panel(
            f"[bold]{pv.get('home_team','?')} vs {pv.get('away_team','?')}[/bold]\n\n"
            f"{pv.get('preview','')}\n\n"
            f"[cyan]Prediction:[/cyan] {pv.get('prediction','')} ({pv.get('confidence','')}) confidence\n"
            + ("\n".join(f"  • {s}" for s in pv.get('key_stats', [])) or "")
            + ("\n\n[yellow]Value bets:[/yellow]\n" + "\n".join(f"  → {v}" for v in pv.get('value_bets', [])) if pv.get('value_bets') else ""),
            title="Match Preview",
        ))
    elif league:
        summary = league_round_summary(league.upper())
        console.print(Panel(
            f"[bold]{summary.get('competition','')} Round Summary[/bold] ({summary.get('matches',0)} matches)\n\n"
            f"{summary.get('summary','')}\n\n"
            f"[cyan]Headline pick:[/cyan] {summary.get('headline_pick','')}\n\n"
            + "\n".join(
                f"  {p['home']} vs {p['away']} — {p['pred']} ({p['probs']})"
                for p in summary.get('predictions', [])
            ),
            title="Round Preview",
        ))
    else:
        # Default: show all leagues
        for code in ["PL", "PD", "SA", "BL1", "FL1"]:
            summary = league_round_summary(code)
            if summary.get("matches", 0) > 0:
                print(f"\n[bold cyan]{code}[/bold cyan] — {summary.get('summary','')}")
                print(f"  Headline: {summary.get('headline_pick','')}")

@app.command()
def ai_value(min_edge: float = 0.05):
    """Scan upcoming matches for value bets where model edge exceeds market odds."""
    from footy.llm.insights import value_bet_scan, ai_value_commentary
    from rich.table import Table

    values = value_bet_scan(min_edge=min_edge)

    if not values:
        print(f"[yellow]No value bets found with edge ≥ {min_edge:.0%}[/yellow]")
        return

    table = Table(title=f"Value Bets (edge ≥ {min_edge:.0%})")
    table.add_column("Match", style="cyan")
    table.add_column("Bet", style="white")
    table.add_column("Odds", style="yellow")
    table.add_column("Model", style="green")
    table.add_column("Implied", style="red")
    table.add_column("Edge", style="bold green")

    for v in values[:15]:
        table.add_row(
            f"{v['home_team']} v {v['away_team']}",
            v["bet"], f"{v['odds']:.2f}",
            f"{v['model_prob']:.0%}", f"{v['implied_prob']:.0%}",
            f"{v['edge']:+.1%}",
        )
    print(table)

    commentary = ai_value_commentary(values)
    print(f"\n[italic]{commentary}[/italic]")

@app.command()
def ai_review(days: int = 3, league: str = None):
    """AI review of recent prediction accuracy."""
    from footy.llm.insights import post_match_review
    from rich.panel import Panel
    from rich.console import Console
    console = Console()

    review = post_match_review(days_back=days, competition_code=league)

    if review["matches_reviewed"] == 0:
        print("[yellow]No recent finished matches with predictions to review[/yellow]")
        return

    acc_color = "green" if review["accuracy"] >= 0.50 else ("yellow" if review["accuracy"] >= 0.40 else "red")
    console.print(Panel(
        f"[bold]Post-Match Review[/bold] (last {days} days"
        + (f", {league}" if league else "") + ")\n\n"
        f"[{acc_color}]Accuracy: {review['accuracy']:.0%}[/{acc_color}] "
        f"({review['correct']}/{review['matches_reviewed']})\n\n"
        f"{review['review']}\n\n"
        + ("[red]Notable misses:[/red]\n" + "\n".join(
            f"  ✗ {m['match']} {m['score']} — predicted {m['predicted']} "
            f"(was {m['actual']}, conf {m['confidence']:.0%})"
            for m in review.get("misses", [])
        ) if review.get("misses") else "[green]No major misses![/green]"),
        title="AI Review",
    ))

@app.command()
def serve(port: int = 8000, host: str = "0.0.0.0", max_retries: int = 10):
    """Start the web UI server (FastAPI + modern frontend).

    Kills any previous footy-serve / uvicorn processes first, then
    finds a free port starting from --port (tries up to --max-retries).
    """
    import os
    import signal
    import socket
    import subprocess
    import sys
    from pathlib import Path

    # Ensure project root is on sys.path so uvicorn can resolve "web.api"
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import uvicorn

    # --- kill any stale footy-serve / uvicorn web.api processes -----------
    my_pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "uvicorn.*web\\.api"], text=True
        ).strip()
        for pid_str in out.splitlines():
            pid = int(pid_str)
            if pid != my_pid:
                print(f"[yellow]Killing stale server process {pid}…[/yellow]")
                os.kill(pid, signal.SIGTERM)
    except (subprocess.CalledProcessError, ValueError):
        pass  # no matching processes

    # --- find a free port ------------------------------------------------
    def _port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) == 0

    chosen = port
    for attempt in range(max_retries):
        if not _port_in_use(chosen):
            break
        print(f"[yellow]Port {chosen} is in use, trying {chosen + 1}…[/yellow]")
        chosen += 1
    else:
        print(f"[red]Could not find a free port in range {port}–{chosen}. Aborting.[/red]")
        raise typer.Exit(code=1)

    if chosen != port:
        print(f"[green]Found free port {chosen}[/green]")
    print(f"[green]Starting Footy Predictor on http://{host}:{chosen}[/green]")
    print(f"[cyan]Access from other devices at http://<this-machine-ip>:{chosen}[/cyan]")
    uvicorn.run("web.api:app", host=host, port=chosen, log_level="info")

@app.command()
def ingest(days_back: int = 30, days_forward: int = 7, chunk_days: int = 10):
    t0 = time.perf_counter()
    n = pipeline.ingest(days_back=days_back, days_forward=days_forward, chunk_days=chunk_days, verbose=True)
    dt = time.perf_counter() - t0
    print(f"[green]Inserted/updated[/green] {n} matches in {dt:.1f}s")

@app.command()
def train():
    t0 = time.perf_counter()
    n = pipeline.update_elo_from_finished(verbose=True)
    print(f"[green]Elo applied[/green] {n} new finished matches")
    state = pipeline.refit_poisson(verbose=True)
    dt = time.perf_counter() - t0
    print(f"[green]Train done[/green] teams={len(state.get('teams', []))} in {dt:.1f}s")

@app.command()
def predict():
    t0 = time.perf_counter()
    n = pipeline.predict_upcoming(verbose=True)
    dt = time.perf_counter() - t0
    print(f"[green]Predictions written[/green] {n} upcoming matches in {dt:.1f}s")

@app.command()
def metrics():
    m = pipeline.backtest_metrics()
    print(m)

@app.command()
def update():
    t0 = time.perf_counter()
    print("[cyan]Step 1/4[/cyan] ingest", flush=True)
    pipeline.ingest(verbose=True)

    print("[cyan]Step 2/4[/cyan] train (Elo + Poisson)", flush=True)
    new_elo = pipeline.update_elo_from_finished(verbose=True)
    pipeline.refit_poisson(verbose=True)

    print("[cyan]Step 3/4[/cyan] predict upcoming", flush=True)
    n_pred = pipeline.predict_upcoming(verbose=True)

    print("[cyan]Step 4/4[/cyan] metrics (only counts matches that were predicted before finishing)", flush=True)
    m = pipeline.backtest_metrics()
    dt = time.perf_counter() - t0
    print(f"[green]update complete[/green] new_elo={new_elo} predicted={n_pred} in {dt:.1f}s")
    print(f"[cyan]metrics[/cyan] {m}")


@app.command()
def ingest_history(n_seasons: int = 8):
    n = pipeline.ingest_history_fdcuk(n_seasons=n_seasons, verbose=True)
    print(f"[green]History ingested[/green] rows={n}")

@app.command()
def reset_states():
    pipeline.reset_states(verbose=True)

@app.command()
def news(days_back: int = 2, max_records: int = 10):
    n = pipeline.ingest_news_for_teams(days_back=days_back, max_records=max_records)
    print(f"[green]Inserted[/green] {n} news rows")

@app.command()
def backtest(days: int = 180, test_days: int = 14):
    r = pipeline.backtest_time_split(days=days, test_days=test_days, verbose=True)
    print(r)

@app.command()
def train_meta(days: int = 365, test_days: int = 28):
    r = pipeline.train_meta_model(days=days, test_days=test_days, verbose=True)
    print(r)

@app.command()
def ingest_extras(n_seasons: int = 8):
    n = ingest_extras_fdcuk(n_seasons=n_seasons, verbose=True)
    print(f"[green]Extras upserted[/green] rows={n}")


@app.command()
def ingest_fixtures_odds():
    n = ingest_upcoming_odds(verbose=True)
    print({"attached_odds": n})

@app.command()
def compute_h2h():
    """Recompute all head-to-head statistics from finished matches."""
    from footy.db import connect
    from footy.h2h import recompute_h2h_stats
    con = connect()
    r = recompute_h2h_stats(con, verbose=True)
    print(r)

@app.command()
def compute_xg():
    """Compute xG for all finished matches from available statistics."""
    from footy.db import connect
    from footy.xg import backfill_xg_for_finished_matches
    con = connect()
    r = backfill_xg_for_finished_matches(con, verbose=True)
    print(r)

@app.command()
def update_odds():
    """Update odds for upcoming matches using external sources or predictions."""
    from footy.db import connect
    from footy.providers.odds_scraper import update_upcoming_match_odds, fill_upcoming_odds_from_predictions
    con = connect()
    
    print("[cyan]Step 1/2[/cyan] External sources (ODD-API, etc.)", flush=True)
    r1 = update_upcoming_match_odds(con, verbose=True)
    print(r1, flush=True)
    
    print("[cyan]Step 2/2[/cyan] Fallback to model predictions", flush=True)
    r2 = fill_upcoming_odds_from_predictions(con, verbose=True)
    print(r2, flush=True)

@app.command()
def ingest_af(lookahead_days: int = 7, stale_hours: int = 6):
    """Fetch API-Football context (fixtures/lineups/stats + injuries) for upcoming matches; cached in DuckDB."""
    from footy.db import connect
    from footy.providers.api_football import map_upcoming_matches, upsert_context

    con = connect()
    print(f"Step 1/2 map fixtures (lookahead_days={lookahead_days})", flush=True)
    nmap = map_upcoming_matches(con, lookahead_days=lookahead_days, verbose=True)

    print(f"Step 2/2 ingest context (stale_hours={stale_hours})", flush=True)
    nctx = upsert_context(con, stale_hours=stale_hours, verbose=True)

    print({"mapped": nmap, "context_rows_written": nctx}, flush=True)

@app.command()
def cache_stats():
    """Show cache statistics and usage."""
    from footy.cache import get_cache
    
    cache = get_cache()
    stats = cache.get_stats()
    
    print("\n[cyan]Cache Statistics[/cyan]")
    print(f"  Predictions: {stats['total_predictions']} total, {stats['expired_predictions']} expired")
    print(f"  Metadata: {stats['total_metadata']} total, {stats['expired_metadata']} expired")
    print(f"  By category:")
    for cat, count in stats["metadata_by_category"].items():
        print(f"    {cat:15s}: {count:5d}")

@app.command()
def cache_cleanup(full: bool = False):
    """Clean up cache (remove expired entries)."""
    from footy.cache import get_cache
    
    if full:
        print("[yellow]Clearing entire cache...[/yellow]", flush=True)
        get_cache().clear()
        print("[green]Cache cleared[/green]")
    else:
        print("[cyan]Removing expired entries...[/cyan]", flush=True)
        result = get_cache().cleanup(delete_expired=True)
        print(f"[green]Cleanup complete:[/green]")
        print(f"  Deleted predictions: {result['deleted_predictions']}")
        print(f"  Deleted metadata: {result['deleted_metadata']}")


# ============================================================================
# Phase 3: Ollama AI Integration
# ============================================================================

@app.command()
def extract_news(team: str = None, days_back: int = 2):
    """Extract team news from GDELT and analyze with Ollama."""
    from footy.llm.insights import extract_team_news_signal
    
    if not team:
        print("[red]Error: --team is required[/red]")
        return
    
    print(f"[cyan]Extracting news for {team}...[/cyan]", flush=True)
    signal = extract_team_news_signal(team, days_back=days_back)
    
    print(f"\n[bold]{team}[/bold]")
    print(f"  Availability: {signal['availability_score']:+.1f}")
    print(f"  Headlines: {signal['headline_count']}")
    print(f"  Summary: {signal.get('summary', 'N/A')}")
    if signal.get('key_notes'):
        print(f"  Key Notes:")
        for note in signal['key_notes']:
            print(f"    - {note}")
    if signal.get('likely_absences'):
        print(f"  Likely Absences:")
        for absence in signal['likely_absences']:
            print(f"    - {absence}")


@app.command()
def analyze_form(team: str = None, matches: int = 10):
    """Analyze recent team form using Ollama."""
    from footy.llm.insights import analyze_team_form
    
    if not team:
        print("[red]Error: --team is required[/red]")
        return
    
    print(f"[cyan]Analyzing form for {team}...[/cyan]", flush=True)
    form = analyze_team_form(team, matches_window=matches)
    
    print(f"\n[bold]{team}[/bold]")
    print(f"  Recent Form: {form['recent_form']}")
    if form.get('record'):
        print(f"  Record: {form['record']}")
    print(f"  Momentum: {form['momentum']:+.1f}")
    if form.get('key_trends'):
        print(f"  Key Trends:")
        for trend in form['key_trends']:
            print(f"    - {trend}")
    if form.get('concern_areas'):
        print(f"  Concern Areas:")
        for area in form['concern_areas']:
            print(f"    - {area}")


@app.command()
def explain_match(
    match_id: int = None, 
    home_team: str = None, 
    away_team: str = None,
    home_prob: float = 0.5,
    draw_prob: float = 0.25,
    away_prob: float = 0.25
):
    """Generate LLM explanation for a match prediction."""
    from footy.llm.insights import explain_match
    
    if not (home_team and away_team):
        print("[red]Error: --home-team and --away-team are required[/red]")
        return
    
    match_id = match_id or 0
    
    print(f"[cyan]Generating explanation for {home_team} vs {away_team}...[/cyan]", flush=True)
    explanation = explain_match(
        match_id=match_id,
        home_team=home_team,
        away_team=away_team,
        home_pred=home_prob,
        draw_pred=draw_prob,
        away_pred=away_prob
    )
    
    print(f"\n[bold]{home_team} ({home_prob:.1%})[/bold]" 
          f" vs [bold]{away_team} ({away_prob:.1%})[/bold] "
          f"(Draw: {draw_prob:.1%})")
    print(f"\n[cyan]Explanation:[/cyan]")
    print(f"  {explanation.get('explanation', 'N/A')}")
    
    if explanation.get('key_factors'):
        print(f"\n[cyan]Key Factors:[/cyan]")
        for factor in explanation['key_factors']:
            print(f"  - {factor}")
    
    print(f"\n[cyan]Confidence:[/cyan] {explanation.get('confidence_level', 'N/A')}")


@app.command()
def insights_status():
    """Check status of Ollama AI insights system."""
    from footy.llm.insights import get_insights_status
    
    status = get_insights_status()
    
    print("\n[cyan]Insights System Status[/cyan]")
    print(f"  Status: {status['status']}")
    print(f"  LLM Health: {status.get('llm_health', 'Unknown')}")
    print(f"  Cache - Predictions: {status.get('cache_predictions', 0)}")
    print(f"  Cache - Metadata: {status.get('cache_metadata', 0)}")
    print(f"  Last Check: {status.get('timestamp', 'N/A')}")


# ============================================================================
# Phase 4: Training Scheduler
# ============================================================================

@app.command()
def scheduler_add(
    job_id: str,
    job_type: str,
    cron: str,
    days_back: int = 30,
    days_forward: int = 7,
):
    """Add a new scheduled job.
    
    Job types: ingest, train_base, train_council, predict, score
    
    Cron examples:
      "0 2 * * *"     - Daily at 2 AM
      "0 */6 * * *"   - Every 6 hours
      "0 3 * * 0"     - Weekly on Sunday at 3 AM
    """
    from footy.scheduler import get_scheduler
    
    try:
        scheduler = get_scheduler()
        params = {}
        
        if job_type == "ingest":
            params = {"days_back": days_back, "days_forward": days_forward}
        
        result = scheduler.add_job(job_id, job_type, cron, params=params)
        print(f"[green]Job created:[/green] {job_id}")
        print(f"  Type: {result['job_type']}")
        print(f"  Schedule: {result['cron_schedule']}")
        print(f"  Enabled: {result['enabled']}")
    except Exception as e:
        print(f"[red]Error:[/red] {e}")


@app.command()
def scheduler_start():
    """Start the background scheduler."""
    from footy.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    scheduler.start()
    print("[green]Scheduler started[/green]")
    
    jobs = scheduler.list_jobs()
    print(f"[cyan]Active jobs: {len(jobs)}[/cyan]")
    for job in jobs:
        if job["enabled"]:
            status = f"[{job['last_status'] or 'pending'}]"
            print(f"  - {job['job_id']:30s} {status:15s} (next: {job['next_run_at']})")


@app.command()
def scheduler_stop():
    """Stop the background scheduler."""
    from footy.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    scheduler.stop()
    print("[green]Scheduler stopped[/green]")


@app.command()
def scheduler_list():
    """List all scheduled jobs."""
    from footy.scheduler import get_scheduler
    from rich.table import Table
    
    scheduler = get_scheduler()
    jobs = scheduler.list_jobs()
    
    if not jobs:
        print("[yellow]No scheduled jobs[/yellow]")
        return
    
    table = Table(title="Scheduled Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Schedule", style="yellow")
    table.add_column("Enabled", style="green")
    table.add_column("Last Status", style="blue")
    table.add_column("Last Run", style="white")
    table.add_column("Next Run", style="white")
    
    for job in jobs:
        last_run_str = str(job["last_run_at"]).split(".")[0] if job["last_run_at"] else "Never"
        next_run_str = str(job["next_run_at"]).split(".")[0] if job["next_run_at"] else "N/A"
        status_color = "green" if job["last_status"] == "SUCCESS" else ("red" if job["last_status"] == "FAILED" else "yellow")
        
        table.add_row(
            job["job_id"],
            job["job_type"],
            job["cron_schedule"],
            "✓" if job["enabled"] else "✗",
            f"[{status_color}]{job['last_status'] or '-'}[/{status_color}]",
            last_run_str,
            next_run_str,
        )
    
    print(table)


@app.command()
def scheduler_enable(job_id: str):
    """Enable a scheduled job."""
    from footy.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    try:
        result = scheduler.enable_job(job_id)
        print(f"[green]Job enabled:[/green] {result['enabled']}")
    except Exception as e:
        print(f"[red]Error:[/red] {e}")


@app.command()
def scheduler_disable(job_id: str):
    """Disable a scheduled job."""
    from footy.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    result = scheduler.disable_job(job_id)
    print(f"[green]Job disabled:[/green] {result['disabled']}")


@app.command()
def scheduler_remove(job_id: str, confirm: bool = False):
    """Remove a scheduled job."""
    from footy.scheduler import get_scheduler
    
    if not confirm:
        print(f"[yellow]Warning:[/yellow] This will delete job '{job_id}'")
        print("         Use --confirm to proceed")
        return
    
    scheduler = get_scheduler()
    result = scheduler.remove_job(job_id)
    print(f"[green]Job removed:[/green] {result['removed']}")


@app.command()
def scheduler_history(job_id: str, limit: int = 10):
    """Show execution history for a job."""
    from footy.scheduler import get_scheduler
    from rich.table import Table
    
    scheduler = get_scheduler()
    history = scheduler.get_job_history(job_id, limit=limit)
    
    if not history:
        print(f"[yellow]No history for job:[/yellow] {job_id}")
        return
    
    table = Table(title=f"Job History: {job_id}")
    table.add_column("Started At", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Duration (s)", style="yellow")
    table.add_column("Error / Result", style="white")
    
    for run in history:
        error_or_result = run["error"] or (str(run["result"])[:60] if run["result"] else "-")
        status_color = "green" if run["status"] == "SUCCESS" else ("red" if run["status"] == "FAILED" else "yellow")
        
        table.add_row(
            str(run["started_at"]).split(".")[0],
            f"[{status_color}]{run['status']}[/{status_color}]",
            f"{run['duration_seconds']:.1f}" if run['duration_seconds'] else "-",
            error_or_result,
        )
    
    print(table)


@app.command()
def scheduler_stats():
    """Show scheduler statistics."""
    from footy.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    stats = scheduler.get_stats()
    
    print("\n[cyan]Scheduler Statistics[/cyan]")
    print(f"  Running: {stats['scheduler_running']}")
    print(f"  Total Jobs: {stats['total_jobs']}")
    print(f"  Active Jobs: {len(stats['active_jobs'])}")
    
    print(f"\n[cyan]Stats by Job Type:[/cyan]")
    for job_type, type_stats in stats["stats_by_type"].items():
        print(f"  {job_type}:")
        print(f"    Total Runs: {type_stats['total_runs']}")
        print(f"    Successful: {type_stats['successful']}")
        print(f"    Failed: {type_stats['failed']}")
        print(f"    Avg Duration: {type_stats['avg_duration_seconds']:.1f}s")


# ============================================================================
# Phase 4.2: Continuous Model Retraining
# ============================================================================

@app.command()
def retrain(force: bool = False):
    """Auto-retrain council model: check thresholds → train → validate → deploy/rollback.
    
    Checks if retraining is needed (new match threshold or drift detected),
    trains a new council model, compares performance vs the current model, and
    deploys only if performance improves. Automatically rolls back on regression.
    
    Use --force to retrain regardless of thresholds.
    """
    from footy.continuous_training import get_training_manager

    manager = get_training_manager()
    # Ensure v7_council schedule exists
    try:
        manager.setup_continuous_training("v7_council", 20, 0.005)
    except Exception:
        pass

    result = manager.auto_retrain(force=force, verbose=True)
    action = result.get("action", "unknown")

    if action == "none":
        check = result.get("check", {})
        print(f"[yellow]No retraining needed[/yellow]")
        if check.get("status") == "waiting":
            print(f"  New matches: {check.get('new_matches', '?')}/{check.get('threshold', '?')}")
            print(f"  Ready in ~{check.get('ready_in', '?')} more finished matches")
    elif action == "deployed":
        print(f"[green bold]Model deployed:[/green bold] {result['version']}")
        m = result.get("metrics", {})
        print(f"  Accuracy: {m.get('accuracy', 0):.1%}  LogLoss: {m.get('logloss', 0):.4f}")
        print(f"  Improvement: {result.get('improvement_pct', 0):+.2f}%")
    elif action == "rolled_back":
        print(f"[red]Rolled back[/red] — new model was worse")
        m = result.get("metrics", {})
        print(f"  New accuracy: {m.get('accuracy', 0):.1%}  LogLoss: {m.get('logloss', 0):.4f}")
        print(f"  Regression: {result.get('improvement_pct', 0):+.2f}%")
    elif action == "train_failed":
        print(f"[red]Training failed:[/red] {result.get('error', 'unknown')}")
    else:
        print(f"[yellow]Result:[/yellow] {result}")

@app.command()
def drift_check():
    """Check for prediction accuracy drift on recent matches."""
    from footy.continuous_training import get_training_manager

    manager = get_training_manager()
    drift = manager.detect_drift("v7_council")

    if drift.get("reason") == "insufficient_data":
        print(f"[yellow]Not enough data for drift detection[/yellow]")
        print(f"  Baseline matches: {drift.get('baseline_n', 0)}")
        print(f"  Recent matches: {drift.get('recent_n', 0)}")
        return

    if drift.get("drifted"):
        print(f"[red bold]DRIFT DETECTED[/red bold]")
    else:
        print(f"[green]No drift detected[/green]")

    print(f"  Baseline accuracy (>60d ago): {drift.get('baseline_acc', 0):.1%} ({drift.get('baseline_n', 0)} matches)")
    print(f"  Recent accuracy  (<60d):      {drift.get('recent_acc', 0):.1%} ({drift.get('recent_n', 0)} matches)")
    print(f"  Accuracy change: {-drift.get('accuracy_drop', 0):+.1%}")

@app.command()
def retraining_setup(
    model_type: str,
    threshold_matches: int = 10,
    threshold_improvement: float = 0.01,
):
    """Setup continuous retraining for a model type.
    
    Model types: v7_council (current), v5_ultimate (legacy)
    """
    from footy.continuous_training import get_training_manager
    
    manager = get_training_manager()
    result = manager.setup_continuous_training(model_type, threshold_matches, threshold_improvement)
    
    print(f"[green]Retraining configured for {result['model_type']}[/green]")
    print(f"  Threshold: {result['retrain_threshold_matches']} new matches")
    print(f"  Min Improvement: {result['performance_threshold_improvement']:.1%}")


@app.command()
def retraining_status():
    """Show retraining status for all models."""
    from footy.continuous_training import get_training_manager
    from rich.table import Table
    
    manager = get_training_manager()
    status = manager.get_retraining_status()
    
    if not status:
        print("[yellow]No models configured for retraining[/yellow]")
        return
    
    table = Table(title="Model Retraining Status")
    table.add_column("Model Type", style="cyan")
    table.add_column("New Matches", style="magenta")
    table.add_column("Threshold", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Last Trained", style="white")
    
    for model_type, stats in status.items():
        status_color = "green" if stats["ready"] else "yellow"
        status_str = f"[{status_color}]{'Ready' if stats['ready'] else 'Waiting'}[/{status_color}]"
        
        last_trained = str(stats["last_trained"]).split(".")[0] if stats["last_trained"] else "Never"
        
        table.add_row(
            model_type,
            str(stats["new_matches"]),
            str(stats["threshold"]),
            status_str,
            last_trained,
        )
    
    print(table)


@app.command()
def retraining_record(
    model_version: str,
    model_type: str,
    window_days: int = 3650,
    n_matches_train: int = 100,
    n_matches_test: int = 20,
    accuracy: float = 0.65,
    logloss: float = 0.42,
):
    """Record completion of a manual training run."""
    from footy.continuous_training import get_training_manager
    
    manager = get_training_manager()
    
    metrics = {"accuracy": accuracy, "logloss": logloss}
    result = manager.record_training(
        model_version=model_version,
        model_type=model_type,
        training_window_days=window_days,
        n_matches_used=n_matches_train,
        n_matches_test=n_matches_test,
        metrics=metrics,
        test_metrics=metrics,
    )
    
    print(f"[green]Training recorded for {result['model_version']}[/green]")
    print(f"  Model Type: {result['model_type']}")
    print(f"  Improvement: {result['improvement_pct']:+.2f}%")
    if result['previous_version']:
        print(f"  Previous: {result['previous_version']}")


@app.command()
def retraining_deploy(model_version: str, model_type: str, force: bool = False):
    """Deploy a trained model version to production."""
    from footy.continuous_training import get_training_manager
    
    manager = get_training_manager()
    
    if not force:
        print(f"[yellow]⚠️  Deploying {model_version} to production[/yellow]")
        print("    Use --force to confirm")
        return
    
    result = manager.deploy_model(model_version, model_type, force=force)
    
    if result.get("status") == "error":
        print(f"[red]Error:[/red] {result['error']}")
        return
    
    print(f"[green]Model deployed:[/green] {result['model_version']}")
    print(f"  Type: {result['model_type']}")
    print(f"  Improvement: {result['improvement_pct']:+.2f}%")
    if result['previous_version']:
        print(f"  Previous: {result['previous_version']}")


@app.command()
def retraining_rollback(model_type: str, force: bool = False):
    """Rollback to previous model version."""
    from footy.continuous_training import get_training_manager
    
    manager = get_training_manager()
    
    if not force:
        print(f"[yellow]⚠️  Rolling back {model_type}[/yellow]")
        print("    Use --force to confirm")
        return
    
    result = manager.rollback_model(model_type)
    
    if result.get("status") == "error":
        print(f"[red]Error:[/red] {result['error']}")
        return
    
    print(f"[green]Model rolled back:[/green] {result['model_type']}")
    print(f"  Restored: {result['restored_version']}")


@app.command()
def retraining_history(model_type: str, limit: int = 10):
    """Show training history for a model type."""
    from footy.continuous_training import get_training_manager
    from rich.table import Table
    
    manager = get_training_manager()
    history = manager.get_training_history(model_type, limit=limit)
    
    if not history:
        print(f"[yellow]No training history for {model_type}[/yellow]")
        return
    
    table = Table(title=f"Training History: {model_type}")
    table.add_column("Version", style="cyan")
    table.add_column("Training Date", style="magenta")
    table.add_column("Matches", style="yellow")
    table.add_column("Improvement", style="green")
    table.add_column("Deployed", style="white")
    
    for run in history:
        deployed_str = "✓" if run["deployed"] else "✗"
        improvement_color = "green" if run["improvement_pct"] >= 0 else "red"
        
        table.add_row(
            run["model_version"],
            str(run["training_date"]).split(".")[0],
            str(run["n_matches_used"]),
            f"[{improvement_color}]{run['improvement_pct']:+.2f}%[/{improvement_color}]",
            deployed_str,
        )
    
    print(table)


@app.command()
def retraining_deployments():
    """Show current model deployments."""
    from footy.continuous_training import get_training_manager
    from rich.table import Table
    
    manager = get_training_manager()
    deployments = manager.get_deployment_status()
    
    if not deployments:
        print("[yellow]No models deployed[/yellow]")
        return
    
    table = Table(title="Model Deployments")
    table.add_column("Model Type", style="cyan")
    table.add_column("Active Version", style="magenta")
    table.add_column("Deployed At", style="yellow")
    table.add_column("Previous", style="white")
    
    for model_type, info in deployments.items():
        table.add_row(
            model_type,
            info["active_version"],
            str(info["deployed_at"]).split(".")[0],
            info["previous_version"] or "—",
        )
    
    print(table)


# ============================================================================
# Phase 4.3: Performance Tracking System
# ============================================================================

@app.command()
def performance_summary():
    """Show performance summary for all models."""
    from footy.performance_tracker import get_performance_tracker
    from rich.table import Table
    
    tracker = get_performance_tracker()
    summary = tracker.get_summary()
    
    if not summary:
        print("[yellow]No performance data available[/yellow]")
        return
    
    table = Table(title="Model Performance Summary (180 days)")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", style="magenta")
    table.add_column("Logloss", style="yellow")
    table.add_column("Brier", style="green")
    table.add_column("Predictions", style="white")
    
    for model, metrics in sorted(summary.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        table.add_row(
            model,
            f"{metrics['accuracy']:.3f}",
            f"{metrics['logloss']:.3f}",
            f"{metrics['brier']:.3f}",
            str(metrics['n_predictions']),
        )
    
    print(table)


@app.command()
def improvement_report(model: str = "v7_council", days: int = 180):
    """Deep error analysis with actionable recommendations for model improvement."""
    from footy.performance_tracker import generate_improvement_report
    report = generate_improvement_report(model, days)
    print(report)


@app.command()
def error_analysis(model: str = "v7_council", days: int = 180):
    """Show prediction error patterns: confusion matrix, calibration, market accuracy."""
    from footy.performance_tracker import analyze_prediction_errors
    import json
    analysis = analyze_prediction_errors(model, days)
    if analysis["status"] == "no_data":
        print("[yellow]No scored predictions to analyze.[/yellow]")
        return
    print(json.dumps(analysis, indent=2, default=str))


@app.command()
def performance_ranking(days: int = 365):
    """Rank all models by accuracy."""
    from footy.performance_tracker import get_performance_tracker
    from rich.table import Table
    
    tracker = get_performance_tracker()
    rankings = tracker.get_model_rankings(days=days)
    
    if not rankings:
        print("[yellow]No performance data available[/yellow]")
        return
    
    table = Table(title=f"Model Rankings ({days} days)")
    table.add_column("Rank", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Accuracy", style="yellow")
    table.add_column("Logloss", style="green")
    table.add_column("Predictions", style="white")
    
    for rank, model in enumerate(rankings, 1):
        if model.get("n_predictions", 0) == 0:
            continue
        
        table.add_row(
            str(rank),
            model["model_version"],
            f"{model.get('accuracy', 0):.3f}",
            f"{model.get('logloss', 0):.3f}",
            str(model.get("n_predictions", 0)),
        )
    
    print(table)


@app.command()
def performance_trend(model_version: str, window_days: int = 30):
    """Show performance trend for a model."""
    from footy.performance_tracker import get_performance_tracker
    
    tracker = get_performance_tracker()
    trend = tracker.get_performance_trend(model_version, window_days=window_days)
    
    if not trend:
        print(f"[yellow]Insufficient data for {model_version}[/yellow]")
        return
    
    print(f"\n[cyan]Performance Trend: {model_version}[/cyan]")
    print(f"  Current Accuracy: {trend['current_accuracy']:.3f}")
    print(f"  Average Accuracy: {trend['avg_accuracy']:.3f}")
    print(f"  Trend Slope: {trend['trend_slope']:.4f}")
    
    if trend["degrading"]:
        print(f"  [red]⚠️  Performance degrading[/red]")
    else:
        print(f"  [green]✓ Performance stable[/green]")
    
    print(f"  R-squared: {trend['r_squared']:.3f}")
    print(f"  Data Points: {trend['n_data_points']}")


@app.command()
def performance_daily(model_version: str, days: int = 30):
    """Show daily performance breakdown."""
    from footy.performance_tracker import get_performance_tracker
    from rich.table import Table
    
    tracker = get_performance_tracker()
    daily = tracker.get_daily_performance(model_version, days=days)
    
    if not daily:
        print(f"[yellow]No data for {model_version}[/yellow]")
        return
    
    table = Table(title=f"Daily Performance: {model_version}")
    table.add_column("Date", style="cyan")
    table.add_column("Predictions", style="magenta")
    table.add_column("Accuracy", style="yellow")
    table.add_column("Logloss", style="green")
    table.add_column("Brier", style="white")
    
    for run in daily[:14]:  # Show last 14 days
        table.add_row(
            str(run["date"]),
            str(run["n_predictions"]),
            f"{run['accuracy']:.3f}",
            f"{run['logloss']:.3f}",
            f"{run['brier']:.3f}",
        )
    
    print(table)


@app.command()
def performance_health(model_version: str, days: int = 30):
    """Check model performance health against thresholds."""
    from footy.performance_tracker import get_performance_tracker
    
    tracker = get_performance_tracker()
    health = tracker.check_performance_health(model_version, days=days)
    
    status_color = "green" if health.get("status") == "healthy" else "red"
    
    print(f"\n[cyan]Performance Health: {model_version}[/cyan]")
    print(f"  Status: [{status_color}]{health.get('status')}[/{status_color}]")
    
    if health.get("status") in ["healthy", "degraded"]:
        metrics = health.get("metrics", {})
        print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"  Logloss: {metrics.get('logloss', 0):.3f}")
        print(f"  Brier: {metrics.get('brier', 0):.3f}")
        print(f"  Predictions: {metrics.get('n_predictions', 0)}")
        
        if health.get("alerts"):
            print(f"\n  [red]Alerts:[/red]")
            for alert in health["alerts"]:
                print(f"    - {alert}")


@app.command()
def performance_compare(*models: str, days: int = 365):
    """Compare performance across multiple models."""
    from footy.performance_tracker import get_performance_tracker
    from rich.table import Table
    
    if not models:
        print("[red]Error: Specify at least one model[/red]")
        print("  Usage: footy performance-compare v7_council --days 365")
        return
    
    tracker = get_performance_tracker()
    comparison = tracker.compare_models(list(models), days=days)
    
    table = Table(title=f"Model Comparison ({days} days)")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", style="magenta")
    table.add_column("Logloss", style="yellow")
    table.add_column("Brier", style="green")
    table.add_column("Home", style="white")
    table.add_column("Draw", style="white")
    table.add_column("Away", style="white")
    
    for model in comparison["models"]:
        if model.get("n_predictions", 0) == 0:
            continue
        
        table.add_row(
            model["model_version"],
            f"{model.get('accuracy', 0):.3f}",
            f"{model.get('logloss', 0):.3f}",
            f"{model.get('brier', 0):.3f}",
            f"{model.get('home_accuracy', 0):.3f}",
            f"{model.get('draw_accuracy', 0):.3f}",
            f"{model.get('away_accuracy', 0):.3f}",
        )
    
    print(table)


@app.command()
def performance_thresholds(model_version: str, min_accuracy: float = 0.45):
    """Set performance thresholds for a model."""
    from footy.performance_tracker import get_performance_tracker
    
    tracker = get_performance_tracker()
    result = tracker.set_performance_thresholds(
        model_version=model_version,
        min_accuracy=min_accuracy,
        max_logloss=0.65,
        max_brier=0.25,
        alert_threshold=-0.05,
    )
    
    print(f"[green]Thresholds configured for {result['model_version']}[/green]")
    print(f"  Min Accuracy: {result['min_accuracy']:.3f}")
    print(f"  Max Logloss: {result['max_logloss']:.3f}")
    print(f"  Max Brier: {result['max_brier']:.3f}")


# ============================================================================
# Phase 4.4: Model Degradation Alerts
# ============================================================================

@app.command()
def alerts_setup(
    model_version: str,
    accuracy_threshold: float = 0.45,
    logloss_threshold: float = 0.65,
):
    """Setup degradation monitoring for a model."""
    from footy.degradation_alerts import get_degradation_monitor
    
    monitor = get_degradation_monitor()
    result = monitor.setup_monitoring(
        model_version=model_version,
        accuracy_threshold=accuracy_threshold,
        logloss_threshold=logloss_threshold,
    )
    
    print(f"[green]Monitoring configured for {result['model_version']}[/green]")
    print(f"  Accuracy Threshold: {result['accuracy_threshold']:.3f}")
    print(f"  Logloss Threshold: {result['logloss_threshold']:.3f}")


@app.command()
def alerts_check():
    """Check all models for degradation."""
    from footy.degradation_alerts import get_degradation_monitor
    
    monitor = get_degradation_monitor()
    alerts = monitor.check_degradation()
    
    if not alerts:
        print("[green]✓ No degradation detected[/green]")
        return
    
    print(f"[red]⚠️  {len(alerts)} alert(s) triggered[/red]")
    for alert in alerts:
        severity_color = "red" if alert.severity.value == "critical" else "yellow"
        print(f"  [{severity_color}]{alert.severity.value}[/{severity_color}] {alert.model_version}: {alert.message}")


@app.command()
def alerts_list(model_version: str = None, status: str = None):
    """List degradation alerts."""
    from footy.degradation_alerts import get_degradation_monitor
    from rich.table import Table
    
    monitor = get_degradation_monitor()
    
    if model_version:
        alerts = monitor.get_alerts_for_model(model_version, status=status)
        title = f"Alerts for {model_version}"
    else:
        alerts = monitor.get_active_alerts()
        title = "Active Alerts"
    
    if not alerts:
        print(f"[yellow]No alerts{f' for {model_version}' if model_version else ''}[/yellow]")
        return
    
    table = Table(title=title)
    table.add_column("Alert ID", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Severity", style="yellow")
    table.add_column("Metric", style="green")
    table.add_column("Message", style="white")
    table.add_column("Created", style="white")
    
    for alert in alerts[:20]:
        severity_color = "red" if alert.get("severity") == "critical" else "yellow"
        
        table.add_row(
            alert.get("alert_id", "")[:30],
            alert.get("model_version", ""),
            f"[{severity_color}]{alert.get('severity', 'unknown')}[/{severity_color}]",
            alert.get("metric", ""),
            alert.get("message", "")[:40],
            str(alert.get("created_at", "")).split(".")[0],
        )
    
    print(table)


@app.command()
def alerts_summary():
    """Show alert summary."""
    from footy.degradation_alerts import get_degradation_monitor
    
    monitor = get_degradation_monitor()
    summary = monitor.get_alert_summary()
    
    print("\n[cyan]Alert Summary[/cyan]")
    print(f"  Total Alerts: {summary['total_alerts']}")
    
    print(f"\n  By Status:")
    for status, count in summary["by_status"].items():
        print(f"    {status:15s}: {count}")
    
    if summary["active_by_model"]:
        print(f"\n  Active Alerts by Model:")
        for model, count in summary["active_by_model"].items():
            print(f"    {model:30s}: {count}")


@app.command()
def alerts_acknowledge(alert_id: str):
    """Acknowledge an alert."""
    from footy.degradation_alerts import get_degradation_monitor
    
    monitor = get_degradation_monitor()
    result = monitor.acknowledge_alert(alert_id)
    
    print(f"[green]Alert acknowledged:[/green] {result['alert_id']}")


@app.command()
def alerts_resolve(alert_id: str):
    """Resolve an alert."""
    from footy.degradation_alerts import get_degradation_monitor
    
    monitor = get_degradation_monitor()
    result = monitor.resolve_alert(alert_id)
    
    print(f"[green]Alert resolved:[/green] {result['alert_id']}")


@app.command()
def alerts_snooze(alert_id: str, hours: int = 24):
    """Snooze an alert."""
    from footy.degradation_alerts import get_degradation_monitor
    
    monitor = get_degradation_monitor()
    result = monitor.snooze_alert(alert_id, hours=hours)
    
    print(f"[green]Alert snoozed until:[/green] {result['until']}")


# ============================================================================
# Phase 5.1: Understat xG Integration
# ============================================================================

@app.command()
def understat_status():
    """Show Understat provider integration status."""
    from footy.understat import get_understat_provider
    
    provider = get_understat_provider()
    status = provider.get_provider_status()
    
    print("\n[cyan]Understat Integration Status[/cyan]")
    print(f"  Provider: {status['provider']}")
    print(f"  API Configured: {status['api_configured']}")
    print(f"  Cache Enabled: {status['cache_enabled']}")
    print(f"  Cache Retention: {status['cache_retention_days']} days")
    print(f"\n  Features:")
    for feature in status['features']:
        print(f"    - {feature}")


@app.command()
def understat_team(team_name: str, season: int = 2024):
    """Get team xG statistics from Understat."""
    from footy.understat import get_understat_provider
    
    provider = get_understat_provider()
    stats = provider.get_team_season_stats(team_name, season=season)
    
    if not stats:
        print(f"[yellow]No data for {team_name}[/yellow]")
        return
    
    source_note = f" (simulated)" if stats.get("source") == "simulated" else ""
    print(f"\n[cyan]{stats['team_name']} - Season {stats['season']}{source_note}[/cyan]")
    print(f"  Games Played: {stats['games_played']}")
    print(f"  Goals: {stats['goals_for']} scored, {stats['goals_against']} conceded")
    print(f"  \n  Expected Goals:")
    print(f"    xG For: {stats['xg_for']:.2f}")
    print(f"    xG Against: {stats['xg_against']:.2f}")
    print(f"    xG Difference: {stats['xg_diff']:+.2f}")
    print(f"  \n  Non-Penalty xG:")
    print(f"    NPxG For: {stats['npxg_for']:.2f}")
    print(f"    NPxG Against: {stats['npxg_against']:.2f}")
    print(f"    NPxG Diff: {stats['npxg_diff']:+.2f}")


@app.command()
def understat_match(match_id: int):
    """Get xG statistics for a specific match."""
    from footy.understat import get_understat_provider
    
    provider = get_understat_provider()
    xg = provider.get_match_xg(match_id)
    
    if not xg:
        print(f"[yellow]No xG data for match {match_id}[/yellow]")
        return
    
    print(f"\n[cyan]{xg['home_team']} vs {xg['away_team']}[/cyan]")
    print(f"  Final Score: {xg['home_goals']} - {xg['away_goals']}")
    print(f"\n  Expected Goals:")
    print(f"    {xg['home_team']:20s} {xg['home_xg']:5.2f} xG | {xg['away_xg']:5.2f} xG {xg['away_team']}")
    print(f"  \n  Non-Penalty xG:")
    print(f"    {xg['home_team']:20s} {xg['home_npxg']:5.2f} NPxG | {xg['away_npxg']:5.2f} NPxG {xg['away_team']}")


@app.command()
def understat_team_rolling(team_name: str, matches: int = 5):
    """Get rolling xG averages for a team."""
    from footy.understat import get_understat_provider
    
    provider = get_understat_provider()
    rolling = provider.compute_team_rolling_xg(team_name, matches_window=matches)
    
    if not rolling:
        print(f"[yellow]No data for {team_name}[/yellow]")
        return
    
    print(f"\n[cyan]{rolling['team_name']} - Last {rolling['window_matches']} Matches[/cyan]")
    print(f"  Avg xG For: {rolling['avg_xg_for']:.2f}")
    print(f"  Avg xG Against: {rolling['avg_xg_against']:.2f}")
    print(f"  Avg xG Diff: {rolling['avg_xg_diff']:+.2f}")


# ============================================================================
# FBREF - ADVANCED STATISTICS PROVIDER (Phase 5.2)
# ============================================================================

@app.command()
def fbref_status():
    """Check FBRef provider health and integration status."""
    from footy.fbref import get_fbref_provider
    
    provider = get_fbref_provider()
    status = provider.get_provider_status()
    
    print("\n[cyan]FBRef Integration Status[/cyan]")
    print(f"  Status: {status.get('status', 'N/A')}")
    print(f"  Mode: {status.get('mode', 'N/A')}")
    print(f"  Cached Records: {status.get('records_cached', 'N/A')}")
    print(f"  Cache TTL: {status.get('cache_ttl', 'N/A')}")

@app.command()
def fbref_shooting(team_name: str, season: int = 2024):
    """Get team shooting statistics."""
    from footy.fbref import get_fbref_provider
    
    provider = get_fbref_provider()
    stats = provider.get_team_shooting_stats(team_name, season)
    
    if not stats:
        print(f"[yellow]No shooting data for {team_name} ({season})[/yellow]")
        return
    
    print(f"\n[cyan]{team_name} Shooting Statistics ({season})[/cyan]")
    print(f"  Shots: {stats['shots_total']:.1f}")
    print(f"  Shots on Target: {stats['shots_on_target']:.1f}")
    print(f"  Conversion: {stats['conversion']:.1%}")
    print(f"  xG: {stats['xg']:.2f}")
    print(f"  NPxG: {stats['npxg']:.2f}")
    print(f"  Shots per 90: {stats['shots_per_90']:.1f}")
    print(f"  xG per Shot: {stats['xg_per_shot']:.3f}")

@app.command()
def fbref_possession(team_name: str, season: int = 2024):
    """Get team possession statistics."""
    from footy.fbref import get_fbref_provider
    
    provider = get_fbref_provider()
    stats = provider.get_team_possession_stats(team_name, season)
    
    if not stats:
        print(f"[yellow]No possession data for {team_name} ({season})[/yellow]")
        return
    
    print(f"\n[cyan]{team_name} Possession Statistics ({season})[/cyan]")
    print(f"  Possession: {stats['possession_pct']:.1f}%")
    print(f"  Touches: {stats['touches']:.0f}")
    print(f"  Passes: {stats['passes']:.0f}")
    print(f"  Pass Completion: {stats['pass_completion']:.1%}")
    print(f"  Avg Pass Distance: {stats['pass_distance_avg']:.1f} yards")
    print(f"  Progressive Passes: {stats['progressive_passes']:.1f}")

@app.command()
def fbref_defense(team_name: str, season: int = 2024):
    """Get team defense statistics."""
    from footy.fbref import get_fbref_provider
    
    provider = get_fbref_provider()
    stats = provider.get_team_defense_stats(team_name, season)
    
    if not stats:
        print(f"[yellow]No defense data for {team_name} ({season})[/yellow]")
        return
    
    print(f"\n[cyan]{team_name} Defense Statistics ({season})[/cyan]")
    print(f"  Tackles: {stats['tackles']:.1f}")
    print(f"  Interceptions: {stats['interceptions']:.1f}")
    print(f"  Blocks: {stats['blocks']:.1f}")
    print(f"  Clearances: {stats['clearances']:.1f}")
    print(f"  Aerial Duels Won: {stats['aerial_duels_won']:.1f}")
    print(f"  Aerial Duel Success: {stats['aerial_duel_success_pct']:.1%}")
    print(f"  Fouls Committed: {stats['fouls_committed']:.1f}")
    print(f"  Fouls Drawn: {stats['fouls_drawn']:.1f}")

@app.command()
def fbref_passing(team_name: str, season: int = 2024):
    """Get team passing statistics."""
    from footy.fbref import get_fbref_provider
    
    provider = get_fbref_provider()
    stats = provider.get_team_passing_stats(team_name, season)
    
    if not stats:
        print(f"[yellow]No passing data for {team_name} ({season})[/yellow]")
        return
    
    print(f"\n[cyan]{team_name} Passing Statistics ({season})[/cyan]")
    print(f"  Short Pass Completion: {stats['pass_completion_short']:.1%}")
    print(f"  Medium Pass Completion: {stats['pass_completion_medium']:.1%}")
    print(f"  Long Pass Completion: {stats['pass_completion_long']:.1%}")
    print(f"  Key Passes: {stats['key_passes']:.1f}")
    print(f"  Passes into Penalty: {stats['passes_into_penalty']:.1f}")
    print(f"  Crosses: {stats['crosses']:.1f}")
    print(f"  Cross Completion: {stats['cross_completion']:.1%}")
    print(f"  Through Balls: {stats['through_balls']:.1f}")

@app.command()
def fbref_compare(team1: str, team2: str, season: int = 2024):
    """Compare FBRef statistics between two teams."""
    from footy.fbref import get_fbref_provider
    
    provider = get_fbref_provider()
    comparison = provider.compute_team_stats_comparison(team1, team2, season)
    
    print(f"\n[cyan]{team1} vs {team2} - Statistical Comparison ({season})[/cyan]")
    
    print(f"\n[yellow]Shooting Advantage ({team1} vs {team2}):[/yellow]")
    for key, val in comparison['shooting_advantage'].items():
        emoji = "📈" if val > 0 else "📉" if val < 0 else "➡️ "
        print(f"  {emoji} {key}: {val:+.1f}%")
    
    print(f"\n[yellow]Possession Advantage ({team1} vs {team2}):[/yellow]")
    for key, val in comparison['possession_advantage'].items():
        emoji = "📈" if val > 0 else "📉" if val < 0 else "➡️ "
        print(f"  {emoji} {key}: {val:+.1f}%")
    
    print(f"\n[yellow]Defense Advantage ({team1} vs {team2}):[/yellow]")
    for key, val in comparison['defense_advantage'].items():
        emoji = "📈" if val > 0 else "📉" if val < 0 else "➡️ "
        print(f"  {emoji} {key}: {val:+.1f}%")

@app.command()
def fbref_all(team_name: str, season: int = 2024):
    """Get complete FBRef statistics for a team."""
    from footy.fbref import get_fbref_provider
    
    provider = get_fbref_provider()
    stats = provider.get_all_team_stats(team_name, season)
    
    print(f"\n[cyan]Complete FBRef Statistics - {team_name} ({season})[/cyan]")
    print(f"\n[yellow]Shooting:[/yellow]")
    for key, val in stats['shooting'].items():
        if key not in ['team_id', 'season']:
            print(f"  {key}: {val}")
    
    print(f"\n[yellow]Possession:[/yellow]")
    for key, val in stats['possession'].items():
        if key not in ['team_id', 'season']:
            print(f"  {key}: {val}")
    
    print(f"\n[yellow]Defense:[/yellow]")
    for key, val in stats['defense'].items():
        if key not in ['team_id', 'season']:
            print(f"  {key}: {val}")
    
    print(f"\n[yellow]Passing:[/yellow]")
    for key, val in stats['passing'].items():
        if key not in ['team_id', 'season']:
            print(f"  {key}: {val}")


# ============================================================================
# Self-Test
# ============================================================================

@app.command()
def self_test(
    smoke: bool = typer.Option(False, "--smoke", help="Include live-database smoke tests"),
    coverage: bool = typer.Option(False, "--cov", help="Measure code coverage"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose test output"),
    fast: bool = typer.Option(False, "--fast", help="Skip slow tests"),
    pattern: str = typer.Option(None, "-k", help="Only run tests matching this pattern"),
):
    """Run the full self-test suite (pytest).

    Quick check:   footy self-test
    Full + smoke:  footy self-test --smoke
    With coverage: footy self-test --cov
    Single file:   footy self-test -k test_models_elo
    """
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=footy", "--cov=web", "--cov-report=term-missing"])

    if fast:
        cmd.extend(["-m", "not slow"])

    if not smoke:
        cmd.extend(["--ignore=tests/test_smoke.py"])

    if pattern:
        cmd.extend(["-k", pattern])

    print(f"[cyan]Running:[/cyan] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    raise typer.Exit(code=result.returncode)


if __name__ == "__main__":
    app()
