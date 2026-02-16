"""Root-level CLI commands: go, refresh, matchday, nuke, serve, update, self-test."""
from __future__ import annotations

import time
import typer

from footy.cli._shared import console, _setup_logging, _pipeline, _council, _odds, _extras

app = typer.Typer(add_completion=False)


def _ingest_new_apis():
    """Call optional API providers (The Odds API, FPL, TheSportsDB).

    Each provider is wrapped in try/except so failures are non-fatal.
    """
    # The Odds API — multi-bookmaker odds for upcoming matches
    try:
        from footy.providers.the_odds_api import ingest_odds_to_db
        from footy.db import connect
        con = connect()
        matched = ingest_odds_to_db(con)
        console.print(f"  The Odds API: {matched} matches with odds")
    except Exception as e:
        console.print(f"[yellow]  The Odds API warning:[/yellow] {e}")

    # FPL — Premier League injury / availability data
    try:
        from footy.providers.fpl import get_all_teams_availability
        avail = get_all_teams_availability()
        if avail:
            console.print(f"  FPL: availability data for {len(avail)} teams")
        else:
            console.print("  FPL: no availability data")
    except Exception as e:
        console.print(f"[yellow]  FPL warning:[/yellow] {e}")

    # TheSportsDB — team metadata / venue enrichment
    try:
        from footy.providers.thesportsdb import enrich_matches_with_venue
        from footy.db import connect
        con = connect()
        n_enriched = enrich_matches_with_venue(con)
        console.print(f"  TheSportsDB: enriched {n_enriched} matches with venue data")
    except Exception as e:
        console.print(f"[yellow]  TheSportsDB warning:[/yellow] {e}")


@app.command()
def go(skip_history: bool = False):
    """Master command: full pipeline from ingest to predictions.

    Runs: history → ingest → extras → odds → new APIs → train (Elo+Poisson) →
          train (9-expert council + multi-model stack) → predict → score →
          compute-h2h + xG → export pages

    v10 council: 11 experts + HistGBM/RF/LR multi-model stack with learned weights.
    """
    from footy.db import connect
    t0 = time.perf_counter()
    _setup_logging()

    steps = [
        ("1/11", "Ingest history (fdcuk)"),
        ("2/11", "Ingest fixtures (API)"),
        ("3/11", "Ingest extras + odds"),
        ("4/11", "Ingest new API sources"),
        ("5/11", "Train Elo + Poisson"),
        ("6/11", "Train council model"),
        ("7/11", "Predict council"),
        ("8/11", "Score finished predictions"),
        ("9/11", "Compute H2H + xG"),
        ("10/11", "Export static pages"),
        ("11/11", "Done"),
    ]

    def step(i, msg):
        console.print(f"[cyan]Step {steps[i][0]}[/cyan] {msg}")

    if not skip_history:
        step(0, "Ingesting history from football-data.co.uk...")
        try:
            n_hist = _pipeline().ingest_history_fdcuk(n_seasons=25, verbose=True)
            console.print(f"  History: {n_hist} matches loaded/updated")
        except Exception as e:
            console.print(f"[yellow]History warning:[/yellow] {e}")
    else:
        console.print("[yellow]Step 1/11[/yellow] Skipping history (--skip-history)")

    step(1, "Ingesting fixtures from API (last 365 days + 7 ahead)...")
    try:
        _pipeline().ingest(days_back=365, verbose=True)
    except Exception as e:
        console.print(f"[yellow]Ingest warning:[/yellow] {e}")

    step(2, "Ingesting extras + odds...")
    try:
        _extras()(verbose=True)
    except Exception as e:
        console.print(f"[yellow]Extras warning:[/yellow] {e}")
    try:
        _odds()(verbose=True)
    except Exception as e:
        console.print(f"[yellow]Odds warning:[/yellow] {e}")

    step(3, "Ingesting new API sources (The Odds API, FPL, TheSportsDB)...")
    _ingest_new_apis()

    step(4, "Training Elo + Poisson...")
    n_elo = _pipeline().update_elo_from_finished(verbose=True)
    state = _pipeline().refit_poisson(verbose=True)
    console.print(f"  Elo: {n_elo} matches | Poisson: {len(state.get('teams', []))} teams")

    step(5, "Training council model (11 experts + multi-model stack)...")
    con = connect()
    rc = _council()[0](con, eval_days=365, verbose=True)
    console.print(f"  council: {rc}")

    step(6, "Predicting council...")
    nc = _council()[1](con, verbose=True)
    console.print(f"  council: {nc} predictions")

    step(7, "Scoring finished predictions...")
    try:
        sc = _pipeline().score_finished_predictions(verbose=True)
        console.print(f"  scored: {sc}")
    except Exception as e:
        console.print(f"[yellow]Scoring warning:[/yellow] {e}")

    step(8, "Computing H2H + xG...")
    try:
        from footy.h2h import recompute_h2h_stats
        recompute_h2h_stats(con, verbose=True)
    except Exception as e:
        console.print(f"[yellow]H2H warning:[/yellow] {e}")
    try:
        from footy.xg import backfill_xg_for_finished_matches
        backfill_xg_for_finished_matches(con, verbose=True)
    except Exception as e:
        console.print(f"[yellow]xG warning:[/yellow] {e}")

    # ── Pages export ──────────────────────────────────────────────────
    step(9, "Exporting static pages...")
    try:
        from footy.cli.pages_cmds import export as _pages_export
        _pages_export(out="docs", days=14)
    except Exception as e:
        console.print(f"[yellow]Pages export warning:[/yellow] {e}")

    dt = time.perf_counter() - t0
    console.print(f"\n[green bold]Pipeline complete in {dt:.0f}s[/green bold]")


@app.command()
def refresh():
    """Quick daily update: ingest recent → extras → odds → APIs → retrain council → predict → H2H → export pages.

    Faster than `go` — skips history download. Use this for daily cron jobs.
    """
    from footy.db import connect
    t0 = time.perf_counter()
    _setup_logging()

    console.print("[cyan]Step 1/9[/cyan] Ingest recent fixtures...")
    try:
        _pipeline().ingest(days_back=30, verbose=True)
    except Exception as e:
        console.print(f"[yellow]Ingest warning:[/yellow] {e}")

    console.print("[cyan]Step 2/9[/cyan] Extras + odds...")
    try:
        _extras()(verbose=True)
    except Exception as e:
        console.print(f"[yellow]Extras warning:[/yellow] {e}")
    try:
        _odds()(verbose=True)
    except Exception as e:
        console.print(f"[yellow]Odds warning:[/yellow] {e}")

    console.print("[cyan]Step 3/9[/cyan] New API sources...")
    _ingest_new_apis()

    console.print("[cyan]Step 4/9[/cyan] Train Elo + Poisson...")
    n_elo = _pipeline().update_elo_from_finished(verbose=True)
    _pipeline().refit_poisson(verbose=True)
    console.print(f"  Elo: {n_elo} matches")

    console.print("[cyan]Step 5/9[/cyan] Retrain council (11 experts + multi-model stack)...")
    con = connect()
    rc = _council()[0](con, eval_days=365, verbose=True)
    console.print(f"  council: {rc}")

    console.print("[cyan]Step 6/9[/cyan] Predict council...")
    nc = _council()[1](con, lookahead_days=14, verbose=True)
    console.print(f"  council: {nc} predictions")

    console.print("[cyan]Step 7/9[/cyan] Score finished predictions...")
    try:
        sc = _pipeline().score_finished_predictions(verbose=True)
        console.print(f"  scored: {sc}")
    except Exception as e:
        console.print(f"[yellow]Scoring warning:[/yellow] {e}")

    console.print("[cyan]Step 8/9[/cyan] Compute H2H...")
    try:
        from footy.h2h import recompute_h2h_stats
        recompute_h2h_stats(con, verbose=True)
    except Exception as e:
        console.print(f"[yellow]H2H warning:[/yellow] {e}")

    console.print("[cyan]Step 9/9[/cyan] Export static pages...")
    try:
        from footy.cli.pages_cmds import export as _pages_export
        _pages_export(out="docs", days=14)
    except Exception as e:
        console.print(f"[yellow]Pages export warning:[/yellow] {e}")

    dt = time.perf_counter() - t0
    console.print(f"\n[green bold]Refresh complete in {dt:.0f}s[/green bold]")


@app.command()
def matchday():
    """Weekend preview: refresh + AI preview for all leagues.

    Runs a full refresh then generates AI previews for every league.
    Perfect to run Friday evening or Saturday morning.
    """
    t0 = time.perf_counter()

    # run refresh first
    console.print("[bold cyan]== REFRESH ==[/bold cyan]")
    refresh()

    # AI preview for each league
    console.print("\n[bold cyan]== MATCHDAY PREVIEW ==[/bold cyan]")
    try:
        from footy.llm.insights import league_round_summary
        from rich.panel import Panel
        from rich.console import Console as _Console
        _con = _Console()

        for code in ["PL", "PD", "SA", "BL1", "FL1"]:
            try:
                summary = league_round_summary(code)
                if summary.get("matches", 0) > 0:
                    _con.print(Panel(
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
                console.print(f"[yellow]{code} preview skipped:[/yellow] {e}")
    except ImportError:
        console.print("[yellow]AI preview unavailable (Ollama not configured)[/yellow]")

    dt = time.perf_counter() - t0
    console.print(f"\n[green bold]Matchday prep complete in {dt:.0f}s[/green bold]")


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
    console.print("[bold red]== NUKE: Full rebuild ==[/bold red]")

    s = get_settings()
    db_path = s.db_path
    for f in [db_path, db_path + ".wal"]:
        if os.path.exists(f):
            os.remove(f)
            console.print(f"[red]Deleted {f}[/red]")

    console.print("[red]Resetting all model states...[/red]")
    _pipeline().reset_states(verbose=True)

    console.print("[cyan]Running full pipeline from scratch...[/cyan]")
    go(skip_history=False)

    dt = time.perf_counter() - t0
    console.print(f"\n[green bold]Nuke complete in {dt:.0f}s[/green bold]")


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

    project_root = str(Path(__file__).resolve().parents[3])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import uvicorn

    my_pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "uvicorn.*web\\.api"], text=True
        ).strip()
        for pid_str in out.splitlines():
            pid = int(pid_str)
            if pid != my_pid:
                console.print(f"[yellow]Killing stale server process {pid}…[/yellow]")
                os.kill(pid, signal.SIGTERM)
    except (subprocess.CalledProcessError, ValueError):
        pass

    def _port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) == 0

    chosen = port
    for attempt in range(max_retries):
        if not _port_in_use(chosen):
            break
        console.print(f"[yellow]Port {chosen} is in use, trying {chosen + 1}…[/yellow]")
        chosen += 1
    else:
        console.print(f"[red]Could not find a free port in range {port}–{chosen}. Aborting.[/red]")
        raise typer.Exit(code=1)

    if chosen != port:
        console.print(f"[green]Found free port {chosen}[/green]")
    console.print(f"[green]Starting Footy Predictor on http://{host}:{chosen}[/green]")
    console.print(f"[cyan]Access from other devices at http://<this-machine-ip>:{chosen}[/cyan]")
    uvicorn.run("web.api:app", host=host, port=chosen, log_level="info")


@app.command()
def update():
    """Quick ingest → train → predict → metrics."""
    t0 = time.perf_counter()
    console.print("[cyan]Step 1/4[/cyan] ingest")
    _pipeline().ingest(verbose=True)

    console.print("[cyan]Step 2/4[/cyan] train (Elo + Poisson)")
    new_elo = _pipeline().update_elo_from_finished(verbose=True)
    _pipeline().refit_poisson(verbose=True)

    console.print("[cyan]Step 3/4[/cyan] predict upcoming")
    from footy.models.council import predict_upcoming as _predict
    from footy.db import connect as _connect
    n_pred = _predict(_connect(), verbose=True)

    console.print("[cyan]Step 4/4[/cyan] metrics (only counts matches that were predicted before finishing)")
    m = _pipeline().backtest_metrics()
    dt = time.perf_counter() - t0
    console.print(f"[green]update complete[/green] new_elo={new_elo} predicted={n_pred} in {dt:.1f}s")
    console.print(f"[cyan]metrics[/cyan] {m}")


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

    console.print(f"[cyan]Running:[/cyan] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(__import__("pathlib").Path(__file__).resolve().parents[3]))
    raise typer.Exit(code=result.returncode)
