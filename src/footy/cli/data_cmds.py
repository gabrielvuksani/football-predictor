"""Data sub-commands: ingest, history, extras, odds, cache, H2H, xG, news."""
from __future__ import annotations

import time
import typer

from footy.cli._shared import console, _pipeline, _odds, _extras

app = typer.Typer(help="Data ingestion and management commands.")


@app.command()
def ingest(days_back: int = 30, days_forward: int = 7, chunk_days: int = 10):
    """Ingest recent fixtures from the active provider stack."""
    t0 = time.perf_counter()
    n = _pipeline().ingest(days_back=days_back, days_forward=days_forward, chunk_days=chunk_days, verbose=True)
    dt = time.perf_counter() - t0
    console.print(f"[green]Inserted/updated[/green] {n} matches in {dt:.1f}s")


@app.command()
def openfootball(season: str | None = None):
    """Seed fixtures from the free openfootball JSON feed."""
    n = _pipeline().ingest_openfootball_schedule(season=season, verbose=True)
    console.print(f"[green]OpenFootball seeded[/green] rows={n}")


@app.command()
def clubelo():
    """Refresh external Club Elo ratings into the live elo_state table."""
    n = _pipeline().refresh_external_elo(verbose=True)
    console.print(f"[green]Club Elo refreshed[/green] teams={n}")


@app.command()
def history(n_seasons: int = 8):
    """Ingest historical match data from football-data.co.uk."""
    n = _pipeline().ingest_history_fdcuk(n_seasons=n_seasons, verbose=True)
    console.print(f"[green]History ingested[/green] rows={n}")


@app.command()
def extras(n_seasons: int = 8):
    """Ingest extra match statistics from football-data.co.uk."""
    n = _extras()(n_seasons=n_seasons, verbose=True)
    console.print(f"[green]Extras upserted[/green] rows={n}")


@app.command()
def fixtures_odds():
    """Ingest upcoming fixtures and attach odds."""
    n = _odds()(verbose=True)
    console.print({"attached_odds": n})


@app.command()
def odds():
    """Update odds for upcoming matches using external sources or predictions."""
    from footy.db import connect
    from footy.providers.odds_scraper import update_upcoming_match_odds, fill_upcoming_odds_from_predictions
    con = connect()

    console.print("[cyan]Step 1/2[/cyan] External sources (ODD-API, etc.)")
    r1 = update_upcoming_match_odds(con, verbose=True)
    console.print(r1)

    console.print("[cyan]Step 2/2[/cyan] Fallback to model predictions")
    r2 = fill_upcoming_odds_from_predictions(con, verbose=True)
    console.print(r2)


@app.command()
def api_football(lookahead_days: int = 7, stale_hours: int = 6):
    """Fetch API-Football context (fixtures/lineups/stats + injuries) for upcoming matches."""
    from footy.db import connect
    from footy.providers.api_football import map_upcoming_matches, upsert_context

    con = connect()
    console.print(f"Step 1/2 map fixtures (lookahead_days={lookahead_days})")
    nmap = map_upcoming_matches(con, lookahead_days=lookahead_days, verbose=True)

    console.print(f"Step 2/2 ingest context (stale_hours={stale_hours})")
    nctx = upsert_context(con, stale_hours=stale_hours, verbose=True)

    console.print({"mapped": nmap, "context_rows_written": nctx})


@app.command()
def news(days_back: int = 2, max_records: int = 10):
    """Ingest team news from GDELT for upcoming matches."""
    n = _pipeline().ingest_news_for_teams(days_back=days_back, max_records=max_records)
    console.print(f"[green]Inserted[/green] {n} news rows")


@app.command()
def h2h():
    """Recompute all head-to-head statistics from finished matches."""
    from footy.db import connect
    from footy.h2h import recompute_h2h_stats
    con = connect()
    r = recompute_h2h_stats(con, verbose=True)
    console.print(r)


@app.command()
def xg():
    """Compute xG for all finished matches from available statistics."""
    from footy.db import connect
    from footy.xg import backfill_xg_for_finished_matches
    con = connect()
    r = backfill_xg_for_finished_matches(con, verbose=True)
    console.print(r)


@app.command()
def fbref():
    """Ingest FBref advanced team stats (xG, shooting, passing, defense)."""
    from footy.providers.data_ingest import ingest_fbref_stats
    t0 = time.perf_counter()
    n = ingest_fbref_stats()
    dt = time.perf_counter() - t0
    console.print(f"[green]FBref stats ingested[/green] rows={n} ({dt:.1f}s)")


@app.command()
def transfermarkt():
    """Ingest Transfermarkt squad values and injuries."""
    from footy.providers.data_ingest import ingest_transfermarkt_values, ingest_transfermarkt_injuries
    t0 = time.perf_counter()
    n_val = ingest_transfermarkt_values()
    n_inj = ingest_transfermarkt_injuries()
    dt = time.perf_counter() - t0
    console.print(f"[green]Transfermarkt ingested[/green] values={n_val} injuries={n_inj} ({dt:.1f}s)")


@app.command()
def derived():
    """Compute derived stats (referee, venue) from match history."""
    from footy.providers.data_ingest import compute_referee_stats, compute_venue_stats
    t0 = time.perf_counter()
    n_ref = compute_referee_stats()
    n_ven = compute_venue_stats()
    dt = time.perf_counter() - t0
    console.print(f"[green]Derived stats computed[/green] referees={n_ref} venues={n_ven} ({dt:.1f}s)")


@app.command(name="enrich-all")
def enrich_all():
    """Run ALL enrichment data pipelines (FBref, Transfermarkt, derived stats)."""
    from footy.providers.data_ingest import refresh_all_enrichment
    t0 = time.perf_counter()
    result = refresh_all_enrichment(verbose=True)
    dt = time.perf_counter() - t0
    console.print(f"[green]Enrichment complete[/green] ({dt:.1f}s)")
    for k, v in result.items():
        console.print(f"  {k}: {v}")


@app.command()
def reset():
    """Reset all model states."""
    _pipeline().reset_states(verbose=True)


@app.command()
def cache_stats():
    """Show cache statistics and usage."""
    from footy.cache import get_cache

    cache = get_cache()
    stats = cache.get_stats()

    console.print("\n[cyan]Cache Statistics[/cyan]")
    console.print(f"  Predictions: {stats['total_predictions']} total, {stats['expired_predictions']} expired")
    console.print(f"  Metadata: {stats['total_metadata']} total, {stats['expired_metadata']} expired")
    console.print(f"  By category:")
    for cat, count in stats["metadata_by_category"].items():
        console.print(f"    {cat:15s}: {count:5d}")


@app.command()
def cache_cleanup(full: bool = False):
    """Clean up cache (remove expired entries)."""
    from footy.cache import get_cache

    if full:
        console.print("[yellow]Clearing entire cache...[/yellow]")
        get_cache().clear()
        console.print("[green]Cache cleared[/green]")
    else:
        console.print("[cyan]Removing expired entries...[/cyan]")
        result = get_cache().cleanup(delete_expired=True)
        console.print(f"[green]Cleanup complete:[/green]")
        console.print(f"  Deleted predictions: {result['deleted_predictions']}")
        console.print(f"  Deleted metadata: {result['deleted_metadata']}")
