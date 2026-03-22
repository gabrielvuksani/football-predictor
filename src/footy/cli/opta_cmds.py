"""Opta sub-commands: fetch and show Opta Analyst predictions."""
from __future__ import annotations

import typer

from footy.cli._shared import console

app = typer.Typer(help="Opta Analyst win-probability predictions.")


@app.command()
def fetch(
    competition: str = typer.Option(
        None, "--league", "-l",
        help="Competition code to filter (PL, PD, BL1, SA, FL1). All if omitted.",
    ),
):
    """Scrape Opta predictions from theanalyst.com."""
    from footy.db import connect
    from footy.providers.opta_analyst import fetch_opta_predictions

    con = connect()
    preds = fetch_opta_predictions(con, competition=competition)
    if preds:
        console.print(f"[green]Fetched {len(preds)} Opta prediction(s)[/green]")
        for p in preds:
            console.print(
                f"  {p['home_team']} vs {p['away_team']} ({p['date']})  "
                f"H {p['home_win']:.0%}  D {p['draw']:.0%}  A {p['away_win']:.0%}"
            )
    else:
        console.print(
            "[yellow]No predictions scraped — page structure may have "
            "changed or content is paywalled.[/yellow]"
        )


@app.command()
def show():
    """Show cached Opta predictions from the database."""
    from footy.db import connect
    from footy.providers.opta_analyst import get_cached_predictions

    con = connect()
    preds = get_cached_predictions(con)
    if not preds:
        console.print("[yellow]No cached Opta predictions.[/yellow]")
        return

    console.print(f"[cyan]Cached Opta predictions ({len(preds)}):[/cyan]\n")
    for p in preds:
        parts = p["match_key"].split("__")
        home = parts[0] if len(parts) > 0 else "?"
        away = parts[1] if len(parts) > 1 else "?"
        date = parts[2] if len(parts) > 2 else "?"
        console.print(
            f"  {home} vs {away} ({date})  "
            f"H {p['home_win']:.0%}  D {p['draw']:.0%}  A {p['away_win']:.0%}  "
            f"[dim](scraped {p['scraped_at']})[/dim]"
        )
