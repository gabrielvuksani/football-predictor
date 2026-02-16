"""AI sub-commands: preview, value bets, review, news extraction, form analysis."""
from __future__ import annotations

import typer

from footy.cli._shared import console

app = typer.Typer(help="AI-powered analysis commands (Ollama LLM).")


@app.command()
def preview(match_id: int = None, league: str = None):
    """AI-generated pre-match preview.

    Provide --match-id for a single match, or --league (PL/PD/SA/BL1) for a round summary.
    """
    from footy.llm.insights import preview_match, league_round_summary
    from rich.panel import Panel
    from rich.console import Console as _Console
    _con = _Console()

    if match_id:
        pv = preview_match(match_id)
        if pv.get("error"):
            console.print(f"[red]{pv['error']}[/red]")
            return
        _con.print(Panel(
            f"[bold]{pv.get('home_team','?')} vs {pv.get('away_team','?')}[/bold]\n\n"
            f"{pv.get('preview','')}\n\n"
            f"[cyan]Prediction:[/cyan] {pv.get('prediction','')} ({pv.get('confidence','')}) confidence\n"
            + ("\n".join(f"  • {s}" for s in pv.get('key_stats', [])) or "")
            + ("\n\n[yellow]Value bets:[/yellow]\n" + "\n".join(f"  → {v}" for v in pv.get('value_bets', [])) if pv.get('value_bets') else ""),
            title="Match Preview",
        ))
    elif league:
        summary = league_round_summary(league.upper())
        _con.print(Panel(
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
        for code in ["PL", "PD", "SA", "BL1", "FL1"]:
            summary = league_round_summary(code)
            if summary.get("matches", 0) > 0:
                console.print(f"\n[bold cyan]{code}[/bold cyan] — {summary.get('summary','')}")
                console.print(f"  Headline: {summary.get('headline_pick','')}")


@app.command()
def value(min_edge: float = 0.05):
    """Scan upcoming matches for value bets where model edge exceeds market odds."""
    from footy.llm.insights import value_bet_scan, ai_value_commentary
    from rich.table import Table

    values = value_bet_scan(min_edge=min_edge)

    if not values:
        console.print(f"[yellow]No value bets found with edge ≥ {min_edge:.0%}[/yellow]")
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
    console.print(table)

    commentary = ai_value_commentary(values)
    console.print(f"\n[italic]{commentary}[/italic]")


@app.command()
def review(days: int = 3, league: str = None):
    """AI review of recent prediction accuracy."""
    from footy.llm.insights import post_match_review
    from rich.panel import Panel
    from rich.console import Console as _Console
    _con = _Console()

    rev = post_match_review(days_back=days, competition_code=league)

    if rev["matches_reviewed"] == 0:
        console.print("[yellow]No recent finished matches with predictions to review[/yellow]")
        return

    acc_color = "green" if rev["accuracy"] >= 0.50 else ("yellow" if rev["accuracy"] >= 0.40 else "red")
    _con.print(Panel(
        f"[bold]Post-Match Review[/bold] (last {days} days"
        + (f", {league}" if league else "") + ")\n\n"
        f"[{acc_color}]Accuracy: {rev['accuracy']:.0%}[/{acc_color}] "
        f"({rev['correct']}/{rev['matches_reviewed']})\n\n"
        f"{rev['review']}\n\n"
        + ("[red]Notable misses:[/red]\n" + "\n".join(
            f"  ✗ {m['match']} {m['score']} — predicted {m['predicted']} "
            f"(was {m['actual']}, conf {m['confidence']:.0%})"
            for m in rev.get("misses", [])
        ) if rev.get("misses") else "[green]No major misses![/green]"),
        title="AI Review",
    ))


@app.command()
def extract_news(team: str = None, days_back: int = 2):
    """Extract team news from GDELT and analyze with Ollama."""
    from footy.llm.insights import extract_team_news_signal

    if not team:
        console.print("[red]Error: --team is required[/red]")
        return

    console.print(f"[cyan]Extracting news for {team}...[/cyan]")
    signal = extract_team_news_signal(team, days_back=days_back)

    console.print(f"\n[bold]{team}[/bold]")
    console.print(f"  Availability: {signal['availability_score']:+.1f}")
    console.print(f"  Headlines: {signal['headline_count']}")
    console.print(f"  Summary: {signal.get('summary', 'N/A')}")
    if signal.get('key_notes'):
        console.print(f"  Key Notes:")
        for note in signal['key_notes']:
            console.print(f"    - {note}")
    if signal.get('likely_absences'):
        console.print(f"  Likely Absences:")
        for absence in signal['likely_absences']:
            console.print(f"    - {absence}")


@app.command()
def analyze_form(team: str = None, matches: int = 10):
    """Analyze recent team form using Ollama."""
    from footy.llm.insights import analyze_team_form

    if not team:
        console.print("[red]Error: --team is required[/red]")
        return

    console.print(f"[cyan]Analyzing form for {team}...[/cyan]")
    form = analyze_team_form(team, matches_window=matches)

    console.print(f"\n[bold]{team}[/bold]")
    console.print(f"  Recent Form: {form['recent_form']}")
    if form.get('record'):
        console.print(f"  Record: {form['record']}")
    console.print(f"  Momentum: {form['momentum']:+.1f}")
    if form.get('key_trends'):
        console.print(f"  Key Trends:")
        for trend in form['key_trends']:
            console.print(f"    - {trend}")
    if form.get('concern_areas'):
        console.print(f"  Concern Areas:")
        for area in form['concern_areas']:
            console.print(f"    - {area}")


@app.command()
def explain_match(
    match_id: int = None,
    home_team: str = None,
    away_team: str = None,
    home_prob: float = 0.5,
    draw_prob: float = 0.25,
    away_prob: float = 0.25,
):
    """Generate LLM explanation for a match prediction."""
    from footy.llm.insights import explain_match as _explain

    if not (home_team and away_team):
        console.print("[red]Error: --home-team and --away-team are required[/red]")
        return

    match_id = match_id or 0

    console.print(f"[cyan]Generating explanation for {home_team} vs {away_team}...[/cyan]")
    explanation = _explain(
        match_id=match_id, home_team=home_team, away_team=away_team,
        home_pred=home_prob, draw_pred=draw_prob, away_pred=away_prob,
    )

    console.print(
        f"\n[bold]{home_team} ({home_prob:.1%})[/bold]"
        f" vs [bold]{away_team} ({away_prob:.1%})[/bold] "
        f"(Draw: {draw_prob:.1%})"
    )
    console.print(f"\n[cyan]Explanation:[/cyan]")
    console.print(f"  {explanation.get('explanation', 'N/A')}")

    if explanation.get('key_factors'):
        console.print(f"\n[cyan]Key Factors:[/cyan]")
        for factor in explanation['key_factors']:
            console.print(f"  - {factor}")

    console.print(f"\n[cyan]Confidence:[/cyan] {explanation.get('confidence_level', 'N/A')}")


@app.command()
def status():
    """Check status of Ollama AI insights system."""
    from footy.llm.insights import get_insights_status

    st = get_insights_status()

    console.print("\n[cyan]Insights System Status[/cyan]")
    console.print(f"  Status: {st['status']}")
    console.print(f"  LLM Health: {st.get('llm_health', 'Unknown')}")
    console.print(f"  Cache - Predictions: {st.get('cache_predictions', 0)}")
    console.print(f"  Cache - Metadata: {st.get('cache_metadata', 0)}")
    console.print(f"  Last Check: {st.get('timestamp', 'N/A')}")
