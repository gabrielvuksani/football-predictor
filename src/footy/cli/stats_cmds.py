"""Stats sub-commands: Understat xG + FBRef advanced statistics."""
from __future__ import annotations

import typer

from footy.cli._shared import console

app = typer.Typer(help="External stats providers (Understat, FBRef).")


# ------------------------------------------------------------------ Understat
@app.command()
def understat_status():
    """Show Understat provider integration status."""
    from footy.understat import get_understat_provider

    provider = get_understat_provider()
    status = provider.get_provider_status()

    console.print("\n[cyan]Understat Integration Status[/cyan]")
    console.print(f"  Provider: {status['provider']}")
    console.print(f"  API Configured: {status['api_configured']}")
    console.print(f"  Cache Enabled: {status['cache_enabled']}")
    console.print(f"  Cache Retention: {status['cache_retention_days']} days")
    console.print(f"\n  Features:")
    for feature in status['features']:
        console.print(f"    - {feature}")


@app.command()
def understat_team(team_name: str, season: int = 2024):
    """Get team xG statistics from Understat."""
    from footy.understat import get_understat_provider

    provider = get_understat_provider()
    stats = provider.get_team_season_stats(team_name, season=season)

    if not stats:
        console.print(f"[yellow]No data for {team_name}[/yellow]")
        return

    source_note = f" (simulated)" if stats.get("source") == "simulated" else ""
    console.print(f"\n[cyan]{stats['team_name']} - Season {stats['season']}{source_note}[/cyan]")
    console.print(f"  Games Played: {stats['games_played']}")
    console.print(f"  Goals: {stats['goals_for']} scored, {stats['goals_against']} conceded")
    console.print(f"  \n  Expected Goals:")
    console.print(f"    xG For: {stats['xg_for']:.2f}")
    console.print(f"    xG Against: {stats['xg_against']:.2f}")
    console.print(f"    xG Difference: {stats['xg_diff']:+.2f}")
    console.print(f"  \n  Non-Penalty xG:")
    console.print(f"    NPxG For: {stats['npxg_for']:.2f}")
    console.print(f"    NPxG Against: {stats['npxg_against']:.2f}")
    console.print(f"    NPxG Diff: {stats['npxg_diff']:+.2f}")


@app.command()
def understat_match(match_id: int):
    """Get xG statistics for a specific match."""
    from footy.understat import get_understat_provider

    provider = get_understat_provider()
    xg = provider.get_match_xg(match_id)

    if not xg:
        console.print(f"[yellow]No xG data for match {match_id}[/yellow]")
        return

    console.print(f"\n[cyan]{xg['home_team']} vs {xg['away_team']}[/cyan]")
    console.print(f"  Final Score: {xg['home_goals']} - {xg['away_goals']}")
    console.print(f"\n  Expected Goals:")
    console.print(f"    {xg['home_team']:20s} {xg['home_xg']:5.2f} xG | {xg['away_xg']:5.2f} xG {xg['away_team']}")
    console.print(f"  \n  Non-Penalty xG:")
    console.print(f"    {xg['home_team']:20s} {xg['home_npxg']:5.2f} NPxG | {xg['away_npxg']:5.2f} NPxG {xg['away_team']}")


@app.command()
def understat_rolling(team_name: str, matches: int = 5):
    """Get rolling xG averages for a team."""
    from footy.understat import get_understat_provider

    provider = get_understat_provider()
    rolling = provider.compute_team_rolling_xg(team_name, matches_window=matches)

    if not rolling:
        console.print(f"[yellow]No data for {team_name}[/yellow]")
        return

    console.print(f"\n[cyan]{rolling['team_name']} - Last {rolling['window_matches']} Matches[/cyan]")
    console.print(f"  Avg xG For: {rolling['avg_xg_for']:.2f}")
    console.print(f"  Avg xG Against: {rolling['avg_xg_against']:.2f}")
    console.print(f"  Avg xG Diff: {rolling['avg_xg_diff']:+.2f}")


# ------------------------------------------------------------------ FBRef
@app.command()
def fbref_status():
    """Check FBRef provider health and integration status."""
    from footy.fbref import get_fbref_provider

    provider = get_fbref_provider()
    status = provider.get_provider_status()

    console.print("\n[cyan]FBRef Integration Status[/cyan]")
    console.print(f"  Status: {status.get('status', 'N/A')}")
    console.print(f"  Mode: {status.get('mode', 'N/A')}")
    console.print(f"  Cached Records: {status.get('records_cached', 'N/A')}")
    console.print(f"  Cache TTL: {status.get('cache_ttl', 'N/A')}")


@app.command()
def fbref_shooting(team_name: str, season: int = 2024):
    """Get team shooting statistics."""
    from footy.fbref import get_fbref_provider

    provider = get_fbref_provider()
    stats = provider.get_team_shooting_stats(team_name, season)

    if not stats:
        console.print(f"[yellow]No shooting data for {team_name} ({season})[/yellow]")
        return

    console.print(f"\n[cyan]{team_name} Shooting Statistics ({season})[/cyan]")
    console.print(f"  Shots: {stats['shots_total']:.1f}")
    console.print(f"  Shots on Target: {stats['shots_on_target']:.1f}")
    console.print(f"  Conversion: {stats['conversion']:.1%}")
    console.print(f"  xG: {stats['xg']:.2f}")
    console.print(f"  NPxG: {stats['npxg']:.2f}")
    console.print(f"  Shots per 90: {stats['shots_per_90']:.1f}")
    console.print(f"  xG per Shot: {stats['xg_per_shot']:.3f}")


@app.command()
def fbref_possession(team_name: str, season: int = 2024):
    """Get team possession statistics."""
    from footy.fbref import get_fbref_provider

    provider = get_fbref_provider()
    stats = provider.get_team_possession_stats(team_name, season)

    if not stats:
        console.print(f"[yellow]No possession data for {team_name} ({season})[/yellow]")
        return

    console.print(f"\n[cyan]{team_name} Possession Statistics ({season})[/cyan]")
    console.print(f"  Possession: {stats['possession_pct']:.1f}%")
    console.print(f"  Touches: {stats['touches']:.0f}")
    console.print(f"  Passes: {stats['passes']:.0f}")
    console.print(f"  Pass Completion: {stats['pass_completion']:.1%}")
    console.print(f"  Avg Pass Distance: {stats['pass_distance_avg']:.1f} yards")
    console.print(f"  Progressive Passes: {stats['progressive_passes']:.1f}")


@app.command()
def fbref_defense(team_name: str, season: int = 2024):
    """Get team defense statistics."""
    from footy.fbref import get_fbref_provider

    provider = get_fbref_provider()
    stats = provider.get_team_defense_stats(team_name, season)

    if not stats:
        console.print(f"[yellow]No defense data for {team_name} ({season})[/yellow]")
        return

    console.print(f"\n[cyan]{team_name} Defense Statistics ({season})[/cyan]")
    console.print(f"  Tackles: {stats['tackles']:.1f}")
    console.print(f"  Interceptions: {stats['interceptions']:.1f}")
    console.print(f"  Blocks: {stats['blocks']:.1f}")
    console.print(f"  Clearances: {stats['clearances']:.1f}")
    console.print(f"  Aerial Duels Won: {stats['aerial_duels_won']:.1f}")
    console.print(f"  Aerial Duel Success: {stats['aerial_duel_success_pct']:.1%}")
    console.print(f"  Fouls Committed: {stats['fouls_committed']:.1f}")
    console.print(f"  Fouls Drawn: {stats['fouls_drawn']:.1f}")


@app.command()
def fbref_passing(team_name: str, season: int = 2024):
    """Get team passing statistics."""
    from footy.fbref import get_fbref_provider

    provider = get_fbref_provider()
    stats = provider.get_team_passing_stats(team_name, season)

    if not stats:
        console.print(f"[yellow]No passing data for {team_name} ({season})[/yellow]")
        return

    console.print(f"\n[cyan]{team_name} Passing Statistics ({season})[/cyan]")
    console.print(f"  Short Pass Completion: {stats['pass_completion_short']:.1%}")
    console.print(f"  Medium Pass Completion: {stats['pass_completion_medium']:.1%}")
    console.print(f"  Long Pass Completion: {stats['pass_completion_long']:.1%}")
    console.print(f"  Key Passes: {stats['key_passes']:.1f}")
    console.print(f"  Passes into Penalty: {stats['passes_into_penalty']:.1f}")
    console.print(f"  Crosses: {stats['crosses']:.1f}")
    console.print(f"  Cross Completion: {stats['cross_completion']:.1%}")
    console.print(f"  Through Balls: {stats['through_balls']:.1f}")


@app.command()
def fbref_compare(team1: str, team2: str, season: int = 2024):
    """Compare FBRef statistics between two teams."""
    from footy.fbref import get_fbref_provider

    provider = get_fbref_provider()
    comparison = provider.compute_team_stats_comparison(team1, team2, season)

    console.print(f"\n[cyan]{team1} vs {team2} - Statistical Comparison ({season})[/cyan]")

    for section in ("shooting_advantage", "possession_advantage", "defense_advantage"):
        label = section.replace("_", " ").title()
        console.print(f"\n[yellow]{label} ({team1} vs {team2}):[/yellow]")
        for key, val in comparison[section].items():
            emoji = "📈" if val > 0 else "📉" if val < 0 else "➡️ "
            console.print(f"  {emoji} {key}: {val:+.1f}%")


@app.command()
def fbref_all(team_name: str, season: int = 2024):
    """Get complete FBRef statistics for a team."""
    from footy.fbref import get_fbref_provider

    provider = get_fbref_provider()
    stats = provider.get_all_team_stats(team_name, season)

    console.print(f"\n[cyan]Complete FBRef Statistics - {team_name} ({season})[/cyan]")
    for section in ("shooting", "possession", "defense", "passing"):
        console.print(f"\n[yellow]{section.title()}:[/yellow]")
        for key, val in stats[section].items():
            if key not in ('team_id', 'season'):
                console.print(f"  {key}: {val}")
