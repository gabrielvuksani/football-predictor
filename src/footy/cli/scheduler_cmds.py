"""Scheduler sub-commands: add, start, stop, list, enable, disable, remove, history, stats."""
from __future__ import annotations

import typer

from footy.cli._shared import console

app = typer.Typer(help="Job scheduling commands.")


@app.command()
def add(
    job_id: str,
    job_type: str,
    cron: str,
    days_back: int = 30,
    days_forward: int = 7,
):
    """Add a new scheduled job.

    Job types: ingest, train_base, train_council, predict, score, retrain, full_refresh

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
        console.print(f"[green]Job created:[/green] {job_id}")
        console.print(f"  Type: {result['job_type']}")
        console.print(f"  Schedule: {result['cron_schedule']}")
        console.print(f"  Enabled: {result['enabled']}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@app.command()
def start():
    """Start the background scheduler."""
    from footy.scheduler import get_scheduler

    scheduler = get_scheduler()
    scheduler.start()
    console.print("[green]Scheduler started[/green]")

    jobs = scheduler.list_jobs()
    console.print(f"[cyan]Active jobs: {len(jobs)}[/cyan]")
    for job in jobs:
        if job["enabled"]:
            status = f"[{job['last_status'] or 'pending'}]"
            console.print(f"  - {job['job_id']:30s} {status:15s} (next: {job['next_run_at']})")


@app.command()
def stop():
    """Stop the background scheduler."""
    from footy.scheduler import get_scheduler

    scheduler = get_scheduler()
    scheduler.stop()
    console.print("[green]Scheduler stopped[/green]")


@app.command("list")
def list_jobs():
    """List all scheduled jobs."""
    from footy.scheduler import get_scheduler
    from rich.table import Table

    scheduler = get_scheduler()
    jobs = scheduler.list_jobs()

    if not jobs:
        console.print("[yellow]No scheduled jobs[/yellow]")
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
            job["job_id"], job["job_type"], job["cron_schedule"],
            "✓" if job["enabled"] else "✗",
            f"[{status_color}]{job['last_status'] or '-'}[/{status_color}]",
            last_run_str, next_run_str,
        )

    console.print(table)


@app.command()
def enable(job_id: str):
    """Enable a scheduled job."""
    from footy.scheduler import get_scheduler

    scheduler = get_scheduler()
    try:
        result = scheduler.enable_job(job_id)
        console.print(f"[green]Job enabled:[/green] {result['enabled']}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@app.command()
def disable(job_id: str):
    """Disable a scheduled job."""
    from footy.scheduler import get_scheduler

    scheduler = get_scheduler()
    result = scheduler.disable_job(job_id)
    console.print(f"[green]Job disabled:[/green] {result['disabled']}")


@app.command()
def remove(job_id: str, confirm: bool = False):
    """Remove a scheduled job."""
    from footy.scheduler import get_scheduler

    if not confirm:
        console.print(f"[yellow]Warning:[/yellow] This will delete job '{job_id}'")
        console.print("         Use --confirm to proceed")
        return

    scheduler = get_scheduler()
    result = scheduler.remove_job(job_id)
    console.print(f"[green]Job removed:[/green] {result['removed']}")


@app.command()
def history(job_id: str, limit: int = 10):
    """Show execution history for a job."""
    from footy.scheduler import get_scheduler
    from rich.table import Table

    scheduler = get_scheduler()
    hist = scheduler.get_job_history(job_id, limit=limit)

    if not hist:
        console.print(f"[yellow]No history for job:[/yellow] {job_id}")
        return

    table = Table(title=f"Job History: {job_id}")
    table.add_column("Started At", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Duration (s)", style="yellow")
    table.add_column("Error / Result", style="white")

    for run in hist:
        error_or_result = run["error"] or (str(run["result"])[:60] if run["result"] else "-")
        status_color = "green" if run["status"] == "SUCCESS" else ("red" if run["status"] == "FAILED" else "yellow")

        table.add_row(
            str(run["started_at"]).split(".")[0],
            f"[{status_color}]{run['status']}[/{status_color}]",
            f"{run['duration_seconds']:.1f}" if run['duration_seconds'] else "-",
            error_or_result,
        )

    console.print(table)


@app.command()
def stats():
    """Show scheduler statistics."""
    from footy.scheduler import get_scheduler

    scheduler = get_scheduler()
    st = scheduler.get_stats()

    console.print("\n[cyan]Scheduler Statistics[/cyan]")
    console.print(f"  Running: {st['scheduler_running']}")
    console.print(f"  Total Jobs: {st['total_jobs']}")
    console.print(f"  Active Jobs: {st['active_jobs']}")

    console.print(f"\n[cyan]Stats by Job Type:[/cyan]")
    for job_type, type_stats in st["stats_by_type"].items():
        console.print(f"  {job_type}:")
        console.print(f"    Total Runs: {type_stats['total_runs']}")
        console.print(f"    Successful: {type_stats['successful']}")
        console.print(f"    Failed: {type_stats['failed']}")
        console.print(f"    Avg Duration: {type_stats['avg_duration_seconds']:.1f}s")
