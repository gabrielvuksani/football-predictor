"""Performance + alerts sub-commands: summary, ranking, trend, health, compare, alerts."""
from __future__ import annotations

import typer

from footy.cli._shared import console

app = typer.Typer(help="Performance tracking and degradation alerts.")


# ------------------------------------------------------------------ metrics
@app.command()
def summary():
    """Show performance summary for all models."""
    from footy.performance_tracker import get_performance_tracker
    from rich.table import Table

    tracker = get_performance_tracker()
    sm = tracker.get_summary()

    if not sm:
        console.print("[yellow]No performance data available[/yellow]")
        return

    table = Table(title="Model Performance Summary (180 days)")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", style="magenta")
    table.add_column("Logloss", style="yellow")
    table.add_column("Brier", style="green")
    table.add_column("Predictions", style="white")

    for model, metrics in sorted(sm.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        table.add_row(
            model,
            f"{metrics['accuracy']:.3f}", f"{metrics['logloss']:.3f}",
            f"{metrics['brier']:.3f}", str(metrics['n_predictions']),
        )

    console.print(table)


@app.command()
def improvement(model: str = "v7_council", days: int = 180):
    """Deep error analysis with actionable recommendations for model improvement."""
    from footy.performance_tracker import generate_improvement_report
    report = generate_improvement_report(model, days)
    console.print(report)


@app.command()
def errors(model: str = "v7_council", days: int = 180):
    """Show prediction error patterns: confusion matrix, calibration, market accuracy."""
    from footy.performance_tracker import analyze_prediction_errors
    import json
    analysis = analyze_prediction_errors(model, days)
    if analysis["status"] == "no_data":
        console.print("[yellow]No scored predictions to analyze.[/yellow]")
        return
    console.print(json.dumps(analysis, indent=2, default=str))


@app.command()
def ranking(days: int = 365):
    """Rank all models by accuracy."""
    from footy.performance_tracker import get_performance_tracker
    from rich.table import Table

    tracker = get_performance_tracker()
    rankings = tracker.get_model_rankings(days=days)

    if not rankings:
        console.print("[yellow]No performance data available[/yellow]")
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
            str(rank), model["model_version"],
            f"{model.get('accuracy', 0):.3f}", f"{model.get('logloss', 0):.3f}",
            str(model.get("n_predictions", 0)),
        )

    console.print(table)


@app.command()
def trend(model_version: str, window_days: int = 30):
    """Show performance trend for a model."""
    from footy.performance_tracker import get_performance_tracker

    tracker = get_performance_tracker()
    tr = tracker.get_performance_trend(model_version, window_days=window_days)

    if not tr:
        console.print(f"[yellow]Insufficient data for {model_version}[/yellow]")
        return

    console.print(f"\n[cyan]Performance Trend: {model_version}[/cyan]")
    console.print(f"  Current Accuracy: {tr['current_accuracy']:.3f}")
    console.print(f"  Average Accuracy: {tr['avg_accuracy']:.3f}")
    console.print(f"  Trend Slope: {tr['trend_slope']:.4f}")

    if tr["degrading"]:
        console.print(f"  [red]⚠️  Performance degrading[/red]")
    else:
        console.print(f"  [green]✓ Performance stable[/green]")

    console.print(f"  R-squared: {tr['r_squared']:.3f}")
    console.print(f"  Data Points: {tr['n_data_points']}")


@app.command()
def daily(model_version: str, days: int = 30):
    """Show daily performance breakdown."""
    from footy.performance_tracker import get_performance_tracker
    from rich.table import Table

    tracker = get_performance_tracker()
    dl = tracker.get_daily_performance(model_version, days=days)

    if not dl:
        console.print(f"[yellow]No data for {model_version}[/yellow]")
        return

    table = Table(title=f"Daily Performance: {model_version}")
    table.add_column("Date", style="cyan")
    table.add_column("Predictions", style="magenta")
    table.add_column("Accuracy", style="yellow")
    table.add_column("Logloss", style="green")
    table.add_column("Brier", style="white")

    for run in dl[:14]:
        table.add_row(
            str(run["date"]), str(run["n_predictions"]),
            f"{run['accuracy']:.3f}", f"{run['logloss']:.3f}", f"{run['brier']:.3f}",
        )

    console.print(table)


@app.command()
def health(model_version: str, days: int = 30):
    """Check model performance health against thresholds."""
    from footy.performance_tracker import get_performance_tracker

    tracker = get_performance_tracker()
    h = tracker.check_performance_health(model_version, days=days)

    status_color = "green" if h.get("status") == "healthy" else "red"

    console.print(f"\n[cyan]Performance Health: {model_version}[/cyan]")
    console.print(f"  Status: [{status_color}]{h.get('status')}[/{status_color}]")

    if h.get("status") in ["healthy", "degraded"]:
        metrics = h.get("metrics", {})
        console.print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
        console.print(f"  Logloss: {metrics.get('logloss', 0):.3f}")
        console.print(f"  Brier: {metrics.get('brier', 0):.3f}")
        console.print(f"  Predictions: {metrics.get('n_predictions', 0)}")

        if h.get("alerts"):
            console.print(f"\n  [red]Alerts:[/red]")
            for alert in h["alerts"]:
                console.print(f"    - {alert}")


@app.command()
def compare(*models: str, days: int = 365):
    """Compare performance across multiple models."""
    from footy.performance_tracker import get_performance_tracker
    from rich.table import Table

    if not models:
        console.print("[red]Error: Specify at least one model[/red]")
        console.print("  Usage: footy perf compare v7_council --days 365")
        return

    tracker = get_performance_tracker()
    comp = tracker.compare_models(list(models), days=days)

    table = Table(title=f"Model Comparison ({days} days)")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", style="magenta")
    table.add_column("Logloss", style="yellow")
    table.add_column("Brier", style="green")
    table.add_column("Home", style="white")
    table.add_column("Draw", style="white")
    table.add_column("Away", style="white")

    for model in comp["models"]:
        if model.get("n_predictions", 0) == 0:
            continue
        table.add_row(
            model["model_version"],
            f"{model.get('accuracy', 0):.3f}", f"{model.get('logloss', 0):.3f}",
            f"{model.get('brier', 0):.3f}", f"{model.get('home_accuracy', 0):.3f}",
            f"{model.get('draw_accuracy', 0):.3f}", f"{model.get('away_accuracy', 0):.3f}",
        )

    console.print(table)


@app.command()
def thresholds(model_version: str, min_accuracy: float = 0.45):
    """Set performance thresholds for a model."""
    from footy.performance_tracker import get_performance_tracker

    tracker = get_performance_tracker()
    result = tracker.set_performance_thresholds(
        model_version=model_version, min_accuracy=min_accuracy,
        max_logloss=0.65, max_brier=0.25, alert_threshold=-0.05,
    )

    console.print(f"[green]Thresholds configured for {result['model_version']}[/green]")
    console.print(f"  Min Accuracy: {result['min_accuracy']:.3f}")
    console.print(f"  Max Logloss: {result['max_logloss']:.3f}")
    console.print(f"  Max Brier: {result['max_brier']:.3f}")


# ------------------------------------------------------------------ alerts
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

    console.print(f"[green]Monitoring configured for {result['model_version']}[/green]")
    console.print(f"  Accuracy Threshold: {result['accuracy_threshold']:.3f}")
    console.print(f"  Logloss Threshold: {result['logloss_threshold']:.3f}")


@app.command()
def alerts_check():
    """Check all models for degradation."""
    from footy.degradation_alerts import get_degradation_monitor

    monitor = get_degradation_monitor()
    alerts = monitor.check_degradation()

    if not alerts:
        console.print("[green]✓ No degradation detected[/green]")
        return

    console.print(f"[red]⚠️  {len(alerts)} alert(s) triggered[/red]")
    for alert in alerts:
        severity_color = "red" if alert.severity.value == "critical" else "yellow"
        console.print(f"  [{severity_color}]{alert.severity.value}[/{severity_color}] {alert.model_version}: {alert.message}")


@app.command("alerts-list")
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
        console.print(f"[yellow]No alerts{f' for {model_version}' if model_version else ''}[/yellow]")
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
            alert.get("alert_id", "")[:30], alert.get("model_version", ""),
            f"[{severity_color}]{alert.get('severity', 'unknown')}[/{severity_color}]",
            alert.get("metric", ""), alert.get("message", "")[:40],
            str(alert.get("created_at", "")).split(".")[0],
        )

    console.print(table)


@app.command()
def alerts_summary():
    """Show alert summary."""
    from footy.degradation_alerts import get_degradation_monitor

    monitor = get_degradation_monitor()
    sm = monitor.get_alert_summary()

    console.print("\n[cyan]Alert Summary[/cyan]")
    console.print(f"  Total Alerts: {sm['total_alerts']}")

    console.print(f"\n  By Status:")
    for status, count in sm["by_status"].items():
        console.print(f"    {status:15s}: {count}")

    if sm["active_by_model"]:
        console.print(f"\n  Active Alerts by Model:")
        for model, count in sm["active_by_model"].items():
            console.print(f"    {model:30s}: {count}")


@app.command()
def alerts_ack(alert_id: str):
    """Acknowledge an alert."""
    from footy.degradation_alerts import get_degradation_monitor

    monitor = get_degradation_monitor()
    result = monitor.acknowledge_alert(alert_id)
    console.print(f"[green]Alert acknowledged:[/green] {result['alert_id']}")


@app.command()
def alerts_resolve(alert_id: str):
    """Resolve an alert."""
    from footy.degradation_alerts import get_degradation_monitor

    monitor = get_degradation_monitor()
    result = monitor.resolve_alert(alert_id)
    console.print(f"[green]Alert resolved:[/green] {result['alert_id']}")


@app.command()
def alerts_snooze(alert_id: str, hours: int = 24):
    """Snooze an alert."""
    from footy.degradation_alerts import get_degradation_monitor

    monitor = get_degradation_monitor()
    result = monitor.snooze_alert(alert_id, hours=hours)
    console.print(f"[green]Alert snoozed until:[/green] {result['until']}")
