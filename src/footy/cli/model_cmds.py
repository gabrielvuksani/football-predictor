"""Model sub-commands: train, predict, backtest, retrain, drift, retraining ops."""
from __future__ import annotations

import time
import typer

from footy.cli._shared import console, _pipeline, _council

app = typer.Typer(help="Model training, prediction, and retraining commands.")


@app.command()
def train():
    """Train Elo + Poisson models from finished matches."""
    t0 = time.perf_counter()
    n = _pipeline().update_elo_from_finished(verbose=True)
    console.print(f"[green]Elo applied[/green] {n} new finished matches")
    state = _pipeline().refit_poisson(verbose=True)
    dt = time.perf_counter() - t0
    console.print(f"[green]Train done[/green] teams={len(state.get('teams', []))} in {dt:.1f}s")


@app.command()
def predict():
    """Generate predictions for upcoming matches using the v10 council."""
    t0 = time.perf_counter()
    _, predict_upcoming = _council()
    from footy.db import connect
    n = predict_upcoming(connect(), verbose=True)
    dt = time.perf_counter() - t0
    console.print(f"[green]Predictions written[/green] {n} upcoming matches in {dt:.1f}s")


@app.command()
def metrics():
    """Show scored prediction metrics."""
    m = _pipeline().score_finished_predictions(verbose=True)
    console.print(m)


@app.command()
def backtest(days: int = 365, test_days: int = 28):
    """Walk-forward backtest using the v10 council model."""
    from footy.db import connect
    from footy.walkforward import walk_forward_cv
    con = connect()
    results = walk_forward_cv(con, n_splits=4, verbose=True)
    console.print(results)


@app.command()
def train_meta(days: int = 365, test_days: int = 28):
    """Train the v10 council model (supersedes old meta-learner)."""
    train_and_save, _ = _council()
    from footy.db import connect
    r = train_and_save(connect(), days=days, verbose=True)
    console.print(r)


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
    try:
        manager.setup_continuous_training("v10_council", 20, 0.005)
    except Exception:
        pass

    result = manager.auto_retrain(force=force, verbose=True)
    action = result.get("action", "unknown")

    if action == "none":
        check = result.get("check", {})
        console.print(f"[yellow]No retraining needed[/yellow]")
        if check.get("status") == "waiting":
            console.print(f"  New matches: {check.get('new_matches', '?')}/{check.get('threshold', '?')}")
            console.print(f"  Ready in ~{check.get('ready_in', '?')} more finished matches")
    elif action == "deployed":
        console.print(f"[green bold]Model deployed:[/green bold] {result['version']}")
        m = result.get("metrics", {})
        console.print(f"  Accuracy: {m.get('accuracy', 0):.1%}  LogLoss: {m.get('logloss', 0):.4f}")
        console.print(f"  Improvement: {result.get('improvement_pct', 0):+.2f}%")
    elif action == "rolled_back":
        console.print(f"[red]Rolled back[/red] — new model was worse")
        m = result.get("metrics", {})
        console.print(f"  New accuracy: {m.get('accuracy', 0):.1%}  LogLoss: {m.get('logloss', 0):.4f}")
        console.print(f"  Regression: {result.get('improvement_pct', 0):+.2f}%")
    elif action == "train_failed":
        console.print(f"[red]Training failed:[/red] {result.get('error', 'unknown')}")
    else:
        console.print(f"[yellow]Result:[/yellow] {result}")


@app.command()
def drift_check():
    """Check for prediction accuracy drift on recent matches."""
    from footy.continuous_training import get_training_manager

    manager = get_training_manager()
    drift = manager.detect_drift("v10_council")

    if drift.get("reason") == "insufficient_data":
        console.print(f"[yellow]Not enough data for drift detection[/yellow]")
        console.print(f"  Baseline matches: {drift.get('baseline_n', 0)}")
        console.print(f"  Recent matches: {drift.get('recent_n', 0)}")
        return

    if drift.get("drifted"):
        console.print(f"[red bold]DRIFT DETECTED[/red bold]")
    else:
        console.print(f"[green]No drift detected[/green]")

    console.print(f"  Baseline accuracy (>60d ago): {drift.get('baseline_acc', 0):.1%} ({drift.get('baseline_n', 0)} matches)")
    console.print(f"  Recent accuracy  (<60d):      {drift.get('recent_acc', 0):.1%} ({drift.get('recent_n', 0)} matches)")
    console.print(f"  Accuracy change: {-drift.get('accuracy_drop', 0):+.1%}")


@app.command()
def setup(
    model_type: str,
    threshold_matches: int = 10,
    threshold_improvement: float = 0.01,
):
    """Setup continuous retraining for a model type.

    Model types: v10_council (current), v5_ultimate (legacy)
    """
    from footy.continuous_training import get_training_manager

    manager = get_training_manager()
    result = manager.setup_continuous_training(model_type, threshold_matches, threshold_improvement)

    console.print(f"[green]Retraining configured for {result['model_type']}[/green]")
    console.print(f"  Threshold: {result['retrain_threshold_matches']} new matches")
    console.print(f"  Min Improvement: {result['performance_threshold_improvement']:.1%}")


@app.command()
def status():
    """Show retraining status for all models."""
    from footy.continuous_training import get_training_manager
    from rich.table import Table

    manager = get_training_manager()
    status = manager.get_retraining_status()

    if not status:
        console.print("[yellow]No models configured for retraining[/yellow]")
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
            model_type, str(stats["new_matches"]), str(stats["threshold"]),
            status_str, last_trained,
        )

    console.print(table)


@app.command()
def record(
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
        model_version=model_version, model_type=model_type,
        training_window_days=window_days, n_matches_used=n_matches_train,
        n_matches_test=n_matches_test, metrics=metrics, test_metrics=metrics,
    )

    console.print(f"[green]Training recorded for {result['model_version']}[/green]")
    console.print(f"  Model Type: {result['model_type']}")
    console.print(f"  Improvement: {result['improvement_pct']:+.2f}%")
    if result['previous_version']:
        console.print(f"  Previous: {result['previous_version']}")


@app.command()
def deploy(model_version: str, model_type: str, force: bool = False):
    """Deploy a trained model version to production."""
    from footy.continuous_training import get_training_manager

    manager = get_training_manager()
    if not force:
        console.print(f"[yellow]⚠️  Deploying {model_version} to production[/yellow]")
        console.print("    Use --force to confirm")
        return

    result = manager.deploy_model(model_version, model_type, force=force)
    if result.get("status") == "error":
        console.print(f"[red]Error:[/red] {result['error']}")
        return

    console.print(f"[green]Model deployed:[/green] {result['model_version']}")
    console.print(f"  Type: {result['model_type']}")
    console.print(f"  Improvement: {result['improvement_pct']:+.2f}%")
    if result['previous_version']:
        console.print(f"  Previous: {result['previous_version']}")


@app.command()
def rollback(model_type: str, force: bool = False):
    """Rollback to previous model version."""
    from footy.continuous_training import get_training_manager

    manager = get_training_manager()
    if not force:
        console.print(f"[yellow]⚠️  Rolling back {model_type}[/yellow]")
        console.print("    Use --force to confirm")
        return

    result = manager.rollback_model(model_type)
    if result.get("status") == "error":
        console.print(f"[red]Error:[/red] {result['error']}")
        return

    console.print(f"[green]Model rolled back:[/green] {result['model_type']}")
    console.print(f"  Restored: {result['restored_version']}")


@app.command()
def history(model_type: str, limit: int = 10):
    """Show training history for a model type."""
    from footy.continuous_training import get_training_manager
    from rich.table import Table

    manager = get_training_manager()
    hist = manager.get_training_history(model_type, limit=limit)

    if not hist:
        console.print(f"[yellow]No training history for {model_type}[/yellow]")
        return

    table = Table(title=f"Training History: {model_type}")
    table.add_column("Version", style="cyan")
    table.add_column("Training Date", style="magenta")
    table.add_column("Matches", style="yellow")
    table.add_column("Improvement", style="green")
    table.add_column("Deployed", style="white")

    for run in hist:
        improvement_color = "green" if run["improvement_pct"] >= 0 else "red"
        table.add_row(
            run["model_version"],
            str(run["training_date"]).split(".")[0],
            str(run["n_matches_used"]),
            f"[{improvement_color}]{run['improvement_pct']:+.2f}%[/{improvement_color}]",
            "✓" if run["deployed"] else "✗",
        )

    console.print(table)


@app.command()
def deployments():
    """Show current model deployments."""
    from footy.continuous_training import get_training_manager
    from rich.table import Table

    manager = get_training_manager()
    deps = manager.get_deployment_status()

    if not deps:
        console.print("[yellow]No models deployed[/yellow]")
        return

    table = Table(title="Model Deployments")
    table.add_column("Model Type", style="cyan")
    table.add_column("Active Version", style="magenta")
    table.add_column("Deployed At", style="yellow")
    table.add_column("Previous", style="white")

    for model_type, info in deps.items():
        table.add_row(
            model_type, info["active_version"],
            str(info["deployed_at"]).split(".")[0],
            info["previous_version"] or "—",
        )

    console.print(table)
