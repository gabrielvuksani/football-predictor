"""Footy CLI — organised into sub-command groups.

Usage examples:
    footy go                     # full pipeline
    footy refresh                # quick daily update
    footy data ingest            # ingest recent fixtures
    footy model train            # train Elo + Poisson
    footy ai preview --league PL # AI match preview
    footy scheduler start        # start background scheduler
    footy perf summary           # performance overview
    footy pages export           # export static pages for GitHub Pages
"""
from __future__ import annotations

import typer

from footy.cli.pipeline_cmds import app as _pipeline_app
from footy.cli.data_cmds import app as _data_app
from footy.cli.model_cmds import app as _model_app
from footy.cli.ai_cmds import app as _ai_app
from footy.cli.scheduler_cmds import app as _scheduler_app
from footy.cli.perf_cmds import app as _perf_app
from footy.cli.opta_cmds import app as _opta_app
from footy.cli.pages_cmds import app as _pages_app

# Root app — inherits the root-level commands (go, refresh, serve, etc.)
app = typer.Typer(add_completion=False)

# Register root-level commands from pipeline_cmds
# (go, refresh, matchday, nuke, serve, update, self-test)
for cmd in _pipeline_app.registered_commands:
    app.registered_commands.append(cmd)

# Attach command-group Typer aliases to sub-groups
app.add_typer(_data_app, name="data")
app.add_typer(_model_app, name="model")
app.add_typer(_ai_app, name="ai")
app.add_typer(_scheduler_app, name="scheduler")
app.add_typer(_perf_app, name="perf")
app.add_typer(_opta_app, name="opta")
app.add_typer(_pages_app, name="pages")


if __name__ == "__main__":
    app()
