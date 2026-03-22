"""Shared CLI utilities: console, logging setup, lazy import factories."""
from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
        force=True,
    )


# ---------------------------------------------------------------------------
# Lazy imports — keep `footy --help` fast
# ---------------------------------------------------------------------------

def _pipeline():
    from footy import pipeline
    return pipeline


def _council():
    from footy.models.council import train_and_save, predict_upcoming
    return train_and_save, predict_upcoming


def _odds():
    from footy.fixtures_odds import ingest_upcoming_odds
    return ingest_upcoming_odds


def _extras():
    from footy.extras import ingest_extras_fdcuk
    return ingest_extras_fdcuk
