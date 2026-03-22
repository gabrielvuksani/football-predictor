"""
Footy Predictor v12 — Modular API Routes

Shared utilities and router registration for the FastAPI application.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import duckdb

from footy.config import settings as get_settings
from footy.normalize import canonical_team_name

log = logging.getLogger("footy.api")
settings = get_settings()

# Validated sets used across multiple routers
VALID_MODELS = frozenset({"v12_analyst", "v10_council"})
VALID_COMPETITIONS = frozenset({
    "PL", "PD", "SA", "BL1", "FL1", "PT1", "TL1", "DED",
    "ISL", "AUS", "ELC", "PPL", "BEL", "SL", "SWS", "DK1", "SE1", "NO1", "PL1",
})


def con() -> duckdb.DuckDBPyConnection:
    """Open a short-lived read-only DuckDB connection.

    Each API call gets its own connection that is released as soon as
    the caller's reference goes out of scope (CPython ref-counting).
    """
    for attempt in range(3):
        try:
            return duckdb.connect(settings.db_path, read_only=True)
        except duckdb.IOException:
            if attempt < 2:
                time.sleep(0.5)
    return duckdb.connect(settings.db_path, read_only=True)


def safe_error(e: Exception, context: str = "request"):
    """Return a safe JSON error response without leaking internals."""
    from fastapi.responses import JSONResponse

    log.exception("Error handling %s", context)
    return JSONResponse({"error": f"Failed to process {context}. Please try again."}, 500)


def validate_model(model: str) -> str:
    """Validate and return a safe model version string."""
    return model if model in VALID_MODELS else "v12_analyst"


def validate_competition(competition: str) -> str | None:
    """Validate competition code. Returns None if invalid."""
    return competition if competition in VALID_COMPETITIONS else None


def parse_notes(raw_notes) -> dict:
    """Parse notes field from predictions table into a dict."""
    if not raw_notes:
        return {}
    if isinstance(raw_notes, dict):
        return raw_notes
    if isinstance(raw_notes, str):
        try:
            data = json.loads(raw_notes)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def db_or_503():
    """Get a DB connection or raise 503."""
    from fastapi.responses import JSONResponse

    try:
        return con(), None
    except duckdb.IOException:
        return None, JSONResponse({"error": "Database busy"}, 503)
