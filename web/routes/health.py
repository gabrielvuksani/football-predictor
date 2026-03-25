"""Health, status, and data freshness endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from . import con, safe_error

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def api_health():
    """Lightweight health endpoint for uptime checks and monitoring."""
    return {
        "status": "ok",
        "app": "footy-predictor",
        "version": "15.0",
        "model": "v15_architect",
        "experts": 49,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/last-updated")
async def api_last_updated():
    """Get when predictions were last updated."""
    try:
        db = con()
    except Exception:
        return {"last_updated": None}

    try:
        row = db.execute(
            "SELECT MAX(created_at) FROM predictions WHERE model_version IN ('v15_architect','v13_oracle')"
        ).fetchone()
        return {"last_updated": str(row[0])[:19] if row and row[0] else None}
    except Exception:
        return {"last_updated": None}


@router.get("/data-freshness")
async def api_data_freshness():
    """Show when each data source was last updated."""
    try:
        db = con()
        sources = {}

        row = db.execute("SELECT MAX(utc_date), COUNT(*) FROM matches WHERE status='FINISHED'").fetchone()
        sources["matches"] = {"last_updated": str(row[0]) if row[0] else None, "count": int(row[1] or 0)}

        row = db.execute("SELECT MAX(created_at), COUNT(*) FROM predictions WHERE model_version='v13_oracle'").fetchone()
        sources["predictions"] = {"last_updated": str(row[0]) if row[0] else None, "count": int(row[1] or 0)}

        for table, key in [
            ("fbref_team_stats", "fbref"),
            ("market_values", "transfermarkt_values"),
            ("transfermarkt_injuries", "injuries"),
            ("referee_stats", "referee_stats"),
            ("venue_stats", "venue_stats"),
        ]:
            try:
                ts_col = "fetched_at" if table == "weather_data" else "updated_at"
                row = db.execute(f"SELECT MAX({ts_col}), COUNT(*) FROM {table}").fetchone()
                sources[key] = {"last_updated": str(row[0]) if row[0] else None, "count": int(row[1] or 0)}
            except Exception:
                sources[key] = {"last_updated": None, "count": 0}

        try:
            row = db.execute("SELECT MAX(fetched_at), COUNT(*) FROM weather_data").fetchone()
            sources["weather"] = {"last_updated": str(row[0]) if row[0] else None, "count": int(row[1] or 0)}
        except Exception:
            sources["weather"] = {"last_updated": None, "count": 0}

        return {"sources": sources, "model_version": "v13_oracle"}
    except Exception as e:
        return safe_error(e, "data freshness")


@router.get("/sources")
async def api_sources():
    """Get data source status."""
    try:
        db = con()
    except Exception:
        return _fallback_sources()

    try:
        # Query from provider_status using its actual schema
        sources_data = db.execute("""
            SELECT provider, status, fetched_at
            FROM provider_status
            ORDER BY provider
        """).fetchall()

        sources = []
        if sources_data:
            for row in sources_data:
                sources.append({
                    "name": row[0],
                    "active": row[1] == "ok" if isinstance(row[1], str) else bool(row[1]),
                    "last_update": str(row[2])[:19] if row[2] else None,
                })
        else:
            return _fallback_sources()

        return {"sources": sources}
    except Exception:
        return _fallback_sources()


def _fallback_sources():
    """Graceful fallback when provider_status table is unavailable."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "sources": [
            {"name": "football-data.co.uk", "active": True, "last_update": now},
            {"name": "SofaScore", "active": True, "last_update": now},
            {"name": "FBref", "active": True, "last_update": now},
            {"name": "ClubElo", "active": True, "last_update": now},
            {"name": "Open-Meteo", "active": True, "last_update": now},
            {"name": "Transfermarkt", "active": True, "last_update": now},
        ]
    }
