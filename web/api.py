"""
Footy Predictor v14 — Apex
FastAPI application factory with modular route registration.

Serves:
    - REST JSON API at /api/*
    - Static files (CSS / JS) at /static/*
    - Single-page HTML frontend at /
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

from web.routes import health, matches, insights, stats, system, predictions, advanced

log = logging.getLogger("footy.api")

WEB_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))

app = FastAPI(
    title="Footy Predictor v14 — Apex",
    description="AI-powered football match predictions with 50-expert ensemble",
    version="14.0.0",
)

# ── Middleware ──

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_response_headers(request: Request, call_next):
    """Add timing, cache control, and security headers to all responses."""
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000

    response.headers["X-Response-Time"] = f"{dt:.0f}ms"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

    if request.url.path.startswith("/api/"):
        if request.url.path in ("/api/health", "/api/stats/dashboard"):
            response.headers["Cache-Control"] = "public, max-age=300"
        elif "/matches" in request.url.path:
            response.headers["Cache-Control"] = "public, max-age=300"
        else:
            response.headers["Cache-Control"] = "public, max-age=600"

    log.info("HTTP %s %s %s (%.0fms)", request.method, request.url.path, response.status_code, dt)
    return response


# ── Static files ──

app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

# ── Register API routers ──

app.include_router(health.router)
app.include_router(matches.router)
app.include_router(insights.router)
app.include_router(stats.router)
app.include_router(system.router)
app.include_router(predictions.router)
app.include_router(advanced.router)


# ── Frontend routes ──

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse(request, "index.html")


@app.get("/match/{match_id}", response_class=HTMLResponse)
async def match_page(request: Request, match_id: int):
    return TEMPLATES.TemplateResponse(request, "index.html")
