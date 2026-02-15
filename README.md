# Footy Predictor

ML-powered football match prediction system with a 6-expert AI council, FastAPI backend, and modern dark-mode frontend.

## Leagues Tracked

| Code | League | Country |
|------|--------|---------|
| PL   | Premier League | England |
| PD   | La Liga | Spain |
| SA   | Serie A | Italy |
| BL1  | Bundesliga | Germany |

## Quick Start

```bash
# 1. Setup
cp .env.example .env          # Fill in API keys
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Full pipeline (ingest → train → predict → H2H → xG)
footy go

# 3. Launch the web UI
footy serve                   # http://localhost:8000

# 4. Daily refresh (quick: ingest recent + retrain + predict)
footy refresh

# 5. Weekend preview (AI analysis of all upcoming matches)
footy matchday
```

## Master Commands

These are one-shot commands that run full workflows:

| Command | What It Does | When To Use |
|---------|-------------|-------------|
| `footy go` | Full pipeline: history → ingest → extras → odds → train Elo/Poisson → train council → predict → H2H → xG | First-time setup or full rebuild |
| `footy go --skip-history` | Same but skips 8-season history download | When history already loaded |
| `footy refresh` | Quick daily update: ingest recent → extras → odds → retrain council → predict → H2H | Daily cron job or manual refresh |
| `footy matchday` | Weekend preview: refresh + AI preview for all leagues | Before a matchday |
| `footy nuke` | Full rebuild: reset all states → go | When something is broken |
| `footy serve` | Start FastAPI web server on port 8000 | To view predictions in browser |
| `footy update` | Legacy: ingest → train base → predict v1 → metrics | Kept for backwards compatibility |

## Architecture

### Expert Council (v7) — Primary Model

```
Layer 1 — SIX SPECIALIST EXPERTS
┌──────────────┬──────────────┬──────────────┐
│  EloExpert   │ MarketExpert │  FormExpert  │
│ Team-specific│ Multi-tier   │ Opposition-  │
│ home adv,    │ odds, line   │ adjusted PPG,│
│ momentum,    │ movement,    │ BTTS, CS,    │
│ volatility   │ O/U, source  │ streaks,     │
│              │ quality      │ shot-on-tgt  │
├──────────────┼──────────────┼──────────────┤
│PoissonExpert │  H2HExpert   │ContextExpert │
│ Venue-split  │ Bayesian     │ Season stage,│
│ attack/def,  │ Dirichlet    │ day-of-week, │
│ BTTS, O2.5,  │ prior, time- │ congestion,  │
│ score matrix,│ decayed, eff │ rest ratio,  │
│ goal skew    │ sample size  │ short-rest   │
└──────────────┴──────────────┴──────────────┘
                     ↓
Layer 2 — CONFLICT & CONSENSUS SIGNALS
  Expert variance, spread, 5 pairwise agreements,
  max disagreement, winner vote concentration,
  confidence-weighted ensemble, entropy
                     ↓
Layer 3 — META-LEARNER
  HistGradientBoosting (lr=0.02, depth=5, 1800 iter)
  + Isotonic calibration (cv=5)
  + Dixon-Coles pseudo-expert (per-league)
  = 140 features → P(Home, Draw, Away)
```

### v7 Performance

| Metric | Value |
|--------|-------|
| Log Loss | **0.623** |
| Brier Score | **0.343** |
| Accuracy | **76.1%** |
| ECE (calibration) | **0.103** |
| Features | **140** |
| Train/Test | 14,547 / 3,265 |

### All Models (historical)

| Model | Role | Status |
|-------|------|--------|
| v1_elo_poisson | Elo + Poisson blend | Base layer |
| v2_meta_stack | LogReg stacker | Base layer |
| v3_gbdt_form | GBDT over rolling form | Base layer |
| v4_super_ensemble | Calibrated ensemble | Retired (v5 subsumes) |
| v5_ultimate | GBDT over all signals | Superseded by v7 |
| **v7_council** | **Expert Council** | **Primary — 6 experts + meta-learner** |

## CLI Commands — Complete Reference

### Master Commands (workflows)

```bash
footy go                          # Full pipeline: history → train → predict
footy go --skip-history           # Skip 8-season download
footy refresh                     # Quick daily: ingest → retrain → predict
footy matchday                    # Refresh + AI preview all leagues
footy nuke                        # Reset everything and rebuild from scratch
footy serve                       # Start web UI (port 8000)
footy serve --port 8080           # Custom port
footy update                      # Legacy: ingest → train base → predict v1
```

### Data Ingestion

```bash
footy ingest                      # Fetch fixtures (30d back, 7d forward)
footy ingest --days-back 365      # Wider window
footy ingest-history              # Pull 8 seasons from football-data.co.uk
footy ingest-history --n-seasons 25  # All available seasons
footy ingest-extras               # Odds + match stats (shots, corners, cards)
footy ingest-extras --n-seasons 8
footy ingest-fixtures-odds        # Attach odds to upcoming matches
footy ingest-af                   # API-Football context (lineups, injuries)
footy ingest-af --lookahead-days 14
footy update-odds                 # External odds + model fallback
footy news                        # GDELT headlines for teams
footy news --days-back 7
```

### Training

```bash
footy train                       # Elo + Poisson (base models)
footy train-meta                  # v2 LogReg stacker (legacy)
# v7 council is trained inside `footy go` and `footy refresh`
```

### Prediction

```bash
footy predict                     # v1 + v2 predictions (legacy)
# v7 council predictions run inside `footy go` and `footy refresh`
```

### Analysis & Metrics

```bash
footy metrics                     # Backtest metrics on finished predictions
footy compute-h2h                 # Recompute all H2H stats
footy compute-xg                  # Backfill xG for finished matches
footy performance-summary         # All model comparison
footy performance-ranking         # Models ranked by accuracy
footy performance-trend MODEL     # Logloss trend over time
footy performance-daily MODEL     # Daily accuracy breakdown
footy performance-health MODEL    # Model health check
footy performance-compare M1 M2   # Side-by-side comparison
footy performance-thresholds MODEL # Accuracy threshold alerting
footy drift-check                 # Check for prediction accuracy drift
footy backtest                    # Time-split backtest
```

### AI / Ollama

```bash
footy ai-preview                  # AI preview for all leagues
footy ai-preview --league PL      # Single league
footy ai-preview --match-id 12345 # Single match
footy ai-value                    # Value bet scanner (model vs market)
footy ai-value --min-edge 0.10    # Higher edge threshold
footy ai-review                   # Post-match accuracy review
footy ai-review --days 7          # Wider review window
footy extract-news --team Arsenal # Team news from GDELT → LLM
footy analyze-form --team Arsenal # LLM form analysis
footy explain-match --home-team X --away-team Y  # AI explanation
footy insights-status             # Check Ollama health
```

### Retraining System

```bash
footy retrain                     # Auto-retrain: check → train → validate → deploy
footy retrain --force             # Force retrain
footy retraining-status           # Show readiness
footy retraining-setup            # Configure thresholds
footy retraining-history MODEL    # Training audit trail
footy retraining-deploy V T       # Deploy specific version
footy retraining-rollback T       # Rollback model type
footy retraining-deployments      # Active deployments
footy retraining-record V T       # Record a training run
```

### Scheduler

```bash
footy scheduler-start             # Start background job scheduler
footy scheduler-stop              # Stop scheduler
footy scheduler-list              # List all jobs
footy scheduler-add ID TYPE CRON  # Add job (ingest, train_base, predict, etc.)
footy scheduler-enable ID         # Enable a job
footy scheduler-disable ID        # Disable a job
footy scheduler-remove ID --confirm  # Delete a job
footy scheduler-history ID        # Execution history
footy scheduler-stats             # Scheduler statistics
```

### Alerts & Monitoring

```bash
footy alerts-setup                # Configure degradation alerts
footy alerts-check                # Run alert checks now
footy alerts-list                 # List active alerts
footy alerts-summary              # Alert summary
footy alerts-acknowledge ID       # Acknowledge an alert
footy alerts-resolve ID           # Mark alert resolved
footy alerts-snooze ID            # Snooze alert (default 24h)
```

### Advanced Stats Providers

```bash
# FBRef
footy fbref-status                # Provider status
footy fbref-shooting TEAM         # Shooting stats
footy fbref-possession TEAM       # Possession stats
footy fbref-defense TEAM          # Defensive stats
footy fbref-passing TEAM          # Passing stats
footy fbref-compare T1 T2         # Head-to-head stats comparison
footy fbref-all TEAM              # All stats for a team

# Understat
footy understat-status            # Provider status
footy understat-team TEAM         # Team xG stats
footy understat-match ID          # Match xG breakdown
footy understat-team-rolling TEAM # Rolling xG averages
```

### Maintenance

```bash
footy reset-states                # Clear all model state (full rebuild)
footy cache-stats                 # Show cache usage
footy cache-cleanup               # Remove expired cache entries
footy cache-cleanup --full        # Clear entire cache
```

### Testing

```bash
footy self-test                   # Run full test suite (unit + integration)
footy self-test --smoke           # Include live-database smoke tests
footy self-test --cov             # With code coverage report
footy self-test -v                # Verbose output
footy self-test --fast            # Skip slow tests
footy self-test -k test_models    # Run only matching tests
```

## Testing

The project includes a comprehensive self-testing workflow powered by pytest.

### Test Suite Structure

| File | Tests | Scope |
|------|-------|-------|
| `test_config.py` | Settings loading, defaults, validation | Unit |
| `test_db.py` | Schema creation, PKs, seeded data integrity | Unit |
| `test_normalize.py` | Team name normalization, aliases, edge cases | Unit |
| `test_models_elo.py` | Elo calculations, DB ops, predict_probs | Unit |
| `test_models_poisson.py` | Poisson fitting, expected goals, outcomes | Unit |
| `test_models_dc.py` | Dixon-Coles fitting, tau function, 1×2 | Unit |
| `test_models_council.py` | Expert Council helpers, ExpertResult, constants | Unit |
| `test_h2h.py` | H2H table creation, recompute, lookups | Integration |
| `test_xg.py` | xG computation, table creation, backfill | Integration |
| `test_cli.py` | All CLI commands respond to `--help` | Integration |
| `test_api.py` | All FastAPI endpoints (TestClient) | Integration |
| `test_smoke.py` | Live database health checks (read-only) | Smoke |

### Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Quick check (116 unit + integration tests, ~4s)
footy self-test

# Full suite including live-DB smoke tests
footy self-test --smoke

# With coverage report
footy self-test --smoke --cov

# Or use pytest directly
pytest tests/ -v
pytest tests/test_models_elo.py -v        # Single file
pytest tests/ -k "poisson or elo" -v      # Pattern matching
pytest tests/ --cov=footy --cov-report=html  # HTML coverage report
```

## API Endpoints

The FastAPI backend serves at `http://localhost:8000`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI (single-page app) |
| GET | `/api/matches?days=14` | Upcoming matches with predictions |
| GET | `/api/matches/{id}` | Match detail (prediction, odds, Elo) |
| GET | `/api/matches/{id}/experts` | Expert council breakdown (6 experts) |
| GET | `/api/matches/{id}/h2h` | Head-to-head history |
| GET | `/api/matches/{id}/form` | Recent form (W/D/L streak + PPG) |
| GET | `/api/matches/{id}/ai` | AI narrative (requires Ollama) |
| GET | `/api/insights/value-bets` | Value bets with Kelly criterion |
| GET | `/api/stats` | Database statistics |

## Data Sources

| Source | What | API Key |
|--------|------|---------|
| football-data.org | Live fixtures, results, status | Yes (`FOOTBALL_DATA_ORG_TOKEN`) |
| football-data.co.uk | Historical results + odds + stats (25 seasons) | No |
| GDELT | Team news headlines | No |
| Ollama | Local LLM for AI analysis | No (local) |
| API-Football | Lineups, injuries, detail | Yes (`API_FOOTBALL_KEY`) |

## Environment Variables

```env
# Required
FOOTBALL_DATA_ORG_TOKEN=your_token
API_FOOTBALL_KEY=your_key

# Optional
THE_ODDS_API_KEY=
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Config
TRACKED_COMPETITIONS=PL,PD,SA,BL1
LOOKAHEAD_DAYS=7
DB_PATH=./data/footy.duckdb
```

## Database

DuckDB at `data/footy.duckdb`:

| Table | Rows | Description |
|-------|------|-------------|
| matches | ~38,300 | All fixtures (finished + upcoming, 25 seasons) |
| match_extras | ~17,700 | Odds + stats (shots, corners, cards) |
| predictions | varies | Model predictions per match per version |
| elo_state | ~174 | Current Elo rating per team |
| poisson_state | 1 | Fitted Poisson parameters |
| h2h_stats | ~2,822 | Head-to-head stats (any venue) |
| h2h_venue_stats | ~5,577 | Head-to-head stats (specific venue) |

## Docker

```bash
docker build -t footy .
docker run -p 8000:8000 --env-file .env footy

# Or with docker-compose:
docker-compose up -d
```

## Project Structure

```
football-predictor/
├── README.md                 # This file
├── MASTER_PLAN.md            # Development history & roadmap
├── pyproject.toml            # Python package config
├── bootstrap.sh              # Initial setup script
├── Dockerfile                # Container image
├── docker-compose.yml        # Container orchestration
├── data/
│   ├── footy.duckdb          # Main database
│   └── models/
│       └── v7_council.joblib # Primary model (Expert Council)
├── tests/
│   ├── conftest.py           # Shared fixtures (in-memory DB, seeded data)
│   ├── test_config.py        # Settings tests
│   ├── test_db.py            # Schema & data integrity tests
│   ├── test_normalize.py     # Team name normalization tests
│   ├── test_models_elo.py    # Elo rating model tests
│   ├── test_models_poisson.py # Poisson model tests
│   ├── test_models_dc.py     # Dixon-Coles model tests
│   ├── test_models_council.py # Expert Council tests
│   ├── test_h2h.py           # H2H statistics tests
│   ├── test_xg.py            # xG computation tests
│   ├── test_cli.py           # CLI --help smoke tests
│   ├── test_api.py           # FastAPI endpoint tests
│   └── test_smoke.py         # Live database health checks
├── src/footy/
│   ├── cli.py                # All CLI commands (Typer)
│   ├── config.py             # Settings from .env
│   ├── db.py                 # DuckDB connection + schema
│   ├── pipeline.py           # Core pipeline logic
│   ├── normalize.py          # Team name normalization
│   ├── team_mapping.py       # Canonical team name mappings
│   ├── extras.py             # Match extras ingestion
│   ├── fixtures_odds.py      # Odds for upcoming matches
│   ├── h2h.py                # Head-to-head computation
│   ├── xg.py                 # Expected goals computation
│   ├── cache.py              # Prediction/insight caching
│   ├── fbref.py              # FBRef stats provider
│   ├── understat.py          # Understat stats provider
│   ├── continuous_training.py # Auto-retrain management
│   ├── scheduler.py          # Background job scheduler
│   ├── degradation_alerts.py # Model drift detection
│   ├── performance_tracker.py # Model performance tracking
│   ├── models/
│   │   ├── council.py        # v7 Expert Council (primary)
│   │   ├── elo.py            # Dynamic Elo ratings
│   │   ├── poisson.py        # Weighted Poisson model
│   │   ├── dixon_coles.py    # Dixon-Coles correction
│   │   ├── meta.py           # v2 LogReg stacker (legacy)
│   │   ├── v3.py             # v3 GBDT form model (legacy)
│   │   └── v5.py             # v5 ultimate model (legacy)
│   ├── providers/
│   │   ├── football_data_org.py
│   │   ├── fdcuk_history.py
│   │   ├── fdcuk_fixtures.py
│   │   ├── api_football.py
│   │   ├── news_gdelt.py
│   │   ├── odds_scraper.py
│   │   └── ratelimit.py
│   └── llm/
│       ├── ollama_client.py
│       ├── news_extractor.py
│       └── insights.py
└── web/
    ├── api.py                # FastAPI backend (9 endpoints)
    ├── static/
    │   ├── style.css         # Dark glassmorphism CSS
    │   └── app.js            # Alpine.js frontend logic
    └── templates/
        └── index.html        # Single-page app template
```

## Accessing from Other Devices

The server binds to `0.0.0.0:8000`, accessible from any device on the network:

```
http://<server-ip>:8000
```

Works on iOS Safari, Android Chrome, any modern browser.
