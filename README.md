# Footy Predictor

ML-powered football match prediction system with an 8-expert AI council, FastAPI backend, and dark-mode frontend.

## Leagues Tracked

| Code | League | Country |
|------|--------|---------|
| PL   | Premier League | England |
| PD   | La Liga | Spain |
| SA   | Serie A | Italy |
| BL1  | Bundesliga | Germany |
| FL1  | Ligue 1 | France |

## Quick Start

```bash
# 1. Clone & enter
git clone <repo-url> && cd football-predictor

# 2. One-liner setup (creates venv, installs deps, copies .env)
bash bootstrap.sh

# 3. Add your API keys
nano .env   # set FOOTBALL_DATA_ORG_TOKEN and API_FOOTBALL_KEY

# 4. Activate the environment
source .venv/bin/activate

# 5. Run the full pipeline (ingest 25 seasons → train → predict)
footy go

# 6. Launch the web UI
footy serve                   # http://localhost:8000
```

### Manual Setup (if you prefer)

```bash
cp .env.example .env          # Fill in API keys
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
footy go
footy serve
```

### Docker

```bash
cp .env.example .env          # Fill in API keys
docker-compose up -d          # Builds, runs pipeline, serves UI on :8000
```

## Required API Keys

| Key | Source | Free Tier |
|-----|--------|-----------|
| `FOOTBALL_DATA_ORG_TOKEN` | [football-data.org](https://www.football-data.org/client/register) | Yes — 10 req/min |
| `API_FOOTBALL_KEY` | [api-football.com](https://www.api-football.com/) | Yes — 100 req/day |

Both are free to register. The system works without `API_FOOTBALL_KEY` (you just lose lineup/injury context).

## Commands

Commands are organised into sub-groups. Run `footy --help` or `footy <group> --help` for full details.

### Day-to-Day (root-level)

| Command | What It Does |
|---------|-------------|
| `footy go` | Full pipeline: 25-season history → train Elo/Poisson → train council → predict → H2H → xG |
| `footy go --skip-history` | Same but skips history download (fast daily use) |
| `footy refresh` | Quick update: ingest recent → retrain council → predict |
| `footy matchday` | Refresh + AI preview for all leagues (requires Ollama) |
| `footy update` | Lightweight ingest → train → predict → metrics |
| `footy serve` | Start web UI on port 8000 |
| `footy nuke` | Full reset and rebuild from scratch |
| `footy self-test` | Run the pytest suite |

### `footy data` — Data Ingestion & Management

| Command | Description |
|---------|-------------|
| `footy data ingest` | Fetch fixtures (30d back, 7d forward) |
| `footy data history` | Pull historical seasons from football-data.co.uk |
| `footy data extras` | Odds + match stats (shots, corners, cards) |
| `footy data fixtures-odds` | Attach odds to upcoming matches |
| `footy data odds` | Update odds from external sources + model fallback |
| `footy data api-football` | Fetch lineups, injuries, stats from API-Football |
| `footy data news` | GDELT headlines for upcoming teams |
| `footy data h2h` | Recompute all H2H stats |
| `footy data xg` | Compute xG for finished matches |
| `footy data reset` | Reset all model states |
| `footy data cache-stats` | Show cache usage |
| `footy data cache-cleanup` | Remove expired cache entries |

### `footy model` — Training & Prediction

| Command | Description |
|---------|-------------|
| `footy model train` | Train Elo + Poisson base models |
| `footy model predict` | Generate predictions for upcoming matches |
| `footy model metrics` | Backtest accuracy metrics |
| `footy model backtest` | Walk-forward time-split backtest |
| `footy model train-meta` | Train the meta-learner stacking model |
| `footy model retrain` | Auto-retrain council: check → train → validate → deploy |
| `footy model drift-check` | Check for prediction accuracy drift |
| `footy model deploy` | Deploy a trained model version to production |
| `footy model rollback` | Rollback to previous model version |
| `footy model status` | Retraining status for all models |
| `footy model deployments` | Show current model deployments |

### `footy ai` — AI / Ollama (optional)

| Command | Description |
|---------|-------------|
| `footy ai preview` | AI match preview (single match or full league round) |
| `footy ai value` | Value bet scanner (model edge vs market odds) |
| `footy ai review` | Post-match accuracy review |
| `footy ai extract-news` | Extract + analyse team news via Ollama |
| `footy ai analyze-form` | Analyse recent team form |
| `footy ai explain-match` | LLM explanation for a specific prediction |
| `footy ai status` | Ollama/insights system health check |

### `footy scheduler` — Background Jobs

| Command | Description |
|---------|-------------|
| `footy scheduler add` | Create a scheduled job (cron syntax) |
| `footy scheduler start` | Start the background scheduler |
| `footy scheduler stop` | Stop the scheduler |
| `footy scheduler list` | List all jobs |
| `footy scheduler history` | Execution history for a job |

### `footy perf` — Performance & Alerts

| Command | Description |
|---------|-------------|
| `footy perf summary` | All-model comparison table |
| `footy perf ranking` | Models ranked by accuracy |
| `footy perf trend` | Performance trend for a model |
| `footy perf daily` | Daily breakdown |
| `footy perf health` | Health check against thresholds |
| `footy perf compare` | Side-by-side model comparison |
| `footy perf alerts-check` | Check all models for degradation |
| `footy perf alerts-list` | List active alerts |

### `footy stats` — External Stats (Understat, FBRef)

| Command | Description |
|---------|-------------|
| `footy stats understat-team` | Team xG statistics |
| `footy stats understat-match` | Match xG breakdown |
| `footy stats understat-rolling` | Rolling xG averages |
| `footy stats fbref-shooting` | Team shooting stats |
| `footy stats fbref-possession` | Possession stats |
| `footy stats fbref-defense` | Defense stats |
| `footy stats fbref-passing` | Passing stats |
| `footy stats fbref-compare` | Compare two teams |
| `footy stats fbref-all` | Complete stat dump for a team |

### `footy opta` — Opta Predictions

| Command | Description |
|---------|-------------|
| `footy opta fetch` | Scrape Opta win probabilities (all leagues) |
| `footy opta fetch --league PL` | Single league |
| `footy opta show` | Display cached Opta predictions |

## Architecture

### Expert Council (v7) — Primary Model

```
Layer 1 — SIX SPECIALIST EXPERTS
┌──────────────┬──────────────┬──────────────┐
│  EloExpert   │ MarketExpert │  FormExpert  │
│ Rating       │ Multi-tier   │ Opposition-  │
│ dynamics,    │ odds, line   │ adjusted PPG,│
│ momentum,    │ movement,    │ BTTS, CS,    │
│ home adv     │ sharp money  │ streaks      │
├──────────────┼──────────────┼──────────────┤
│PoissonExpert │  H2HExpert   │ContextExpert │
│ Venue-split  │ Bayesian     │ Season stage,│
│ attack/def,  │ Dirichlet    │ congestion,  │
│ BTTS, O2.5,  │ prior, time  │ rest ratio,  │
│ score matrix │ decay        │ day-of-week  │
└──────────────┴──────────────┴──────────────┘
                     ↓
Layer 2 — CONFLICT & CONSENSUS SIGNALS
  Expert variance, pairwise agreements,
  winner vote concentration, entropy
                     ↓
Layer 3 — META-LEARNER
  HistGradientBoosting (lr=0.02, depth=5, 1800 iter)
  + Isotonic calibration (cv=5)
  + Dixon-Coles pseudo-expert (per-league)
  → ~140 features → P(Home, Draw, Away)
```

## API Endpoints

The FastAPI backend serves at `http://localhost:8000`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI (single-page app) |
| GET | `/api/matches?days=14` | Upcoming matches with predictions |
| GET | `/api/matches/{id}` | Match detail (prediction, odds, Elo) |
| GET | `/api/matches/{id}/experts` | Expert council breakdown |
| GET | `/api/matches/{id}/h2h` | Head-to-head history |
| GET | `/api/matches/{id}/form` | Recent form (W/D/L streak + PPG) |
| GET | `/api/matches/{id}/ai` | AI narrative (requires Ollama) |
| GET | `/api/insights/value-bets` | Value bets with Kelly criterion |
| GET | `/api/stats` | Database statistics |
| GET | `/api/performance` | Model performance + calibration |
| GET | `/api/matches/{id}/xg` | xG breakdown for match |
| GET | `/api/matches/{id}/patterns` | Goal pattern analysis |
| GET | `/api/league-table/{comp}` | Simulated league standings |
| GET | `/api/competitions` | Available competitions |
| GET | `/api/last-updated` | Last prediction timestamp |

## Data Sources

| Source | What | Key Required |
|--------|------|:---:|
| football-data.org | Live fixtures & results | Yes |
| football-data.co.uk | 25 seasons of historical results + odds | No |
| API-Football | Lineups, injuries, pre-match context | Yes (optional) |
| Understat | Expected goals (xG) | No (stub) |
| FBRef | Advanced team stats | No (stub) |
| GDELT | Team news headlines | No |
| Opta / The Analyst | Match win probabilities | No (scraped) |
| Ollama | Local LLM for AI analysis | No (local) |
| The Odds API | Market odds | Optional |

## Environment Variables

```env
# Required
FOOTBALL_DATA_ORG_TOKEN=your_token_here
API_FOOTBALL_KEY=your_key_here          # optional but recommended

# Optional providers
THE_ODDS_API_KEY=                        # market odds
ODD_API_KEY=                             # alternative odds source
SPORTAPI_AI_KEY=                         # SportAPI.ai

# AI
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Config
TRACKED_COMPETITIONS=PL,PD,SA,BL1,FL1
LOOKAHEAD_DAYS=7
DB_PATH=./data/footy.duckdb
```

## Database

DuckDB at `data/footy.duckdb`:

| Table | Description |
|-------|-------------|
| matches | All fixtures (finished + upcoming) |
| match_extras | Odds + stats (shots, corners, cards) |
| match_xg | Expected goals per match |
| predictions | Model predictions per match per version |
| prediction_scores | Scored predictions with accuracy metrics |
| elo_state | Current Elo rating per team |
| elo_applied | Tracks which matches have been applied to Elo |
| poisson_state | Fitted Poisson parameters |
| h2h_stats | Head-to-head stats (any venue) |
| h2h_venue_stats | Head-to-head stats (specific venue) |
| h2h_recent | Recent head-to-head form |
| news | Team news headlines |
| expert_cache | Cached expert council breakdowns |
| llm_insights | LLM-generated match narratives |
| metrics | Model performance metrics over time |
| opta_predictions | Scraped Opta win probabilities |
| team_mappings | Cross-provider team name mappings |

## Project Structure

```
football-predictor/
├── bootstrap.sh              # One-command setup script
├── pyproject.toml             # Python package config
├── .env.example               # Template for API keys
├── docker-compose.yml         # Container orchestration
├── data/
│   ├── footy.duckdb           # Main database
│   └── models/                # Trained model artifacts
├── src/footy/
│   ├── cli/                   # CLI package (Typer sub-apps)
│   │   ├── __init__.py        # Root app + sub-group wiring
│   │   ├── _shared.py         # Console, logging, lazy imports
│   │   ├── pipeline_cmds.py   # go, refresh, matchday, nuke, serve
│   │   ├── data_cmds.py       # footy data …
│   │   ├── model_cmds.py      # footy model …
│   │   ├── ai_cmds.py         # footy ai …
│   │   ├── scheduler_cmds.py  # footy scheduler …
│   │   ├── perf_cmds.py       # footy perf …
│   │   └── stats_cmds.py      # footy stats …
│   ├── config.py              # Settings from .env (cached)
│   ├── db.py                  # DuckDB connection + schema
│   ├── pipeline.py            # Core pipeline logic
│   ├── utils.py               # Shared helpers (safe_num, outcome_label, metrics)
│   ├── normalize.py           # Team name normalization
│   ├── extras.py              # Match extras ingestion
│   ├── fixtures_odds.py       # Odds for upcoming matches
│   ├── scheduler.py           # Background job scheduler
│   ├── continuous_training.py # Auto-retraining pipeline
│   ├── performance_tracker.py # Model performance tracking
│   ├── degradation_alerts.py  # Accuracy drift alerts
│   ├── cache.py               # Prediction caching
│   ├── h2h.py                 # Head-to-head statistics
│   ├── xg.py                  # xG computation
│   ├── understat.py           # Understat xG provider
│   ├── fbref.py               # FBRef advanced stats provider
│   ├── models/
│   │   ├── council.py         # v8 Expert Council (primary)
│   │   ├── elo.py             # Dynamic Elo ratings
│   │   ├── elo_core.py        # Shared Elo primitives
│   │   ├── poisson.py         # Weighted Poisson model
│   │   ├── dixon_coles.py     # Dixon-Coles correction
│   │   ├── v3.py              # v3 GBDT form model
│   │   └── v5.py              # v5 ultimate ensemble
│   ├── providers/
│   │   ├── football_data_org.py  # football-data.org API (with retry)
│   │   ├── fdcuk_history.py      # Historical data scraper
│   │   ├── fdcuk_fixtures.py     # Fixture scraper
│   │   ├── api_football.py       # API-Football provider
│   │   ├── opta_analyst.py       # Opta predictions scraper
│   │   ├── odds_scraper.py       # Odds aggregation
│   │   └── news_gdelt.py         # GDELT news feed
│   └── llm/
│       ├── ollama_client.py
│       └── news_extractor.py
└── web/
    ├── api.py                 # FastAPI backend
    ├── static/
    │   ├── style.css
    │   └── app.js             # Alpine.js frontend
    └── templates/
        └── index.html
```

## Testing

```bash
pip install -e ".[test]"
footy self-test              # Unit + integration tests
footy self-test --smoke      # Include live-DB smoke tests
footy self-test --cov        # With coverage report
```

## Network Access

The server binds to `0.0.0.0:8000`, accessible from any device on the local network:

```
http://<server-ip>:8000
```

## Deployment

### Docker (self-hosted)

```bash
docker-compose up -d          # Build + run on port 8000
```

### Render.com (free tier)

1. Push to GitHub
2. Connect repo at [render.com](https://render.com)
3. Render auto-detects `render.yaml` — creates a free web service
4. Set `FOOTBALL_DATA_ORG_TOKEN` in the Render dashboard
5. Deploy — live at `https://footy-predictor.onrender.com`

> Free tier sleeps after 15 min inactivity; first request takes ~30s to wake.

### Railway.app (free tier)

1. Push to GitHub
2. Connect repo at [railway.app](https://railway.app)
3. Railway detects the `Procfile`
4. Set environment variables in the dashboard
5. Deploy — auto-assigned URL

### Manual (any VPS)

```bash
git clone <repo> && cd football-predictor
bash bootstrap.sh
source .venv/bin/activate
footy go                      # Full pipeline
uvicorn web.api:app --host 0.0.0.0 --port 8000
```
