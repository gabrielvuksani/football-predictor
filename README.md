# Footy Predictor

ML-powered football match prediction system with an **11-expert AI council**, research-backed mathematical models, multi-model stacking with learned weights, walk-forward validation, continuous retraining, FastAPI backend, and dark-mode frontend.

Current Version: **v10\_council** (v0.0.2b4 - Untested)

## Highlights — v10 Council

- **11 specialist experts** (was 9): added BayesianRateExpert + InjuryAvailabilityExpert
- **Dixon-Coles τ correction** for low-scoring match bias
- **Monte Carlo simulation** (2 000 DC-correlated draws) for BTTS / O2.5 / scoreline distributions
- **Skellam distribution** for exact goal-difference probabilities
- **Beta-Binomial Bayesian shrinkage** for noisy rate estimates (win rate, CS, BTTS)
- **Learned stacking weights** via Nelder-Mead optimisation on validation logloss (replaces fixed 60/25/15)
- **KL divergence** per expert vs ensemble mean for disagreement signalling
- **Logit-space market analysis**, Shannon entropy, Pinnacle-specific features
- **Schedule difficulty** and **shot conversion** in FormExpert
- **286 unit/integration tests** passing (scheduler, training manager, math, experts, API)

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

### Manual Setup

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

### Optional API Keys (all free)

| Key | Source | What It Adds |
|-----|--------|-------------|
| `THE_ODDS_API_KEY` | [the-odds-api.com](https://the-odds-api.com/) | Multi-bookmaker odds from 40+ bookmakers |
| `THESPORTSDB_KEY` | [thesportsdb.com](https://www.thesportsdb.com/api.php) | Team metadata, stadium info, badges |

FPL (Fantasy Premier League) data is fetched automatically — no key needed.

## Commands

Commands are organised into sub-groups. Run `footy --help` or `footy <group> --help` for full details.

### Day-to-Day (root-level)

| Command | What It Does |
|---------|-------------|
| `footy go` | Full pipeline: 25-season history → new APIs → train Elo/Poisson → train 11-expert council → predict → H2H → xG → export pages |
| `footy go --skip-history` | Same but skips history download (fast daily use) |
| `footy refresh` | Quick update: ingest recent → new APIs → retrain council → predict → export pages |
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
| `footy model train-meta` | Train the v10 council model |
| `footy model retrain` | Auto-retrain council: check → train → validate → deploy/rollback |
| `footy model drift-check` | Check for prediction accuracy drift |
| `footy model setup` | Configure continuous retraining (threshold + improvement gate) |
| `footy model deploy` | Deploy a trained model version to production |
| `footy model rollback` | Rollback to previous model version |
| `footy model status` | Retraining status for all models |
| `footy model record` | Record a manual training run |
| `footy model history` | Show training history for a model type |
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
| `footy scheduler add` | Create a scheduled job (cron syntax). Types: ingest, train\_base, train\_council, predict, score, retrain, full\_refresh |
| `footy scheduler start` | Start the background scheduler |
| `footy scheduler stop` | Stop the scheduler |
| `footy scheduler list` | List all jobs |
| `footy scheduler enable` | Enable a disabled job |
| `footy scheduler disable` | Disable a job (keeps config) |
| `footy scheduler remove` | Remove a job (use --confirm) |
| `footy scheduler history` | Execution history for a job |
| `footy scheduler stats` | Aggregate stats by job type |

### `footy perf` — Performance & Alerts

| Command | Description |
|---------|-------------|
| `footy perf summary` | All-model comparison table |
| `footy perf ranking` | Models ranked by accuracy |
| `footy perf trend` | Performance trend for a model |
| `footy perf daily` | Daily breakdown |
| `footy perf health` | Health check against thresholds |
| `footy perf compare` | Side-by-side model comparison |
| `footy perf improvement` | Deep error analysis with actionable recommendations |
| `footy perf errors` | Detailed error breakdown |
| `footy perf alerts-check` | Check all models for degradation |
| `footy perf alerts-list` | List active alerts |

### `footy opta` — Opta Predictions

| Command | Description |
|---------|-------------|
| `footy opta fetch` | Scrape Opta win probabilities (all leagues) |
| `footy opta fetch --league PL` | Single league |
| `footy opta show` | Display cached Opta predictions |

### `footy pages` — Static Site Export

| Command | Description |
|---------|-------------|
| `footy pages export` | Export predictions to static JSON for GitHub Pages |

## Architecture

### Expert Council (v10) — Primary Model

```
Layer 1 — ELEVEN SPECIALIST EXPERTS
┌──────────────┬──────────────┬──────────────┐
│  EloExpert   │ MarketExpert │  FormExpert  │
│ Rating diff, │ Multi-tier   │ Bayesian     │
│ tanh/log     │ odds, logit  │ shrinkage,   │
│ transforms,  │ probs, Shan- │ schedule     │
│ weighted     │ non entropy, │ difficulty,  │
│ momentum     │ Pinnacle pin │ conversion   │
├──────────────┼──────────────┼──────────────┤
│PoissonExpert │  H2HExpert   │ContextExpert │
│ DC-adjusted  │ Bayesian     │ Season stage,│
│ score matrix,│ Dirichlet    │ congestion,  │
│ Skellam GD,  │ prior, time  │ rest ratio,  │
│ Monte Carlo  │ decay        │ day-of-week  │
│ (2000 sims)  │              │              │
├──────────────┼──────────────┼──────────────┤
│GoalPattern   │ LeagueTable  │  Momentum    │
│ Scoring &    │ Simulated    │ EMA cross-   │
│ conceding    │ standings,   │ overs, slope │
│ streaks,     │ position     │ regression,  │
│ burst &      │ delta, pts   │ volatility,  │
│ drought      │ per game     │ burst detect │
├──────────────┼──────────────┼──────────────┤
│ Bayesian     │   Injury     │              │
│ Rate Expert  │ Availability │              │
│ Beta-Binom   │ FPL injuries │              │
│ shrinkage    │ fixture      │              │
│ (18 feats)   │ difficulty   │              │
└──────────────┴──────────────┴──────────────┘
                     ↓
Layer 2 — CONFLICT & CONSENSUS SIGNALS
  Expert variance, pairwise agreements (55 pairs),
  winner vote concentration, per-expert entropy,
  KL divergence per expert vs ensemble mean,
  cross-domain interactions (Elo×Form, Market×Poisson,
  Momentum×Form, DC_ph×Market_ph, MC_ph×Market_ph,
  BayesWR×PoissonLam), one-hot competition encoding
                     ↓
Layer 3 — MULTI-MODEL STACK (META-LEARNER)
  HistGradientBoosting + RandomForest + LogisticRegression
  Weights learned via Nelder-Mead on validation logloss
  Each wrapped in isotonic calibration (cv=5)
  → ~250+ features → P(Home, Draw, Away)
                     ↓
Layer 4 — WALK-FORWARD VALIDATION
  4-fold expanding-window temporal CV
  Per-fold: logloss, brier, accuracy, ECE
  Feature importance stability analysis
```

### Mathematical Methods (v10)

| Method | Module | Usage |
|--------|--------|-------|
| Beta-Binomial shrinkage | `advanced_math.py` | Noisy rate estimation (win rate, CS, BTTS, O2.5) |
| Dixon-Coles τ correction | `advanced_math.py` | Low-score joint probability adjustment |
| Skellam distribution | `advanced_math.py` | Goal-difference probabilities from Poisson rates |
| Monte Carlo simulation | `advanced_math.py` | 2000 DC-correlated draws → P(BTTS), P(O2.5), etc. |
| Logit-space operations | `advanced_math.py` | Market probability analysis in log-odds space |
| Adaptive EWMA | `advanced_math.py` | Time-weighted exponential moving averages |
| KL divergence | `advanced_math.py` | Expert disagreement quantification |
| Shannon entropy | `advanced_math.py` | Market uncertainty measurement |
| Nelder-Mead optimisation | `council.py` | Stacking weight learning on validation set |
| Isotonic calibration | `council.py` | Probability calibration per base model |
| Walk-forward CV | `walkforward.py` | Temporal cross-validation (no future leakage) |

## API Endpoints

The FastAPI backend serves at `http://localhost:8000`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI (single-page app) |
| GET | `/api/matches?days=14` | Upcoming matches with predictions |
| GET | `/api/matches/{id}` | Match detail (prediction, odds, Elo, DC, MC) |
| GET | `/api/matches/{id}/experts` | Expert council breakdown |
| GET | `/api/matches/{id}/h2h` | Head-to-head history |
| GET | `/api/matches/{id}/form` | Recent form (W/D/L streak + PPG) |
| GET | `/api/matches/{id}/xg` | xG breakdown for match |
| GET | `/api/matches/{id}/patterns` | Goal pattern analysis |
| GET | `/api/matches/{id}/ai` | AI narrative (requires Ollama) |
| GET | `/api/insights/value-bets` | Value bets with Kelly criterion |
| GET | `/api/insights/btts-ou` | BTTS & Over/Under 2.5 analysis |
| GET | `/api/insights/accumulators` | Auto-generated accumulator bets |
| GET | `/api/insights/form-table/{comp}` | League form table (PPG, BTTS%, O2.5%) |
| GET | `/api/insights/accuracy` | Prediction accuracy dashboard |
| GET | `/api/insights/round-preview/{comp}` | AI round preview |
| GET | `/api/insights/post-match-review` | Post-match accuracy review |
| GET | `/api/training/status` | Drift detection & retraining status |
| GET | `/api/stats` | Database statistics |
| GET | `/api/performance` | Model performance + calibration |
| GET | `/api/league-table/{comp}` | Simulated league standings |
| GET | `/api/last-updated` | Last prediction timestamp |

## Data Sources

| Source | What | Key Required |
|--------|------|:---:|
| football-data.org | Live fixtures & results (v4 API) | Yes |
| football-data.co.uk | 25 seasons of historical results + odds | No |
| API-Football | Lineups, injuries, pre-match context | Yes (optional) |
| The Odds API | Multi-bookmaker odds (40+ bookmakers) | Yes (optional) |
| FPL API | Player injuries, availability, fixture difficulty | No |
| TheSportsDB | Team metadata, stadium info, badges | Optional (free key) |
| GDELT | Team news headlines | No |
| Opta / The Analyst | Match win probabilities | No (scraped) |
| Ollama | Local LLM for AI analysis | No (local) |

## Environment Variables

```env
# Required
FOOTBALL_DATA_ORG_TOKEN=your_token_here
API_FOOTBALL_KEY=your_key_here          # optional but recommended

# Optional data providers (all free)
THE_ODDS_API_KEY=                        # multi-bookmaker market odds
THESPORTSDB_KEY=3                        # team metadata (default: 3 = free dev key)

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
| match\_extras | Odds + stats (shots, corners, cards) |
| match\_xg | Expected goals per match |
| predictions | Model predictions per match per version |
| prediction\_scores | Scored predictions with accuracy metrics |
| elo\_state | Current Elo rating per team |
| elo\_applied | Tracks which matches have been applied to Elo |
| poisson\_state | Fitted Poisson parameters |
| h2h\_stats | Head-to-head stats (any venue) |
| h2h\_venue\_stats | Head-to-head stats (specific venue) |
| news | Team news headlines |
| expert\_cache | Cached expert council breakdowns |
| llm\_insights | LLM-generated match narratives |
| metrics | Model performance metrics over time |
| opta\_predictions | Scraped Opta win probabilities |
| team\_mappings | Cross-provider team name mappings |
| scheduled\_jobs | Background scheduler job definitions |
| job\_runs | Scheduler execution history |
| model\_training\_records | Continuous training run audit trail |
| model\_deployments | Active model deployment registry |
| retraining\_schedules | Auto-retrain configuration per model |

## Continuous Training

The system supports fully automated model retraining:

```bash
# Configure auto-retrain: retrain after 20 new matches, deploy if >0.5% improvement
footy model setup v10_council --threshold-matches 20 --threshold-improvement 0.005

# Check if retraining is needed
footy model status

# Auto-retrain: check → train → validate → deploy (or rollback)
footy model retrain

# Force retrain regardless of thresholds
footy model retrain --force

# Check for accuracy drift (compares last 60 days vs baseline)
footy model drift-check

# Schedule auto-retrain as a cron job
footy scheduler add daily_retrain retrain "0 4 * * *"
footy scheduler start
```

### Retraining Pipeline

1. **Threshold check**: Count new finished matches since last training
2. **Drift detection**: Compare recent accuracy (60d) vs baseline (365d) — triggers if >5pp drop
3. **Train**: Full v10\_council training with all 11 experts
4. **Validate**: Compare test-set performance vs current model
5. **Deploy or rollback**: Deploy only if performance improves; auto-rollback on regression
6. **Audit trail**: Every train/deploy/rollback logged in DuckDB

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
│   │   ├── model_cmds.py      # footy model … (train, retrain, deploy, rollback)
│   │   ├── ai_cmds.py         # footy ai …
│   │   ├── scheduler_cmds.py  # footy scheduler …
│   │   ├── perf_cmds.py       # footy perf …
│   │   ├── pages_cmds.py      # footy pages …
│   │   └── opta_cmds.py       # footy opta …
│   ├── config.py              # Settings from .env (cached)
│   ├── db.py                  # DuckDB connection + schema
│   ├── pipeline.py            # Core pipeline logic
│   ├── walkforward.py         # Walk-forward temporal CV
│   ├── normalize.py           # Team name normalization
│   ├── extras.py              # Match extras ingestion
│   ├── fixtures_odds.py       # Odds for upcoming matches
│   ├── scheduler.py           # Background job scheduler (7 job types)
│   ├── continuous_training.py # Auto-retraining pipeline (drift + deploy + rollback)
│   ├── performance_tracker.py # Model performance tracking + error analysis
│   ├── degradation_alerts.py  # Accuracy drift alerts
│   ├── cache.py               # Prediction caching
│   ├── h2h.py                 # Head-to-head statistics
│   ├── xg.py                  # xG computation
│   ├── models/
│   │   ├── council.py         # v10 Expert Council (primary — 11 experts + learned weights)
│   │   ├── advanced_math.py   # Mathematical foundations (DC, Skellam, MC, shrinkage, etc.)
│   │   ├── elo.py             # Dynamic Elo ratings
│   │   ├── elo_core.py        # Shared Elo primitives
│   │   ├── poisson.py         # Weighted Poisson model
│   │   ├── dixon_coles.py     # Dixon-Coles correction
│   │   ├── v3.py              # v3 GBDT form model (legacy)
│   │   └── v5.py              # v5 ultimate ensemble (legacy)
│   ├── providers/
│   │   ├── football_data_org.py  # football-data.org (v4 API)
│   │   ├── fdcuk_history.py      # Historical data scraper
│   │   ├── fdcuk_fixtures.py     # Fixture scraper
│   │   ├── api_football.py       # API-Football provider
│   │   ├── the_odds_api.py       # The Odds API (multi-bookmaker odds)
│   │   ├── fpl.py                # Fantasy Premier League (injuries, availability)
│   │   ├── thesportsdb.py        # TheSportsDB (team metadata, venues)
│   │   ├── opta_analyst.py       # Opta predictions scraper
│   │   ├── odds_scraper.py       # Odds aggregation
│   │   └── news_gdelt.py         # GDELT news feed
│   └── llm/
│       ├── ollama_client.py
│       └── news_extractor.py
├── web/
│   ├── api.py                 # FastAPI backend (REST API)
│   ├── static/
│   │   ├── style.css
│   │   └── app.js             # Alpine.js frontend
│   └── templates/
│       └── index.html         # SPA template
├── docs/                      # GitHub Pages static export
└── tests/                     # 286 tests (pytest)
    ├── conftest.py            # Shared fixtures (in-memory DuckDB, sample data)
    ├── test_advanced_math.py  # 38 tests — all 10 mathematical modules
    ├── test_v10_council.py    # 29 tests — new experts, upgraded features, meta-learner
    ├── test_scheduler.py      # 24 tests — scheduler CRUD, execution, lifecycle
    ├── test_continuous_training.py  # 21 tests — training records, drift, deploy, rollback
    ├── test_models_council.py # Council training + prediction unit tests
    ├── test_walkforward.py    # Walk-forward CV tests
    ├── test_api.py            # FastAPI endpoint tests
    ├── test_db.py             # Database schema + query tests
    └── ...                    # Elo, Poisson, Dixon-Coles, H2H, xG, etc.
```

## Testing

```bash
pip install -e ".[test]"
footy self-test              # 286 unit + integration tests
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

## Model Evolution

```
v1 (Poisson)
  └─► v2 (Meta-stacker)
       └─► v3 (GBDT form)
            └─► v4 (Expert council v1)
                 └─► v5 (Ultimate ensemble)
                      └─► v8 (9-expert council)
                           └─► v10 (11-expert council + learned weights + DC/MC/Skellam)  ◄── current
```
