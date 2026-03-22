# Footy Predictor

AI-powered football match prediction system using a 27-expert ensemble with self-learning capabilities.

---

## Overview

Footy Predictor is a production-grade football match prediction platform that combines 27 specialist AI experts, 25 data providers, and a multi-model stacking architecture to generate calibrated probability forecasts for 18 leagues worldwide. The system features walk-forward cross-validation, continuous retraining with automatic drift detection, a FastAPI backend, a PWA-ready web frontend, and a native iOS app.

The prediction pipeline ingests historical and live data, feeds it through a layered expert council architecture, and produces probability distributions for match outcomes (1X2), BTTS, Over/Under 2.5, correct score, and Asian Handicap markets.

**At a glance:**

- **27 specialist experts** (28 with optional TrueSkill)
- **25 data providers** (API + scraper stack)
- **18 leagues** tracked across Europe
- **5-layer architecture** with learned stacking weights
- **930+ automated tests**
- **Zero-cost mode** -- runs entirely on free/public data sources

---

## Key Features

- **27-Expert Council** -- each expert produces independent probability distributions and features, aggregated through a meta-learner
- **Self-Learning Feedback** -- competition-aware expert weighting from scored historical outcomes
- **Multi-Model Stack** -- HistGradientBoosting + RandomForest + LogisticRegression with Nelder-Mead-optimised weights
- **Walk-Forward Validation** -- expanding-window temporal CV with deployment gating (logloss < 1.05, accuracy > 0.38)
- **Continuous Retraining** -- automatic drift detection, retraining, validation, and deploy/rollback
- **Dixon-Coles Correction** -- MLE-estimated rho per competition for low-scoring match bias
- **Monte Carlo Simulation** -- 2,000 DC-correlated draws for BTTS, O2.5, and scoreline distributions
- **Value Betting** -- Kelly criterion edge detection against bookmaker odds
- **AI Match Previews** -- LLM-generated narratives via Ollama or Groq
- **PWA Web App** -- installable, dark/light mode, real-time predictions
- **Native iOS App** -- SwiftUI with offline caching
- **Background Scheduler** -- cron-based job automation for data refresh, training, and prediction
- **Zero-Cost Operation** -- full functionality with free scrapers; optional API keys for richer data

---

## Architecture

The prediction system is organised into five layers:

```
Layer 1 -- 27 SPECIALIST EXPERTS
+--------------+--------------+--------------+--------------+
| EloExpert    | MarketExpert | FormExpert   | PoissonExpert|
| Glicko2      | PiRating     | TrueSkill*   | DoublePois.  |
| H2HExpert    | ContextExp.  | GoalPattern  | LeagueTable  |
| Momentum     | BayesianRate | Injury/Avail | WeatherExp.  |
| RefereeExp.  | MarketValue  | VenueExpert  | SeasonalPat. |
| XPtsExpert   | NetworkStr.  | ZIPExpert    | BayesianSS   |
| CopulaExpert | WeibullExp.  | Motivation   | SquadRotation|
+--------------+--------------+--------------+--------------+
                             |
                             v
Layer 2 -- CONFLICT & CONSENSUS ANALYSIS
  Expert variance, pairwise agreement (351 pairs),
  winner vote concentration, per-expert entropy,
  KL divergence vs ensemble mean, cross-domain
  interactions (Elo x Form, Market x Poisson, etc.),
  one-hot competition encoding
                             |
                             v
Layer 3 -- MULTI-MODEL STACK (META-LEARNER)
  HistGradientBoosting + RandomForest + LogisticRegression
  Weights learned via Nelder-Mead on validation logloss
  Each base model wrapped in isotonic calibration (cv=5)
  ~300+ features --> P(Home, Draw, Away)
                             |
                             v
Layer 4 -- WALK-FORWARD CROSS-VALIDATION
  4-fold expanding-window temporal CV
  Per-fold: logloss, Brier score, accuracy, ECE
  Deployment gate: logloss < 1.05, accuracy > 0.38
  Feature importance stability analysis
                             |
                             v
Layer 5 -- UNIFIED PREDICTION AGGREGATOR
  Final calibrated probabilities, market comparison,
  value bet detection (Kelly criterion), confidence
  scoring, and prediction export
```

*TrueSkill is optional (requires the `trueskill` package).*

---

## Expert Models (27)

| # | Expert | Description |
|---|--------|-------------|
| 1 | EloExpert | Dynamic Elo ratings with momentum, tanh/log transforms, weighted recency |
| 2 | MarketExpert | Multi-tier bookmaker odds, logit-space analysis, Shannon entropy, Pinnacle pin |
| 3 | FormExpert | Bayesian-shrunk recent form, schedule difficulty, PPG conversion |
| 4 | PoissonExpert | Dixon-Coles adjusted score matrix, Skellam goal-difference, Monte Carlo (2,000 sims) |
| 5 | H2HExpert | Bayesian Dirichlet prior head-to-head with time decay |
| 6 | ContextExpert | Season stage, fixture congestion, rest days ratio, day-of-week effects |
| 7 | GoalPatternExpert | Scoring/conceding streaks, goal bursts, drought detection |
| 8 | LeagueTableExpert | Simulated standings, position delta, points per game |
| 9 | MomentumExpert | EMA crossovers, slope regression, volatility, burst detection |
| 10 | BayesianRateExpert | Beta-Binomial shrinkage for win rate, clean sheet, BTTS, O2.5 (18 features) |
| 11 | Glicko2Expert | Glicko-2 rating system with rating deviation and volatility |
| 12 | PiRatingExpert | Pi-rating system capturing home/away attacking and defensive strength |
| 13 | InjuryAvailabilityExpert | FPL injuries/availability, API-Football injury counts, FDR, squad strength (14 features) |
| 14 | WeatherExpert | Open-Meteo weather conditions -- temperature, rain, wind, snow impact |
| 15 | RefereeExpert | Referee tendencies -- cards, fouls, penalties, home bias |
| 16 | MarketValueExpert | Transfermarkt squad valuations and value differentials |
| 17 | VenueExpert | Stadium capacity, altitude, surface type, travel distance |
| 18 | SeasonalPatternExpert | Month/season phase patterns, holiday effects, end-of-season dynamics |
| 19 | XPtsExpert | Expected points from xG-based match simulations |
| 20 | NetworkStrengthExpert | Graph-based strength propagation through results network |
| 21 | ZIPExpert | Zero-inflated Poisson for modelling excess 0-0 draws |
| 22 | BayesianStateSpaceExpert | Kalman-filter team strength tracking with process noise |
| 23 | CopulaExpert | Copula-based joint goal distribution capturing tail dependencies |
| 24 | DoublePoissonExpert | Double Poisson model for over/under-dispersed goal counts |
| 25 | WeibullExpert | Weibull time-to-goal model for scoring rate analysis |
| 26 | MotivationExpert | Title race, relegation battle, European qualification context |
| 27 | SquadRotationExpert | Rotation risk from fixture congestion and squad depth |

Optional: **TrueSkillExpert** (28th) -- Microsoft TrueSkill Bayesian skill ratings (requires `trueskill` package).

---

## Leagues Tracked (18)

| Code | League | Country |
|------|--------|---------|
| PL | Premier League | England |
| ELC | Championship | England |
| PD | La Liga | Spain |
| SA | Serie A | Italy |
| BL1 | Bundesliga | Germany |
| FL1 | Ligue 1 | France |
| DED | Eredivisie | Netherlands |
| PPL | Primeira Liga | Portugal |
| TR1 | Super Lig | Turkey |
| BEL | Jupiler Pro League | Belgium |
| SL | Super League | Switzerland |
| A1 | Bundesliga | Austria |
| GR1 | Super League | Greece |
| SWS | Allsvenskan | Sweden |
| DK1 | Superliga | Denmark |
| SE1 | Allsvenskan | Sweden |
| NO1 | Eliteserien | Norway |
| PL1 | Ekstraklasa | Poland |

Configurable via `TRACKED_COMPETITIONS` in `.env`.

---

## Quick Start

### Prerequisites

- Python 3.10+
- **No API keys required** -- runs entirely on free public data (football-data.co.uk CSVs, web scrapers, Open-Meteo weather)
- Optional: Xcode 15+ for the iOS app
- Optional: Ollama for AI match previews

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd football-predictor-main

# Option A: One-liner bootstrap (creates venv, installs deps, copies .env)
bash bootstrap.sh

# Option B: Manual setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
cp .env.example .env
```

That's it. No API keys needed. The `.env.example` is pre-configured for zero-cost operation using free public data sources.

### First Run

```bash
# Activate environment (if using manual setup)
source .venv/bin/activate

# Run the full pipeline: ingest historical data -> train models -> predict
footy go

# Start the web server
footy serve
```

Then visit **http://localhost:8000** in your browser.

### iOS App

```
Open ios/FootyPredictor/FootyPredictor.xcodeproj in Xcode 15+
Set the API base URL in Settings
Build and run on simulator or device
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FOOTBALL_DATA_ORG_TOKEN` | -- | football-data.org API key (free tier: 10 req/min) |
| `API_FOOTBALL_KEY` | -- | API-Football key (free tier: 100 req/day) |
| `THE_ODDS_API_KEY` | -- | The Odds API key for multi-bookmaker odds |
| `THESPORTSDB_KEY` | `3` | TheSportsDB key (3 = free dev key) |
| `ENABLE_SCRAPER_STACK` | `true` | Enable zero-cost scraper providers |
| `ENABLE_UNDERSTAT` | `true` | Enable Understat xG scraping |
| `ENABLE_FBREF` | `true` | Enable FBref stats scraping |
| `ENABLE_SOFASCORE` | `true` | Enable SofaScore scraping |
| `ENABLE_TRANSFERMARKT` | `true` | Enable Transfermarkt valuations |
| `ENABLE_ODDSPORTAL` | `true` | Enable OddsPortal odds scraping |
| `ENABLE_CLUBELO` | `true` | Enable ClubElo ratings |
| `ENABLE_OPENFOOTBALL` | `true` | Enable OpenFootball data |
| `ENABLE_OPEN_METEO` | `true` | Enable Open-Meteo weather data |
| `TRACKED_COMPETITIONS` | `PL,PD,SA,...` | Comma-separated league codes (18 leagues) |
| `LOOKAHEAD_DAYS` | `7` | Days ahead to fetch fixtures |
| `DB_PATH` | `./data/footy.duckdb` | DuckDB database path |
| `REQUEST_TIMEOUT_SECONDS` | `20` | HTTP request timeout |
| `SCRAPER_CACHE_TTL_SECONDS` | `900` | Scraper cache TTL (15 min) |
| `PRE_MATCH_REFRESH_MINUTES` | `15` | Auto-refresh before kickoff |
| `ODDS_MOVEMENT_THRESHOLD` | `0.05` | Odds movement alert threshold |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama model for AI previews |
| `GROQ_API_KEY` | -- | Groq cloud LLM API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `LLM_PROVIDER_ORDER` | `groq,ollama` | LLM provider priority |
| `WF_LOGLOSS_GATE` | `1.05` | Walk-forward CV max logloss for deployment |
| `WF_ACCURACY_GATE` | `0.38` | Walk-forward CV min accuracy for deployment |
| `WF_MIN_FOLDS` | `3` | Minimum CV folds required for gating |

### Data Sources (25)

**All core data is free and requires zero API keys.** Optional API keys provide richer data but are not needed.

| # | Source | Key? | Data | Notes |
|---|--------|:---:|------|-------|
| 1 | football-data.co.uk | **No** | 25 seasons of results + odds from 10+ bookmakers | Primary historical data source |
| 2 | SofaScore | **No** | Live scores, lineups, statistics | Primary live data source |
| 3 | OpenFootball | **No** | Fixtures and results (GitHub JSON) | Fallback fixture source |
| 4 | FBref | **No** | Advanced stats (xG, xA, progressive passing) | StatsBomb-powered |
| 5 | Understat | **No** | Shot-level xG data | Covers top 6 leagues |
| 6 | ClubElo | **No** | Club Elo ratings | Updated daily |
| 7 | Open-Meteo | **No** | Weather (temp, wind, rain, humidity) | 10,000 calls/day free |
| 8 | Transfermarkt | **No** | Squad values, injuries, transfers | Web scraping |
| 9 | OddsPortal | **No** | Historical and live odds | Web scraping |
| 10 | FPL API | **No** | Player injuries, availability | Premier League only |
| 11 | GDELT | **No** | Team news headlines | Sentiment analysis |
| 12 | Ollama | **No** | AI match analysis | Requires local install |
| 13 | football-data.org | Optional | Live fixtures, standings | Free tier: 10 req/min |
| 14 | API-Football | Optional | Lineups, injuries, H2H | Free tier: 100 req/day |
| 15 | The Odds API | Optional | Odds from 40+ bookmakers | Free tier: 500 req/month |
| 16 | Groq | Optional | Cloud LLM analysis | Free tier available |

The system automatically falls back through providers: SofaScore -> OpenFootball -> football-data.co.uk CSVs. Odds data is extracted directly from the free CSV files (Pinnacle, Bet365, William Hill, and more).

---

## CLI Commands

Commands are organised into sub-groups. Run `footy --help` or `footy <group> --help` for full details.

### Core Pipeline

| Command | Description |
|---------|-------------|
| `footy go` | Full pipeline: history, enrich, train, predict, score, export |
| `footy go --skip-history` | Skip history download (fast daily use) |
| `footy refresh` | Quick update: ingest recent, retrain, predict, export |
| `footy matchday` | Refresh + AI preview for all leagues (requires LLM) |
| `footy update` | Lightweight ingest, train, predict, metrics |
| `footy serve` | Start web UI on port 8000 |
| `footy nuke` | Full database reset and rebuild |
| `footy self-test` | Run the pytest suite |

### Data Management (`footy data`)

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

### Model Training (`footy model`)

| Command | Description |
|---------|-------------|
| `footy model train` | Train Elo + Poisson base models |
| `footy model train-meta` | Train the expert council meta-learner |
| `footy model predict` | Generate predictions for upcoming matches |
| `footy model metrics` | Backtest accuracy metrics |
| `footy model backtest` | Walk-forward time-split backtest |
| `footy model retrain` | Auto-retrain: check, train, validate, deploy/rollback |
| `footy model drift-check` | Check for prediction accuracy drift |
| `footy model setup` | Configure continuous retraining thresholds |
| `footy model deploy` | Deploy a trained model version |
| `footy model rollback` | Rollback to previous model version |
| `footy model status` | Retraining status for all models |
| `footy model history` | Show training history |
| `footy model deployments` | Show current deployments |

### AI Analysis (`footy ai`)

| Command | Description |
|---------|-------------|
| `footy ai preview` | AI match preview (single match or full round) |
| `footy ai value` | Value bet scanner (model edge vs market) |
| `footy ai review` | Post-match accuracy review |
| `footy ai extract-news` | Extract + analyse team news via LLM |
| `footy ai analyze-form` | Analyse recent team form |
| `footy ai explain-match` | LLM explanation for a prediction |
| `footy ai status` | LLM system health check |

### Scheduler (`footy scheduler`)

| Command | Description |
|---------|-------------|
| `footy scheduler add` | Create a scheduled job (cron syntax) |
| `footy scheduler start` | Start the background scheduler |
| `footy scheduler stop` | Stop the scheduler |
| `footy scheduler list` | List all jobs |
| `footy scheduler enable/disable` | Enable or disable a job |
| `footy scheduler remove` | Remove a job |
| `footy scheduler history` | Execution history |
| `footy scheduler stats` | Aggregate stats by job type |

### Performance (`footy perf`)

| Command | Description |
|---------|-------------|
| `footy perf summary` | All-model comparison table |
| `footy perf ranking` | Models ranked by accuracy |
| `footy perf trend` | Performance trend over time |
| `footy perf health` | Health check against thresholds |
| `footy perf compare` | Side-by-side model comparison |
| `footy perf improvement` | Deep error analysis with recommendations |
| `footy perf alerts-check` | Check all models for degradation |
| `footy perf alerts-list` | List active alerts |

### Other

| Command | Description |
|---------|-------------|
| `footy opta fetch` | Scrape Opta win probabilities |
| `footy opta show` | Display cached Opta predictions |
| `footy pages export` | Export predictions to static JSON for GitHub Pages |

---

## API Reference

The FastAPI backend serves at `http://localhost:8000`:

### Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/matches?days=14` | Upcoming matches with predictions |
| GET | `/api/matches/{id}` | Match detail (prediction, odds, Elo, DC, MC) |
| GET | `/api/matches/{id}/experts` | Expert council breakdown |
| GET | `/api/matches/{id}/h2h` | Head-to-head history |
| GET | `/api/matches/{id}/form` | Recent form (W/D/L streak + PPG) |
| GET | `/api/matches/{id}/xg` | xG breakdown |
| GET | `/api/matches/{id}/patterns` | Goal pattern analysis |
| GET | `/api/matches/{id}/ai` | AI narrative (requires LLM) |

### Insights

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/insights/value-bets` | Value bets with Kelly criterion |
| GET | `/api/insights/btts-ou` | BTTS and Over/Under 2.5 analysis |
| GET | `/api/insights/accumulators` | Auto-generated accumulator bets |
| GET | `/api/insights/form-table/{comp}` | League form table |
| GET | `/api/insights/accuracy` | Prediction accuracy dashboard |
| GET | `/api/insights/round-preview/{comp}` | AI round preview |
| GET | `/api/insights/post-match-review` | Post-match accuracy review |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/training/status` | Drift detection and retraining status |
| GET | `/api/stats` | Database statistics |
| GET | `/api/performance` | Model performance and calibration |
| GET | `/api/league-table/{comp}` | Simulated league standings |
| GET | `/api/last-updated` | Last prediction timestamp |

---

## Web App

The web frontend is a single-page application built with Alpine.js, served by FastAPI.

- **PWA installable** -- add to home screen on mobile
- **Dark/light mode** -- automatic theme detection with manual toggle
- **Predictions view** -- upcoming matches with probability bars, confidence, and value indicators
- **Match detail** -- expert breakdown, H2H, form, xG, goal patterns, AI narrative
- **Insights** -- value bets, BTTS/O2.5, accumulators, form tables
- **Simulation** -- season simulation with league table projections
- **Accuracy dashboard** -- historical prediction accuracy tracking
- **Forecast Engine Lab** -- live model/expert diagnostics

---

## iOS App

Native SwiftUI application at `ios/FootyPredictor/`.

**Views:**

- **Match List** -- upcoming fixtures with prediction cards
- **Match Detail** -- full prediction breakdown with expert analysis
- **Insights** -- value bets, form analysis, statistical insights
- **Season Simulation** -- interactive league table projections
- **Streaks** -- team winning/losing/drawing streaks
- **Team Profile** -- detailed team statistics and history
- **System Status** -- backend health and data freshness
- **Settings** -- API configuration, theme, notifications

**Features:**

- Offline caching for predictions
- Pull-to-refresh data sync
- Native iOS design patterns
- Dark mode support

---

## Database

DuckDB at `data/footy.duckdb`:

| Table | Description |
|-------|-------------|
| `matches` | All fixtures (finished + upcoming) |
| `match_extras` | Odds, stats, formations, lineups, xG, possession |
| `match_xg` | Expected goals per match |
| `predictions` | Model predictions per match per version |
| `prediction_scores` | Scored predictions with accuracy metrics |
| `elo_state` | Current Elo rating per team |
| `poisson_state` | Fitted Poisson parameters |
| `h2h_stats` | Head-to-head statistics |
| `fpl_availability` | FPL injury/availability per team |
| `fpl_fixture_difficulty` | FPL fixture difficulty ratings |
| `news` | Team news headlines |
| `expert_cache` | Cached expert council breakdowns |
| `llm_insights` | LLM-generated match narratives |
| `metrics` | Model performance metrics over time |
| `opta_predictions` | Scraped Opta win probabilities |
| `team_mappings` | Cross-provider team name mappings |
| `scheduled_jobs` | Background scheduler job definitions |
| `job_runs` | Scheduler execution history |
| `model_training_records` | Training run audit trail |
| `model_deployments` | Active model deployment registry |
| `retraining_schedules` | Auto-retrain configuration |

---

## Continuous Training

The system supports fully automated model retraining with drift detection and safe deployment:

```bash
# Configure: retrain after 20 new matches, deploy if >0.5% improvement
footy model setup v10_council --threshold-matches 20 --threshold-improvement 0.005

# Check status
footy model status

# Auto-retrain: check -> train -> validate -> deploy or rollback
footy model retrain

# Force retrain regardless of thresholds
footy model retrain --force

# Check for accuracy drift (last 60 days vs baseline)
footy model drift-check

# Schedule automatic daily retraining
footy scheduler add daily_retrain retrain "0 4 * * *"
footy scheduler start
```

**Retraining pipeline:**

1. **Threshold check** -- count new finished matches since last training
2. **Drift detection** -- compare recent accuracy (60d) vs baseline (365d); triggers on >5pp drop
3. **Train** -- full council training with all 27 experts
4. **Validate** -- compare test-set performance vs current production model
5. **Deploy or rollback** -- deploy only if performance improves; auto-rollback on regression
6. **Audit trail** -- every train/deploy/rollback logged in DuckDB

---

## Development

### Running Tests

```bash
pip install -e ".[test]"
footy self-test                # ~930 unit + integration tests
footy self-test --smoke        # Include live-DB smoke tests
footy self-test --cov          # With coverage report

# Or directly with pytest
pytest tests/ -x -q
```

### Project Structure

```
football-predictor-main/
|-- bootstrap.sh                  # One-command setup
|-- pyproject.toml                # Package config
|-- .env.example                  # Environment template
|-- docker-compose.yml            # Container orchestration
|-- render.yaml                   # Render.com deployment config
|-- Dockerfile / Procfile         # Container + process definitions
|-- data/
|   |-- footy.duckdb              # Main database
|   +-- models/                   # Trained model artifacts
|-- src/footy/
|   |-- cli/                      # Typer CLI sub-apps
|   |-- config.py                 # Settings from .env
|   |-- db.py                     # DuckDB connection + schema
|   |-- pipeline.py               # Core pipeline logic
|   |-- walkforward.py            # Walk-forward temporal CV
|   |-- continuous_training.py    # Auto-retraining pipeline
|   |-- performance_tracker.py    # Performance tracking + error analysis
|   |-- degradation_alerts.py     # Accuracy drift alerts
|   |-- scheduler.py              # Background job scheduler
|   |-- models/
|   |   |-- council.py            # Expert council meta-learner
|   |   |-- advanced_math.py      # Mathematical foundations
|   |   |-- experts/              # 27 expert implementations
|   |   |-- elo.py                # Dynamic Elo ratings
|   |   |-- poisson.py            # Weighted Poisson model
|   |   +-- dixon_coles.py        # Dixon-Coles correction
|   |-- providers/                # 25 data providers
|   +-- llm/                      # Ollama + Groq LLM clients
|-- web/
|   |-- api.py                    # FastAPI backend
|   |-- static/                   # CSS + Alpine.js frontend
|   +-- templates/                # HTML templates
|-- ios/FootyPredictor/           # Native SwiftUI iOS app
|-- docs/                         # GitHub Pages static export
|-- tests/                        # ~930 tests (pytest)
+-- .github/workflows/            # CI/CD pipelines
```

### Adding a New Expert

1. Create `src/footy/models/experts/your_expert.py`
2. Subclass `Expert` from `_base.py`
3. Implement `name`, `feature_names`, and `predict(match, history, **ctx)` returning an `ExpertResult`
4. Register in `src/footy/models/experts/__init__.py` by adding to `ALL_EXPERTS`
5. Add tests in `tests/`
6. Run `footy model train-meta` to retrain the council with the new expert

---

## Deployment

### Render (Recommended)

Already configured with `render.yaml`:

1. Push to GitHub
2. Connect repo at [render.com](https://render.com)
3. Render auto-detects `render.yaml` and creates a web service
4. Set environment variables in the Render dashboard
5. Deploy

### Docker

```bash
cp .env.example .env        # Edit with your settings
docker-compose up -d         # Build + run on port 8000
```

### Railway

1. Push to GitHub
2. Connect repo at [railway.app](https://railway.app)
3. Railway detects the `Procfile`
4. Set environment variables in the dashboard
5. Deploy

### Manual (Any VPS)

```bash
git clone <repo> && cd football-predictor-main
bash bootstrap.sh
source .venv/bin/activate
footy go
uvicorn web.api:app --host 0.0.0.0 --port 8000
```

### GitHub Actions CI/CD

Six workflows in `.github/workflows/`:

| Workflow | File | Purpose |
|----------|------|---------|
| CI | `ci.yml` | Lint + test on every push |
| Test | `test.yml` | Full test suite |
| Data Refresh | `data-refresh.yml` | Scheduled data ingestion |
| Model Retrain | `model-retrain.yml` | Scheduled model retraining |
| Deploy | `deploy.yml` | Production deployment |
| Static Pages | `static.yml` | GitHub Pages export |

---

## Mathematical Foundation

| Method | Description |
|--------|-------------|
| **Poisson regression** | Weighted MLE for team attack/defence rates |
| **Dixon-Coles** | Tau correction for low-scoring match joint probabilities; MLE rho per competition |
| **Skellam distribution** | Goal-difference probabilities from independent Poisson rates |
| **Monte Carlo simulation** | 2,000 DC-correlated draws for market probabilities (BTTS, O2.5, scorelines) |
| **Beta-Binomial shrinkage** | Bayesian regularisation for noisy rate estimates |
| **Elo / Glicko-2 / Pi-rating** | Dynamic team strength rating systems with uncertainty |
| **TrueSkill** | Microsoft Bayesian skill rating (optional) |
| **Zero-Inflated Poisson** | Excess 0-0 draw modelling |
| **Double Poisson** | Over/under-dispersed goal count modelling |
| **Weibull time-to-goal** | Hazard-rate scoring intensity model |
| **Copula models** | Joint goal distributions capturing tail dependencies |
| **Kalman filter** | Bayesian state-space team strength tracking |
| **Nelder-Mead optimisation** | Stacking weight learning on validation logloss |
| **Isotonic calibration** | Non-parametric probability calibration per base model |
| **KL divergence** | Expert disagreement quantification |
| **Shannon entropy** | Market uncertainty measurement |
| **Kelly criterion** | Optimal bet sizing for value detection |
| **Walk-forward CV** | Expanding-window temporal cross-validation with deployment gating |

---

## Model Evolution

```
v1 (Poisson)
  +-- v2 (Meta-stacker)
       +-- v3 (GBDT form)
            +-- v4 (Expert council v1)
                 +-- v5 (Ultimate ensemble)
                      +-- v8 (9-expert council)
                           +-- v10 (19-expert council + learned weights)
                                +-- v13 (27-expert council + self-learning)  <-- current
```

---

## License

MIT
