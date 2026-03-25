# Footy Predictor

AI-powered football match prediction system. 50 expert models, 115 focused features, CatBoost+XGBoost stacking ensemble, 56.4% three-way accuracy. Fully automated, zero cost.

**[Live Predictions](https://gabrielvuksani.github.io/football-predictor/)** · **[Autopilot](../../actions/workflows/autopilot.yml)** · **[Release](../../releases/latest)**

---

## Architecture

```
50 Experts → 115 Features → CatBoost+XGBoost+HistGBM Stack → Temperature Scaling → Calibrated Probabilities
     ↑                                                              ↓
Self-Learning ←── Score Predictions ←── Match Results ←── Every 6 Hours
```

Each expert analyzes one aspect of a match — team strength, goal models, form, market odds, weather, injuries, manager changes, betting line movements. Their outputs become features for a stacking ensemble with learned weights. Temperature scaling calibrates the final probabilities per-league. The Hedge algorithm continuously reweights experts based on prediction accuracy.

### Expert Models (50)

| Category | Count | Models |
|---|---|---|
| **Rating Systems** | 8 | Elo, Glicko-2, Pi-Rating, Network PageRank, Kalman-Elo, Bayesian Rate, Bayesian State-Space, TrueSkill |
| **Goal Models** | 8 | Dixon-Coles Poisson, Negative Binomial, Zero-Inflated Poisson, Frank Copula, Weibull, Double Poisson, Skellam, Ordered Probit |
| **Market** | 2 | Bookmaker Odds (Power method), Squad Market Value |
| **Form & History** | 5 | Recent Form, Head-to-Head, Momentum, Momentum Indicators, Goal Patterns |
| **Context** | 6 | Match Context, Motivation, Squad Rotation, Injury Impact, Weather, Referee Tendencies |
| **Advanced Stats** | 4 | xG Regression, FBref Advanced Metrics, Expected Points, Opta Win Probabilities |
| **Structural** | 3 | League Table Position, Seasonal Patterns, Venue Effects |
| **Dynamic Strength** | 5 | Pythagorean Expectation, Score-Driven (GAS), Adaptive Bayesian, Hidden Markov Model, Transfer Impact |
| **Real-Time Signals** | 4 | News Sentiment (GDELT), Betting Line Movement, Manager Changes, Lineup Rotation |
| **Match Dynamics** | 3 | Comeback/Collapse Patterns, Schedule Fatigue, Opponent-Adjusted Metrics |
| **Psychology** | 2 | Crowd Momentum, Team Morale |

### Key Methods

- **Dixon-Coles** with rho correction and exponential time decay
- **Diagonal-Inflated Bivariate Poisson (DIBP)** — state-of-the-art score matrix (RPS 0.189)
- **Conway-Maxwell-Poisson** with data-driven dispersion parameters
- **Frank Copula** for modelling goal dependency between teams
- **Power method** overround removal (outperforms Shin normalization)
- **Bayesian Model Averaging** with actual log-likelihoods (not uniform weights)
- **Score-driven (GAS) models** for time-varying team strength
- **Hidden Markov Models** for team regime detection (dominant/average/vulnerable)
- **Spike-and-slab Bayesian shrinkage** for adaptive expert parameters
- **Hedge algorithm** for provably optimal online expert reweighting

---

## Performance

| Metric | Value | Context |
|---|---|---|
| **Three-way accuracy** | **56.4%** | Random baseline = 33%, bookmakers typically achieve 53-55% |
| **Walk-forward CV** | **57.0%** | Honest temporal evaluation — no future data leakage |
| **Brier score** | **0.184** | FiveThirtyEight's model achieved ~0.196 |
| **Calibration (ECE)** | **0.010** | Near-perfect probability calibration |
| **Log-loss** | 0.932 | |
| **BTTS accuracy** | 55.1% | Both Teams To Score |
| **Over 2.5 accuracy** | 57.0% | Over/under 2.5 goals |

---

## Data Sources

All core data is free. No API keys required for basic operation.

| Source | Data | Method |
|---|---|---|
| football-data.co.uk | 25+ years of results, odds for 19 leagues | CSV download |
| Understat (via soccerdata) | Real xG for top 5 European leagues | Library scrape |
| ClubElo (via soccerdata) | Elo ratings for European clubs | Library scrape |
| SofaScore | Upcoming fixtures and live scores | API |
| Open-Meteo | Weather forecasts for match venues | Free API |
| Opta / TheAnalyst | Professional win probability predictions | API |
| FPL API | Premier League player injuries and availability | Free API |
| GDELT | News sentiment for team-related coverage | Free API |
| TheSportsDB | Match metadata and competition info | Free API |

19 competitions tracked: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Eredivisie, Primeira Liga, Championship, Belgian Pro League, Swiss Super League, Turkish Super Lig, and 8 more.

---

## Self-Learning Pipeline

The model continuously improves without manual intervention:

1. **Hedge Algorithm** — After each scored match, experts that predicted well get exponentially more weight. Provably optimal regret bound from online learning theory.
2. **Per-League Temperature Scaling** — Each competition gets its own calibration parameters. La Liga draws are treated differently from Bundesliga goal-fests.
3. **Drift Detection** — Four statistical detectors (CUSUM, Page-Hinkley, ADWIN, DDM) monitor rolling accuracy. When drift is detected, retraining uses higher time-decay to adapt faster.

---

## Automation

The **Autopilot** workflow runs every 6 hours via GitHub Actions:

```
Ingest → Enrich xG → Score & Self-Learn → Predict → Fetch Opta → Export → Deploy
```

1. Ingest latest results and fixtures from free data sources
2. Enrich with real xG data from Understat via soccerdata
3. Score finished predictions and run self-learning feedback loop
4. Generate predictions for the next 14 days using persisted model
5. Fetch Opta/TheAnalyst professional predictions for comparison
6. Export static JSON to `docs/` for GitHub Pages
7. Persist database to GitHub Releases for cross-run state

**Cost: $0/month.** GitHub Actions (free for public repos) + GitHub Releases (free storage) + GitHub Pages (free hosting).

---

## Quick Start

```bash
git clone https://github.com/gabrielvuksani/football-predictor.git
cd football-predictor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pip install catboost xgboost soccerdata

footy go                  # Full pipeline: ingest + train + predict
uvicorn web.api:app --reload  # Dev server at localhost:8000
```

### CLI Commands

```bash
footy data ingest         # Fetch latest fixtures and results
footy data opta           # Fetch Opta win probabilities
footy model train-meta    # Train the stacking ensemble
footy model predict       # Generate predictions for upcoming matches
footy model drift-check   # Check for accuracy drift
footy perf summary        # Print performance metrics
footy pages export        # Update static site JSON files
```

### Run Tests

```bash
pytest tests/ -x -q --tb=short -m "not slow and not smoke"
ruff check src/ web/ --select E,W,F --ignore E501,E702,E741,W293,W291
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Prediction Engine** | Python 3.13, CatBoost, XGBoost, LightGBM, scikit-learn, scipy, numpy |
| **Database** | DuckDB (96 MB, 47+ tables, 93K+ matches) |
| **Web API** | FastAPI with modular route architecture |
| **Frontend** | Alpine.js SPA, Sora + Plus Jakarta Sans + JetBrains Mono |
| **iOS** | Native SwiftUI |
| **Data** | 25 providers via soccerdata, httpx, BeautifulSoup |
| **CI/CD** | 7 GitHub Actions workflows, Render deployment |
| **Static Site** | GitHub Pages with pre-generated JSON API |

## Project Structure

```
src/footy/
  models/experts/       50 expert model files (each produces ExpertResult)
  models/math/          Statistical functions — distributions, copulas, BMA, scoring
  models/council.py     Meta-learner ensemble (2,139 lines)
  providers/            25 data scrapers and API clients
  llm/                  LLM integration (Groq, Ollama) for match narratives
  self_learning.py      Hedge algorithm + 4 drift detectors
  pipeline.py           Data ingestion orchestration
  backtest.py           Walk-forward backtesting framework

web/
  api.py                FastAPI app factory
  routes/               7 modular route files (health, matches, insights, stats, system, predictions, advanced)
  templates/            Alpine.js SPA (single index.html)
  static/               CSS design system, app.js, service worker, PWA manifest

ios/                    Native SwiftUI app (Models, Views, Services, Helpers)
docs/                   GitHub Pages static export with pre-generated JSON
tests/                  36 test files
.github/workflows/      7 CI/CD workflows including Autopilot
```

---

## License

MIT

---

*50 experts. 115 features. 56.4% accuracy. Zero cost. Self-learning. Deployed every 6 hours.*
