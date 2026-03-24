# Footy Predictor v13 "The Oracle"

AI-powered football match prediction system with **50 expert models**, **138 focused features**, and a self-learning CatBoost + XGBoost stacking ensemble achieving **56.2% accuracy** on three-way match outcomes (Home/Draw/Away).

**[Live Predictions](https://gabrielvuksani.github.io/football-predictor/)** | **[Autopilot Workflow](../../actions/workflows/autopilot.yml)**

---

## How It Works

```
50 Expert Models → 719 Raw Features → 138 Focused Features → CatBoost+XGBoost Ensemble → Temperature Scaling → Calibrated Predictions
```

The system runs **50 specialist expert models** that each analyze a different aspect of a football match — from Elo ratings and xG regression to team morale and betting line movements. Their outputs are compressed into **138 high-signal features** and fed to a **CatBoost + XGBoost ensemble** with learned weights. A **temperature scaling layer** calibrates the final probabilities. Every 6 hours, the system self-learns from scored predictions and deploys fresh forecasts — fully automated, zero cost.

---

## Expert Models (50)

| Category | # | Experts | What They Detect |
|----------|---|---------|-----------------|
| **Rating Systems** | 8 | Elo, Glicko-2, Pi-Rating, Network, Kalman-Elo, Bayesian Rate, Bayesian State-Space, TrueSkill | Team strength and uncertainty |
| **Goal Models** | 8 | Poisson (Dixon-Coles), Negative Binomial, ZIP, Frank Copula, Weibull, Double Poisson, Skellam, Ordered Probit | Score distributions and goal dependency |
| **Market** | 2 | Market Odds (Power method), Market Value | Betting odds and squad valuations |
| **Form & History** | 5 | Form, H2H, Momentum, Momentum Indicators, Goal Patterns | Recent performance and trends |
| **Context** | 6 | Context, Motivation, Squad Rotation, Injury, Weather, Referee | Fatigue, motivation and external factors |
| **Advanced Stats** | 4 | xG Regression, FBref Advanced, Expected Points, Opta | xG analysis, pressing and Opta predictions |
| **Structural** | 3 | League Table, Seasonal Pattern, Venue | Position and season phase |
| **Dynamic Strength** | 5 | Pythagorean, GAS, Adaptive Bayesian, HMM, Transfer | Regression-to-mean and regime detection |
| **Real-Time Signals** | 4 | News Sentiment, Betting Movement, Manager, Lineup | Smart money, coaching changes and rotation |
| **Match Dynamics** | 3 | Match Dynamics, Schedule Context, Opponent-Adjusted | Comebacks, fatigue and opponent-quality stats |
| **Psychology** | 2 | Crowd Momentum, Morale | Home fortress, confidence and fragility |

### Scenarios Covered

Home advantage, form and momentum, head-to-head history, fixture congestion, European competition fatigue, injuries, weather, referee tendencies, market odds and smart money, xG regression-to-mean, manager changes, lineup rotation, transfer disruption, news sentiment, team confidence and fragility, international breaks, promoted team honeymoon, streak psychology, comebacks and collapses, set piece dependency, card discipline, altitude and synthetic pitch, points to safety/title, derby effects, dead rubbers, crowd impact and morale.

---

## Model Performance

| Metric | Value | Context |
|--------|-------|---------|
| **Three-way accuracy** | **56.2%** | Random = 33%, bookmakers = 53-55% |
| **Walk-forward CV** | **57.0%** | Honest temporal evaluation |
| **Brier score** | **0.184** | FiveThirtyEight achieved ~0.196 |
| **Log-loss** | 0.932 | |
| **Calibration (ECE)** | 0.014 | Near-perfect calibration |
| **BTTS accuracy** | 55.1% | |
| **Over 2.5 accuracy** | 57.0% | |

---

## Key Methods

- **Dixon-Coles** with rho correction and time decay
- **Diagonal-Inflated Bivariate Poisson (DIBP)** — state of the art
- **Conway-Maxwell-Poisson** with data-driven dispersion
- **Frank Copula** for goal dependency
- **Power method** overround removal
- **Bayesian Model Averaging** with log-likelihoods
- **Score-driven (GAS) models** for dynamic strength
- **Hidden Markov Models** for team regime detection
- **Spike-and-slab Bayesian shrinkage** for adaptive parameters
- **Hedge algorithm** for self-learning expert weights

---

## Data Sources (All Free)

| Source | Data |
|--------|------|
| football-data.co.uk | 25+ years of results and odds |
| Understat via soccerdata | Real xG for top 5 leagues |
| ClubElo via soccerdata | Elo ratings for European clubs |
| SofaScore | Upcoming fixtures |
| Open-Meteo | Weather forecasts |
| Opta / TheAnalyst | Win probability predictions |
| FPL API | Player injuries and availability |

Zero API keys required for core functionality.

---

## Self-Learning

The model continuously improves through three mechanisms:

1. **Hedge Algorithm** — After each scored match, experts that predicted correctly get exponentially more weight. Provably optimal online learning.
2. **Per-League Temperature** — Each league gets adaptive calibration. La Liga draws get different treatment than Bundesliga goals.
3. **Drift Detection** — Four detectors (CUSUM, Page-Hinkley, ADWIN, DDM) monitor accuracy. When drift is detected, retraining uses higher time-decay.

---

## Automation

The **Autopilot** runs every 6 hours via GitHub Actions — free for public repos:

1. Ingest latest match data
2. Enrich with real xG from Understat
3. Score predictions and run self-learning loop
4. Retrain weekly or on drift detection
5. Generate predictions for next 14 days
6. Fetch Opta predictions
7. Export and deploy to GitHub Pages
8. Persist database to GitHub Releases

**Cost: $0/month.** GitHub Actions + Releases + Pages — all free.

---

## Quick Start

```bash
git clone https://github.com/gabrielvuksani/football-predictor.git
cd football-predictor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pip install catboost xgboost soccerdata  # recommended

footy go      # Full pipeline: ingest + train + predict
footy serve   # Web server at localhost:8000
```

### Key Commands

```bash
footy data ingest       # Fetch latest fixtures
footy data opta         # Fetch Opta predictions
footy model train-meta  # Train v13 Oracle model
footy model predict     # Generate predictions
footy model drift-check # Check accuracy drift
footy perf summary      # Performance overview
footy pages export      # Update static site
```

---

## Tech Stack

Python 3.13 | CatBoost + XGBoost | DuckDB | FastAPI | Alpine.js | SwiftUI | GitHub Actions

---

## Project Structure

```
src/footy/
  models/experts/    50 expert model files
  models/math/       Statistical functions (distributions, scoring, BMA)
  models/council.py  Meta-learner ensemble
  providers/         18 data scrapers and API clients
  backtest.py        Historical backtest framework
  self_learning.py   Hedge algorithm + drift detection
  pipeline.py        Data ingestion pipeline

web/                 FastAPI + Alpine.js frontend
ios/                 Native SwiftUI app
docs/                GitHub Pages static export
```

---

## License

MIT

---

*50 expert models. 138 features. Zero cost. Self-learning. Updated every 6 hours.*
