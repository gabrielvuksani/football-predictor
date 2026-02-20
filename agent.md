# Agent Context — Footy Predictor

> This file is written for AI coding agents. It describes the project's architecture,
> conventions, and key decisions so a new agent can pick up work without re-reading
> the entire codebase.

## Project Overview

ML-powered football match prediction system covering 5 European leagues (PL, PD, SA,
BL1, FL1) with 45,000+ historical matches. Uses an **11-expert council** architecture
(v10_council) that feeds ~270+ features into a multi-model stacking meta-learner
(HistGradientBoosting + RandomForest + LogisticRegression with Nelder-Mead learned weights).

**v11 upgrade**: MLE-estimated Dixon-Coles ρ per competition, real xG from API-Football,
Asian Handicap + BTTS markets, FPL data persistence (injuries/availability/FDR), 14-feature
InjuryAvailabilityExpert, walk-forward CV deployment gating, feature-count validation at
predict time, composite schedule difficulty index.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Database | DuckDB (single file: `data/footy.duckdb`) |
| ML | scikit-learn, numpy, scipy |
| CLI | Typer (sub-apps in `src/footy/cli/`) |
| Web API | FastAPI (`web/api.py`) |
| Frontend | Alpine.js + vanilla CSS (no build step) — 10-tab SPA |
| LLM | Ollama (local, optional) |
| Console | Rich |
| HTTP | httpx |
| Static site | GitHub Pages (`docs/`) — manual fork of `web/` with `FOOTY_STATIC` flag |

## Key Files

| File | Purpose |
|------|---------|
| `src/footy/models/council.py` | Core model — 11 experts + multi-model meta-learner |
| `src/footy/models/advanced_math.py` | Mathematical foundations (DC, Skellam, MC, Beta-Binom, etc.) |
| `src/footy/pipeline.py` | Orchestrates the full `footy go` pipeline |
| `src/footy/db.py` | Database schema bootstrap (all CREATE TABLE statements) |
| `src/footy/config.py` | Settings from .env (pydantic-settings) |
| `src/footy/normalize.py` | Team name canonicalization |
| `src/footy/team_mapping.py` | 290+ team name mappings, fuzzy match, provider hints |
| `src/footy/h2h.py` | Head-to-head stats via SQL aggregation |
| `src/footy/xg.py` | Expected goals with data-learned conversion rates |
| `src/footy/llm/insights.py` | 14 LLM insight functions (BTTS/OU, accas, form, accuracy, review) |
| `src/footy/continuous_training.py` | Auto-retraining pipeline (drift + deploy + rollback) |
| `src/footy/performance_tracker.py` | Model performance tracking + error analysis |
| `web/api.py` | All REST endpoints (~25 routes) |
| `web/templates/index.html` | SPA template (Alpine.js, 10 tabs) |
| `web/static/app.js` | Frontend routing & state (v5) |
| `web/static/style.css` | Dark glassmorphism design system |

## Architecture: v10_council

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
│ shrinkage    │ AF injuries, │              │
│ (18 feats)   │ FDR, squad   │              │
│              │ strength (14)│              │
└──────────────┴──────────────┴──────────────┘
                         ↓
Layer 2 — CONFLICT & CONSENSUS
  Expert variance, pairwise agreements (55 pairs),
  winner vote concentration, per-expert entropy,
  KL divergence per expert vs ensemble mean,
  cross-domain interactions, one-hot competition
                         ↓
Layer 3 — MULTI-MODEL STACK (META-LEARNER)
  HistGradientBoosting + RandomForest + LogisticRegression
  Weights learned via Nelder-Mead on validation logloss
  Each wrapped in isotonic calibration (cv=5)
  → ~250+ features → P(Home, Draw, Away)
                         ↓
Layer 4 — WALK-FORWARD VALIDATION
  4-fold expanding-window temporal CV
```

## Web UI — 10-Tab SPA

The frontend is a single-page app (Alpine.js, no build step) with 10 tabs:

| Tab | Route | API Source |
|-----|-------|-----------|
| Matches | `/` | `/api/matches` |
| Insights | `/insights` | `/api/insights/value-bets` |
| BTTS/O2.5 | `/btts` | `/api/insights/btts-ou` |
| Accumulators | `/accas` | `/api/insights/accumulators` |
| Form | `/form` | `/api/insights/form-table/{comp}` |
| Tables | `/table` | `/api/league-table/{comp}` |
| Accuracy | `/accuracy` | `/api/insights/accuracy` |
| Review | `/review` | `/api/insights/round-preview/{comp}`, `/api/insights/post-match-review` |
| Stats | `/stats` | `/api/stats`, `/api/performance` |
| Training | `/training` | `/api/training/status` |

**GitHub Pages:** `docs/` is a manual fork of `web/` with `window.FOOTY_STATIC = true` flag. Static mode uses `_staticUrl()` to transform API paths to local JSON files.

## Database Tables (DuckDB)

Key tables — all in `data/footy.duckdb`:

- `matches` — All fixtures (match_id PK, home_team, away_team, home_goals, away_goals, status, competition, season, utc_date)
- `match_extras` — Odds + match stats (hs, as_, hst, ast, b365h/d/a, hthg, htag, AH line/prices, BTTS yes/no, formations, lineups, AF xG/possession/stats)
- `match_xg` — Computed expected goals
- `predictions` — Model outputs (p_home, p_draw, p_away, eg_home, eg_away, notes JSON)
- `prediction_scores` — Scored predictions (outcome, logloss, brier, correct)
- `elo_state` — Current Elo ratings per team
- `h2h_stats` — Any-venue H2H aggregates
- `h2h_venue_stats` — Venue-specific H2H
- `fpl_availability` — FPL injury/availability data per team (total_players, available, doubtful, injured, suspended, injury_score, squad_strength, key_absences_json)
- `fpl_fixture_difficulty` — FPL fixture difficulty ratings per team (fdr_next_3, fdr_next_6, upcoming_json)
- `opta_predictions` — Scraped Opta win probabilities
- `team_mappings` — Canonical team name mapping
- `metrics` — Aggregated model performance
- `expert_cache` — Cached expert council breakdowns
- `llm_insights` — LLM-generated match narratives
- `scheduled_jobs` / `job_runs` — Background scheduler
- `model_training_records` / `model_deployments` / `retraining_schedules` — Continuous training

## CLI Structure

CLI uses Typer with sub-apps registered in `src/footy/cli/__init__.py`:

```
footy go / refresh / matchday / nuke / serve / update / self-test
footy data {ingest, history, extras, h2h, xg, ...}
footy model {train, predict, retrain, drift-check, ...}
footy ai {preview, value, review, explain-match, ...}
footy scheduler {start, stop, add, list, ...}
footy perf {summary, ranking, trend, daily, health, ...}
footy opta {fetch, show}
footy pages {export}
```

## Conventions

### Code Style
- All print output uses `from rich.console import Console; console = Console()`
- Never pass `flush=True` to `console.print()` (Rich doesn't accept it)
- Use `logging` module for operational logs; Rich console for user-facing output
- SQL queries use parameterized `?` placeholders — never f-strings for user input
- Team names go through `canonical_team_name()` from `footy.normalize`
- Use `datetime.now(timezone.utc)` — never `datetime.utcnow()` (deprecated)

### Model Versions
- Current production: `v10_council`
- Model file: `data/models/v10_council.joblib`
- All predictions write `model_version = 'v10_council'`
- Old models (v2–v5) exist as base layers within the council
- `footy model train-meta` now trains the v10 council (not legacy v2 stacker)
- `footy model predict` now uses council.predict_upcoming (not the removed pipeline.predict_upcoming)

### Data Flow
```
Data Ingestion → match extras → odds → train base models →
train council → predict → score predictions → H2H → xG
```

### Web Routing
- SPA at `/` with Alpine.js handling client-side routing
- Match detail pages at `/match/{id}` (served by same SPA template)
- All API endpoints at `/api/*` (~25 routes)
- DB connections are short-lived read-only (write-lock only held by CLI pipeline)
- Streamlit UI has been deleted — web app is the sole frontend

## Stubs / Not Yet Implemented

- `src/footy/understat.py` — Real Understat integration not implemented (returns None)
- `src/footy/fbref.py` — Real FBRef integration not implemented (returns None)
- `src/footy/providers/opta_analyst.py` — Scraper created but may need adaptation as website changes

## Testing

```bash
footy self-test           # Full suite
pytest tests/ -v          # Direct pytest
```

Test fixtures use in-memory DuckDB connections. Some tests require the model
joblib file to exist (`data/models/v10_council.joblib`).

## Environment Variables

Required:
- `FOOTBALL_DATA_ORG_TOKEN` — football-data.org API key (free tier)

Optional:
- `API_FOOTBALL_KEY` — api-football.com (lineups, injuries, xG, possession, statistics, H2H)
- `THE_ODDS_API_KEY` — multi-bookmaker odds (h2h, totals, Asian Handicap, BTTS)
- `OLLAMA_HOST` / `OLLAMA_MODEL` — local LLM for AI narratives
- `DB_PATH` — override database path (default: `./data/footy.duckdb`)
- `TRACKED_COMPETITIONS` — comma-separated league codes (default: PL,PD,SA,BL1,FL1)
- `WF_LOGLOSS_GATE` — walk-forward logloss threshold for deployment (default: 1.05)
- `WF_ACCURACY_GATE` — walk-forward accuracy threshold for deployment (default: 0.38)
- `WF_MIN_FOLDS` — minimum WF-CV folds required for gating (default: 3)

## Common Tasks for Agents

### Adding a new expert
1. Create class in `council.py` inheriting expert pattern (see `GoalPatternExpert`)
2. Add to `ALL_EXPERTS` list
3. Update `_build_meta_X` pairwise agreements
4. Update `_prepare_df` if new columns needed
5. Update dummy rows in `predict_upcoming` and `get_expert_breakdown`
6. Increment expert count in docstring and tests

### Adding a new API endpoint
1. Add route in `web/api.py`
2. Follow pattern: `con()` for DB, try/except with JSONResponse error handling
3. Add corresponding UI section in `index.html` + state in `app.js`
4. Sync changes to `docs/` for GitHub Pages (app.js, index.html, style.css)

### Adding a new provider
1. Create module in `src/footy/providers/`
2. Use `ratelimit.py` RateLimiter for rate limiting
3. Use `team_mapping.py` for team name resolution
4. Create CLI commands in `src/footy/cli/` and register in `__init__.py`
