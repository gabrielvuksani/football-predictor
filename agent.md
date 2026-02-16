# Agent Context — Footy Predictor

> This file is written for AI coding agents. It describes the project's architecture,
> conventions, and key decisions so a new agent can pick up work without re-reading
> the entire codebase.

## Project Overview

ML-powered football match prediction system covering 5 European leagues (PL, PD, SA,
BL1, FL1) with 45,000+ historical matches. Uses an **8-expert council** architecture
(v8_council) that feeds ~170 features into a HistGradientBoosting meta-learner.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Database | DuckDB (single file: `data/footy.duckdb`) |
| ML | scikit-learn, numpy, scipy |
| CLI | Typer (sub-apps in `src/footy/cli/`) |
| Web API | FastAPI (`web/api.py`) |
| Frontend | Alpine.js + vanilla CSS (no build step) |
| LLM | Ollama (local, optional) |
| Console | Rich |
| HTTP | httpx |

## Key Files

| File | Purpose |
|------|---------|
| `src/footy/models/council.py` | Core model — 8 experts + meta-learner |
| `src/footy/pipeline.py` | Orchestrates the full `footy go` pipeline |
| `src/footy/db.py` | Database schema bootstrap (all CREATE TABLE statements) |
| `src/footy/config.py` | Settings from .env (pydantic-settings) |
| `src/footy/normalize.py` | Team name canonicalization |
| `src/footy/team_mapping.py` | 290+ team name mappings, fuzzy match, provider hints |
| `src/footy/h2h.py` | Head-to-head stats via SQL aggregation |
| `src/footy/xg.py` | Expected goals with data-learned conversion rates |
| `web/api.py` | All REST endpoints |
| `web/templates/index.html` | SPA template (Alpine.js) |
| `web/static/app.js` | Frontend routing & state |
| `web/static/style.css` | Dark glassmorphism design system |

## Architecture: v8_council

```
Layer 1 — EIGHT SPECIALIST EXPERTS
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  EloExpert   │ MarketExpert │  FormExpert  │PoissonExpert │
│ Rating dyn,  │ Multi-tier   │ OAF, venue-  │ Venue-split  │
│ momentum,    │ odds, line   │ split PPG,   │ attack/def,  │
│ home adv     │ movement     │ BTTS, CS     │ BTTS, O2.5   │
├──────────────┼──────────────┼──────────────┼──────────────┤
│  H2HExpert   │ContextExpert │GoalPattern   │ LeagueTable  │
│ Bayesian     │ Rest days,   │ First-goal,  │ Position,    │
│ Dirichlet,   │ congestion,  │ HT scoring,  │ PPG, points  │
│ time decay   │ day-of-week  │ comebacks    │ gap to top   │
└──────────────┴──────────────┴──────────────┴──────────────┘
                         ↓
Layer 2 — CONFLICT & CONSENSUS (9 pairwise features)
                         ↓
Layer 3 — META-LEARNER (HistGradientBoosting + Isotonic calibration)
           + Dixon-Coles pseudo-expert
           → P(Home, Draw, Away)
```

## Database Tables (DuckDB)

Key tables — all in `data/footy.duckdb`:

- `matches` — All fixtures (match_id PK, home_team, away_team, home_goals, away_goals, status, competition, season, utc_date)
- `match_extras` — Odds + match stats (hs, as_, hst, ast, b365h/d/a, hthg, htag, etc.)
- `match_xg` — Computed expected goals
- `predictions` — Model outputs (p_home, p_draw, p_away, eg_home, eg_away, notes JSON)
- `prediction_scores` — Scored predictions (outcome, logloss, brier, correct)
- `elo_state` — Current Elo ratings per team
- `h2h_stats` — Any-venue H2H aggregates
- `h2h_venue_stats` — Venue-specific H2H
- `opta_predictions` — Scraped Opta win probabilities
- `team_mappings` — Canonical team name mapping
- `metrics` — Aggregated model performance

## CLI Structure

CLI uses Typer with sub-apps registered in `src/footy/cli/__init__.py`:

```
footy go / refresh / matchday / nuke / serve / update / self-test
footy data {ingest, history, extras, h2h, xg, ...}
footy model {train, predict, retrain, drift-check, ...}
footy ai {preview, value, review, explain-match, ...}
footy scheduler {start, stop, add, list, ...}
footy perf {summary, ranking, trend, daily, health, ...}
footy stats {understat-team, fbref-shooting, ...}
footy opta {fetch, show}
```

## Conventions

### Code Style
- All print output uses `from rich.console import Console; console = Console()`
- Never pass `flush=True` to `console.print()` (Rich doesn't accept it)
- Use `logging` module for operational logs; Rich console for user-facing output
- SQL queries use parameterized `?` placeholders — never f-strings for user input
- Team names go through `canonical_team_name()` from `footy.normalize`

### Model Versions
- Current production: `v8_council`
- Model file: `data/models/v8_council.joblib`
- All predictions write `model_version = 'v8_council'`
- Old models (v2–v5) exist as base layers within the council

### Data Flow
```
Data Ingestion → match extras → odds → train base models →
train council → predict → score predictions → H2H → xG
```

### Web Routing
- SPA at `/` with Alpine.js handling client-side routing
- Match detail pages at `/match/{id}` (served by same SPA template)
- All API endpoints at `/api/*`
- DB connections are short-lived read-only (write-lock only held by CLI pipeline)

## Stubs / Not Yet Implemented

- `src/footy/understat.py` — Real Understat integration not implemented (returns None)
- `src/footy/fbref.py` — Real FBRef integration not implemented (returns None)
- `src/footy/providers/opta_analyst.py` — Scraper created but may need adaptation as website changes

## Testing

```bash
footy self-test           # Full suite
pytest tests/ -v          # Direct pytest
pytest tests/test_upgrades.py -v  # v8 upgrade tests
```

Test fixtures use in-memory DuckDB connections. Some tests require the model
joblib file to exist (`data/models/v8_council.joblib`).

## Environment Variables

Required:
- `FOOTBALL_DATA_ORG_TOKEN` — football-data.org API key (free tier)

Optional:
- `API_FOOTBALL_KEY` — api-football.com (lineups, injuries)
- `THE_ODDS_API_KEY` — external odds
- `OLLAMA_HOST` / `OLLAMA_MODEL` — local LLM for AI narratives
- `DB_PATH` — override database path (default: `./data/footy.duckdb`)
- `TRACKED_COMPETITIONS` — comma-separated league codes (default: PL,PD,SA,BL1,FL1)

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
2. Follow pattern: `con()` for DB, try/except for IOException
3. Add corresponding UI section in `index.html` + state in `app.js`

### Adding a new provider
1. Create module in `src/footy/providers/`
2. Use `ratelimit.py` RateLimiter for rate limiting
3. Use `team_mapping.py` for team name resolution
4. Create CLI commands in `src/footy/cli/` and register in `__init__.py`
