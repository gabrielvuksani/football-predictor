# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the dev server
uvicorn web.api:app --reload

# Run all tests (excludes slow/smoke/enhanced)
pytest tests/ -x -q --tb=short -m "not slow and not smoke" --ignore=tests/test_enhanced_prediction.py -k "not test_from_db_reconstructs"

# Run a single test file
pytest tests/test_api.py -x -q --tb=short

# Run a single test function
pytest tests/test_api.py::test_health -x -q --tb=short

# Lint
ruff check src/ web/ --select E,W,F --ignore E501,E702,E741,W293,W291

# CLI (data refresh, predictions, training)
python -m footy.cli --help

# Generate static site JSON for GitHub Pages
python -m footy.cli pages export
```

## Architecture

**Stack:** Python 3.13 / FastAPI / DuckDB / Alpine.js SPA / SwiftUI iOS

### Prediction Pipeline (5 layers)

```
Layer 1: 50 Expert Models → ExpertResult(probs, confidence, features)
Layer 2: Conflict/consensus signals, KL divergence, interaction features
Layer 3: CatBoost + XGBoost + HistGBM + LR stacking ensemble (84 features)
Layer 4: Temperature scaling calibration
Layer 5: Optional LLM narrative (Ollama/Groq)
```

The pipeline lives in `src/footy/models/council.py` (~2100 lines). It orchestrates experts, builds the feature matrix via `_build_v13_features()`, and runs the stacking ensemble.

**Important:** `council.py` has two feature builders — `_build_v13_features()` (~84 features, used for prediction) and `_build_meta_X()` (~400+ features, legacy). The trained model expects `_build_v13_features` output. Do not mix them.

### Expert System

All experts inherit from `Expert` ABC in `src/footy/models/experts/_base.py`:

```python
class Expert(ABC):
    def compute(self, df: pd.DataFrame) -> ExpertResult:
        # df sorted by utc_date ASC, with NaN goals for upcoming matches
        ...

@dataclass
class ExpertResult:
    probs: np.ndarray          # (n, 3) — P(Home, Draw, Away)
    confidence: np.ndarray     # (n,) ∈ [0, 1]
    features: dict[str, np.ndarray]  # prefixed features, e.g. "elo_diff"
```

**Critical pattern:** Use `_is_finished(row)` to guard state updates in experts — upcoming dummy rows have NaN goals and must NOT update rolling state (Elo, form, etc.).

Each expert uses a unique feature prefix (e.g., `elo_`, `nb_`, `kalman_`). The full expert list is in `src/footy/models/experts/__init__.py` as `ALL_EXPERTS`.

Shared services exist to avoid duplicate computation:
- `_league_table_tracker.py` — single tracker shared by Context, LeagueTable, and Motivation experts

### Web API

`web/api.py` is a thin FastAPI factory. Routes are modular:

- `web/routes/health.py` — health check, last-updated, data freshness, sources
- `web/routes/matches.py` — match listing, detail, experts, H2H, form, xG, patterns, models, AI
- `web/routes/insights.py` — value bets, BTTS, accumulators, form table, accuracy
- `web/routes/stats.py` — dashboard, stats, performance, league table
- `web/routes/system.py` — training, weights, rankings, model lab, self-learning, refresh
- `web/routes/predictions.py` — Bayesian predict, unified prediction, export
- `web/routes/advanced.py` — season simulation, team profile, streaks, prediction history

Shared utilities in `web/routes/__init__.py`: `con()` for short-lived read-only DuckDB connections, `safe_error()`, `validate_model()`, `validate_competition()`.

### Frontend

Single Alpine.js SPA at `web/templates/index.html` + `web/static/app.js`. One monolithic `Alpine.data('app', ...)` component with 16 tabs. The CSS design system is in `web/static/style.css` using CSS custom properties.

**Fonts:** Sora (display/headings), Plus Jakarta Sans (body), JetBrains Mono (data/numbers) — all loaded from Google Fonts CDN and cached by the service worker.

A static variant for GitHub Pages lives in `docs/` with a URL resolver that maps `/api/X` to `./api/X.json` pre-generated JSON files. **The docs/ frontend diverges from web/ and must be manually synced.** Specifically: docs/index.html is missing the simulation, streaks, and prediction history tabs, and its match detail view uses inline styles instead of CSS classes.

### Data

- DuckDB database at `data/footy.duckdb` (47+ tables, 93K+ matches)
- 25+ data providers in `src/footy/providers/` — all core data is free, no API keys required
- Primary chain: football-data.co.uk → SofaScore → OpenFootball → TheSportsDB

### iOS

Native SwiftUI app at `ios/FootyPredictor/`. MVVM-ish: Models/, Views/, Services/, Helpers/. The iOS app covers ~40% of the web API surface — many features are web-only. The Xcode project needs 7 newer Swift files added to its build phases before it will compile.

## Key Conventions

- **Model version:** The active trained model is `v13_oracle`. The `VALID_MODELS` set in `web/routes/__init__.py` controls accepted versions. The app/UI version is v14 Apex.
- **Competition codes:** 19 tracked competitions (PL, PD, SA, BL1, FL1, etc.) validated by `VALID_COMPETITIONS` in `web/routes/__init__.py`.
- **Version strings:** When bumping versions, grep project-wide — version references exist in `web/api.py`, `web/routes/health.py`, `pyproject.toml`, `Dockerfile`, `requirements.txt`, `src/footy/models/council.py`, `src/footy/models/experts/__init__.py`, `web/static/sw.js`, `web/static/manifest.json`.
- **Alpine.js bindings:** The HTML template uses x-data, x-show, x-for, x-if, @click, x-model, x-text, x-cloak. The `[x-cloak]` CSS rule is required to prevent FOUC.
- **CSS class names used dynamically in JS:** `badgeClass()` returns `badge badge-{PL,SA,...}`, `verdict().cls` returns `verdict-{home,draw,away}`, `confidence()` returns `{high,medium,low}` for `conf-` prefix, `consensusClass()` returns `{agree,mixed,clash}` for `consensus-` prefix. All of these must have corresponding CSS definitions.
- **DuckDB connections:** Always use `con()` from `web/routes/__init__.py` for read-only access with retry. Each API call gets its own short-lived connection.

## CI/CD

7 GitHub Actions workflows in `.github/workflows/`:
- `ci.yml` + `test.yml` — pytest on push/PR (slightly redundant)
- `autopilot.yml` — full pipeline every 6 hours (ingest → predict → score → learn → deploy)
- `deploy.yml` — Render deploy on push to main
- `static.yml` — GitHub Pages deploy on push to main
- `data-refresh.yml` — data refresh every 6 hours
- `model-retrain.yml` — weekly retrain (Mondays 3am)

## Known Gotchas

- `POST /api/refresh` triggers a subprocess with no authentication — CORS is `allow_origins=["*"]`
- `requirements.txt` is missing `xgboost`, `statsbombpy`, and `gdeltdoc` that `pyproject.toml` declares — Docker builds fall back to HistGBM only
- The `docs/app.js` has diverged from `web/static/app.js` — features like success toasts use different implementations
- `council.py` has 5 feature extractions that store results in variables but never add them to the feature matrix (dead code in `_build_v13_features`)
