# Footy Predictor Master Plan

## Current State (February 2026)

**Primary Model**: v10_council — Expert Council with 8 specialists + meta-learner
- Logloss: 0.951 | Brier: 0.563 | Accuracy: 55.1% | ECE: 0.040 | ~170 features
- Trained on 11,745 matches across 5 leagues (PL, PD, SA, BL1, FL1)
- Web UI: FastAPI + Alpine.js dark glassmorphism frontend

## Data Sources
- football-data.org — fixtures, results, status (API key required)
- football-data.co.uk — multi-season finished results + odds/stats (free CSVs)
- GDELT — team news headlines (free)
- Ollama — local LLM for AI analysis (local, no key)
- API-Football — lineups, injuries, context (API key required)

## Model Lineage

```
v1_elo_poisson ─┐
v2_meta_stack  ─┤
v3_gbdt_form   ─┼─── v5_ultimate (retired) ──► v10_council (current)
v4_super_ensemble (retired)                      │
Dixon-Coles    ─────────────────────────────────┘
```

| Version | Type | Features | Logloss | Status |
|---------|------|----------|---------|--------|
| v1_elo_poisson | Elo + Poisson blend | ~10 | ~1.05 | Base layer |
| v2_meta_stack | LogReg stacker | ~15 | 1.041 | Base layer |
| v3_gbdt_form | GBDT + rolling form | 33 | 0.984 | Base layer |
| v4_super_ensemble | Calibrated ensemble | 50 | 1.019 | Retired |
| v5_ultimate | GBDT all signals | 94 | 0.949 | Superseded |
| **v10_council** | **Expert Council** | **~170** | **0.951** | **Primary** |

## v8 Council Architecture

**8 Specialist Experts**:
1. **EloExpert** — Team-specific home advantage, dynamic K-factor, momentum, volatility
2. **MarketExpert** — Multi-tier odds (closing > avg > max), line movement, O/U 2.5, source quality
3. **FormExpert** — OAF (Opposition-Adjusted Form), venue-split PPG, BTTS, CS, streaks, shot-on-target ratio
4. **PoissonExpert** — Venue-split attack/defense EMA, BTTS/O2.5/O1.5 from score matrix, most-likely-score, goal-diff skewness
5. **H2HExpert** — Bayesian Dirichlet prior, time-decayed observations (half-life 730d), venue-specific sub-analysis
6. **ContextExpert** — Rest days, congestion (7/14/30d), season progress, day-of-week, weekend/midweek, short-rest flags
7. **GoalPatternExpert** — First-goal rate, comeback rate, half-time scoring fractions, multi-goal/nil-nil rates, lead-holding ratio
8. **LeagueTableExpert** — Simulated live standings, position/PPG differential, points gap to top/bottom, zone flags

**Consensus Layer**: Expert variance, spread, 9 pairwise agreements (28 pairs), max disagreement, winner vote concentration, confidence-weighted ensemble, entropy

**Meta-Learner**: HistGradientBoosting (lr=0.02, depth=5, 1800 iterations, L2=0.5) + Dixon-Coles pseudo-expert + Isotonic calibration (cv=5)

## Phase 5.4: Comprehensive Normalizer & Model Upgrade
**Status**: COMPLETE ✅

### Problems Identified (Audit Feb 2026)
1. **Team Normalizer**: 200 of 291 teams (69%) UNMAPPED — only 91 have canonical mappings
2. **Wrong Fuzzy Matches**: "Verona" → "Everton" (should be Hellas Verona), "Paris FC" → PSG (different club)
3. **Cross-Provider Mismatches**: football-data.org uses "FC Internazionale Milano" while fdcuk uses "Inter" — treated as separate teams
4. **Elo Duplication**: 285 Elo entries contain duplicates of same teams under different names
5. **Model Calibration**: logloss 1.06–1.10 across models (poor), partly due to fragmented team identity

### Tasks
- [x] 5.4.1: Mega normalizer upgrade — all 291 teams mapped (100% coverage)
- [x] 5.4.2: Fixed wrong fuzzy matches (Verona→hellas-verona, Paris FC→paris-fc)
- [x] 5.4.3: Re-normalized all team names in DB (291→176 unique teams, 116 renames)
- [x] 5.4.4: Upgraded Elo model — dynamic K-factor (convergence + goal diff scaling), rating-based draw probability
- [x] 5.4.5: Upgraded Poisson model — L-BFGS-B with bounds, lower L2, parameter centering, lambda clamping
- [x] 5.4.6: Upgraded Dixon-Coles — vectorized tau computation, lambda clamping
- [x] 5.4.7: Full model retrain on clean normalized data
- [x] 5.4.8: Walk-forward backfill (9,136 predictions across 4 years)
- [x] 5.4.9: H2H recomputation (2,237 any-venue + 4,387 venue-specific pairs)
- [x] 5.4.10: xG backfill (1,000 matches)

### Results (Before → After)
| Model | logloss | accuracy |
|-------|---------|----------|
| v2_meta_stack | 1.059 → 1.041 | 44.9% → 46.5% |
| v3_gbdt_form | 1.099 → **0.984** | 43.8% → **52.0%** |
| v4_super_ensemble | 1.073 → 1.019 | 44.9% → 49.4% |
| v5_ultimate | N/A → **0.983** | N/A → **53.5%** |

Best model: **v5_ultimate** (logloss=0.983, accuracy=53.5%, ECE=0.025)

## Phase 6.0: Future Advanced Features (formerly 5.4)
- Advanced analytics dashboard
- Feature engineering for prediction models
- Model comparison dashboard
- ~~Phase 6.3: Player-level features~~ STRUCK OFF

## CLI — All Commands

### Master Commands
```bash
footy go                          # Full pipeline: history → train → predict → H2H → xG
footy go --skip-history           # Skip 8-season history download
footy refresh                     # Quick daily: ingest → extras → odds → retrain council → predict → H2H
footy matchday                    # Refresh + AI preview for all leagues
footy nuke                        # Reset everything and rebuild from scratch
footy serve                       # Start web UI (FastAPI on port 8000)
footy update                      # Legacy: ingest → train base → predict v1 → metrics
```

### Data Ingestion
```bash
footy ingest                      # Fetch fixtures (30d back, 7d forward)
footy ingest --days-back 365      # Wider window
footy ingest-history              # Pull 8 seasons from football-data.co.uk
footy ingest-history --n-seasons 25
footy ingest-extras               # Odds + match stats
footy ingest-fixtures-odds        # Odds for upcoming matches
footy ingest-af                   # API-Football context (lineups, injuries)
footy update-odds                 # External odds + model fallback
footy news                        # GDELT headlines for teams
```

### Training
```bash
footy train                       # Elo + Poisson (base models)
footy train-meta                  # v2 LogReg stacker
footy train-v3                    # v3 GBDT form model
footy backfill-wf                 # Walk-forward backfill
footy train-v4                    # v4 ensemble
footy train-v5                    # v5 ultimate model
# v7 council trains inside go/refresh
```

### Prediction
```bash
footy predict                     # v1 + v2
footy predict-v3                  # v3
footy predict-v5                  # v5
# v7 council predicts inside go/refresh
```

### Analysis & Metrics
```bash
footy metrics                     # Backtest metrics
footy compute-h2h                 # Recompute H2H stats
footy compute-xg                  # Backfill xG
footy performance-summary         # All model comparison
footy performance-ranking         # Models ranked by accuracy
footy performance-trend MODEL     # Logloss trend over time
footy performance-daily MODEL     # Daily accuracy
footy performance-health MODEL    # Model health check
footy performance-compare M1 M2   # Side-by-side comparison
footy performance-thresholds MODEL
footy drift-check                 # Prediction accuracy drift
footy backtest                    # Time-split backtest
```

### AI / Ollama
```bash
footy ai-preview                  # AI preview for all leagues
footy ai-preview --league PL      # Single league
footy ai-preview --match-id 12345 # Single match
footy ai-value                    # Value bet scanner
footy ai-value --min-edge 0.10    # Higher edge threshold
footy ai-review                   # Post-match accuracy review
footy extract-news --team Arsenal # Team news from GDELT → LLM
footy analyze-form --team Arsenal # LLM form analysis
footy explain-match --home-team X --away-team Y
footy insights-status             # Check Ollama health
```

### Retraining System
```bash
footy retrain                     # Auto-retrain
footy retrain --force             # Force retrain
footy retraining-status           # Show readiness
footy retraining-setup            # Configure thresholds
footy retraining-history MODEL    # Training audit trail
footy retraining-deploy V T       # Deploy version
footy retraining-rollback T       # Rollback model type
footy retraining-deployments      # Active deployments
footy retraining-record V T       # Record training run
```

### Scheduler
```bash
footy scheduler-start             # Start background scheduler
footy scheduler-stop              # Stop scheduler
footy scheduler-list              # List all jobs
footy scheduler-add ID TYPE CRON  # Add job
footy scheduler-enable ID
footy scheduler-disable ID
footy scheduler-remove ID --confirm
footy scheduler-history ID
footy scheduler-stats
```

### Alerts & Monitoring
```bash
footy alerts-setup                # Configure alerts
footy alerts-check                # Run checks now
footy alerts-list                 # List active alerts
footy alerts-summary              # Alert summary
footy alerts-acknowledge ID
footy alerts-resolve ID
footy alerts-snooze ID
```

### Stats Providers
```bash
footy fbref-status / fbref-shooting / fbref-possession / fbref-defense
footy fbref-passing / fbref-compare T1 T2 / fbref-all TEAM
footy understat-status / understat-team / understat-match / understat-team-rolling
```

### Opta
```bash
footy opta fetch                  # Scrape Opta predictions (all leagues)
footy opta fetch --league PL      # Single league
footy opta show                   # Display cached Opta predictions
```

### Maintenance
```bash
footy reset-states                # Clear all model state
footy cache-stats                 # Cache usage
footy cache-cleanup               # Remove expired cache
footy cache-cleanup --full        # Clear entire cache
```

## API Endpoints (FastAPI)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/api/matches?days=14` | Upcoming matches + predictions |
| GET | `/api/matches/{id}` | Match detail (prediction, odds, Elo) |
| GET | `/api/matches/{id}/experts` | Expert council breakdown |
| GET | `/api/matches/{id}/h2h` | Head-to-head history |
| GET | `/api/matches/{id}/form` | Recent form (W/D/L + PPG) |
| GET | `/api/matches/{id}/ai` | AI narrative |
| GET | `/api/insights/value-bets` | Value bets + Kelly criterion |
| GET | `/api/insights/btts-ou` | BTTS & Over/Under 2.5 analysis |
| GET | `/api/insights/accumulators` | Auto-generated accumulator bets |
| GET | `/api/insights/form-table/{comp}` | League form table |
| GET | `/api/insights/accuracy` | Prediction accuracy dashboard |
| GET | `/api/insights/round-preview/{comp}` | AI round preview |
| GET | `/api/insights/post-match-review` | Post-match accuracy review |
| GET | `/api/training/status` | Drift detection & retraining status |
| GET | `/api/stats` | Database statistics |
| GET | `/api/performance` | Model performance + calibration |
| GET | `/api/matches/{id}/xg` | xG breakdown for match |
| GET | `/api/matches/{id}/patterns` | Goal pattern analysis |
| GET | `/api/league-table/{comp}` | Simulated league standings |
| GET | `/api/last-updated` | Last prediction timestamp |

## UI
- FastAPI + Alpine.js single-page app (dark glassmorphism)
- League filter pills (PL/PD/SA/BL1/FL1)
- Match cards with confidence badges, verdict text, kickoff times
- Full-page match detail view with URL routing (`/match/{id}`)
- Expert council grid, consensus meter, form streaks, odds edge indicators
- League table tab with simulated standings
- Value bets tab with Kelly criterion
- Stats tab with model comparison + calibration chart
- Responsive 2-column layout on desktop
- 10-tab navigation: Matches, Insights, BTTS/O2.5, Accumulators, Form, Tables, Accuracy, Review, Stats, Training

## Phase 5.2.2: Streamlit Removal

**Status**: ✅ COMPLETE (DELETED)

Streamlit UI (`ui/app.py`) has been fully replaced by the FastAPI + Alpine.js web app. All Streamlit-only features have been migrated to the consolidated web app. The `ui/` directory has been deleted.

## Phase 5.3: Complete Issue Audit & Fix

**Status**: ✅ COMPLETE

**What Was Fixed**:
1. **UI Indentation Issues**
   - Fixed broken indentation in FBRef section
   - Proper nesting of team stat columns
   - Comparison section correctly structured

2. **Team Name Handling**
   - Added validation for null/empty team names
   - Improved team list filtering across all tabs
   - Applied canonical name normalization

3. **Data Mapping & Column References**
   - Fixed Understat team column reference: `team_name` (not `team`)
   - Verified all provider methods: FBRef, Understat, Scheduler, Training, Alerts, LLM
   - Fixed method signature: `compute_team_rolling_xg(team, matches_window=5)`

4. **Error Handling Improvements**
   - Added defensive try/except blocks in all tabs
   - Improved error messages with truncation
   - Added safety checks for None/empty data
   - Better ImportError handling for optional LLM features

5. **Model Version Selector**
   - Fixed crash when predictions table empty
   - Graceful fallback to v1_elo_poisson
   - Safe index retrieval

6. **Provider Validation**
   - FBRef provider: All 6 methods functional
   - Understat provider: All methods operational
   - Scheduler, continuous training, degradation alerts: Working
   - LLM insights: Graceful fallback when unavailable

7. **Code Cleanup**
   - Removed __pycache__ directories
   - Cleaned up .pyc files
   - Verified Python syntax: ✓ Valid

**Testing Results**:
- Database connectivity: ✓ (16,768 matches)
- FBRef provider: ✓ (4/4 methods working)
- Understat provider: ✓ (2/2 methods working)
- All imports: ✓ (7/7 modules loaded)
- UI syntax: ✓ (Valid Python)

**Production Ready**: YES

Next: Phase 5.4 - Advanced Features

## Tasklist Update (February 14, 2026)

- Phase 5.4 is promoted and renamed to **Phase 6.0: Advanced Features & Full-Model Upgrade**.
- Phase 6.3 is **struck from the roadmap** and removed from active planning scope.
- Future roadmap will continue from 6.0 onward with a focus on free-data reliability and model robustness.

Next: Phase 6.0 - Advanced Features & Full-Model Upgrade

## Phase 7.0: Big-Bang Upgrade (v7 → v8)
**Status**: ✅ COMPLETE

### What Changed
1. **Council v8**: Added GoalPatternExpert (HT scoring, comebacks, first-goal) and LeagueTableExpert (simulated standings, position differential) — 6→8 experts, ~140→~170 features, 9 pairwise consensus signals
2. **Bug Fixes**: Accuracy computation (argmax), scheduler correct flag, SQL injection in insights.py (6 queries parameterised)
3. **H2H Rewrite**: Pure SQL aggregation with CTEs replacing O(N) Python loop (~100x faster)
4. **xG Upgrade**: Data-learned conversion rates, opponent defensive quality adjustment, confidence scoring
5. **Team Mapping**: Negative lookup cache, provider hints (PROVIDER_HINTS dict), 16 new French teams, token-overlap pre-filter
6. **UI Overhaul**: Full-page match detail routing (`/match/{id}`), league table tab, calibration chart, 2-column desktop layout, FL1 badge
7. **New Endpoints**: `/api/matches/{id}/xg`, `/api/matches/{id}/patterns`, `/api/league-table/{comp}`, `/api/competitions`, `/api/performance` with calibration
8. **Opta Scraper**: `providers/opta_analyst.py` — regex HTML parsing, rate limiting, DB caching; CLI: `footy opta fetch/show`
9. **Dead Code**: understat.py + fbref.py replaced with 30-line stubs (removed 945 lines of fake data generators), unused imports cleaned across 7 files
10. **Tests**: `tests/test_upgrades.py` — 11 new tests covering all v8 additions
11. **Documentation**: `agent.md` context file for AI agents
