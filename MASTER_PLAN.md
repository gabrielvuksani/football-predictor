# Footy Predictor Master Plan

## Current State (February 2026)

**Primary Model**: v7_council — Expert Council with 6 specialists + meta-learner
- Logloss: 0.951 | Brier: 0.563 | Accuracy: 55.1% | ECE: 0.040 | 140 features
- Trained on 11,745 matches across 4 leagues (PL, PD, SA, BL1)
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
v3_gbdt_form   ─┼─── v5_ultimate (retired) ──► v7_council (current)
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
| **v7_council** | **Expert Council** | **140** | **0.951** | **Primary** |

## v7 Council Architecture

**6 Specialist Experts**:
1. **EloExpert** — Team-specific home advantage, dynamic K-factor, momentum, volatility
2. **MarketExpert** — Multi-tier odds (closing > avg > max), line movement, O/U 2.5, source quality
3. **FormExpert** — OAF (Opposition-Adjusted Form), venue-split PPG, BTTS, CS, streaks, shot-on-target ratio
4. **PoissonExpert** — Venue-split attack/defense EMA, BTTS/O2.5/O1.5 from score matrix, most-likely-score, goal-diff skewness
5. **H2HExpert** — Bayesian Dirichlet prior, time-decayed observations (half-life 730d), venue-specific sub-analysis
6. **ContextExpert** — Rest days, congestion (7/14/30d), season progress, day-of-week, weekend/midweek, short-rest flags

**Consensus Layer**: Expert variance, spread, 5 pairwise agreements, max disagreement, winner vote concentration, confidence-weighted ensemble, entropy

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
| GET | `/api/stats` | Database statistics |

## UI
- FastAPI + Alpine.js single-page app (dark glassmorphism)
- League filter pills (PL/PD/SA/BL1)
- Match cards with confidence badges, verdict text, kickoff times
- Detail panel: expert council grid, consensus meter, form streaks, odds edge indicators
- Value bets tab with Kelly criterion
- Stats tab with model comparison

## Phase 5.2.2: Root Process Port Blocking Issue

**Problem**: Streamlit process running as root (PID 100830) is blocking port 8501 and cannot be killed without persistent sudo access.

**Root Cause**: 
- Streamlit was previously started with `sudo streamlit run ...`
- Process inherited root ownership
- Interactive sudo password prompts don't work reliably in non-TTY environments
- The stuck process survives normal kill attempts

**Solutions** (in order of preference):

### Option A: Reboot Machine (Nuclear Option)
```bash
sudo reboot
# After reboot, always run: ./clear_port_8501.sh  
# Then: streamlit run ui/app.py
```
**Time**: ~2 minutes
**Success Rate**: 100%

### Option B: Configure Passwordless Sudo (Recommended)
This prevents future password prompts:

```bash
# Enable killall for current user without password prompt
echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/killall" | sudo tee /etc/sudoers.d/streamlit-cleanup
sudo chmod 0440 /etc/sudoers.d/streamlit-cleanup

# Then run:
./clear_port_8501.sh
streamlit run ui/app.py
```
**Time**: 1 minute one-time setup
**Success Rate**: 100%

### Option C: Connect Over SSH with Sudo
If terminal environment is causing password issues:

```bash
# In a new terminal:
ssh $USER@localhost "sudo killall -9 streamlit; sleep 1; streamlit run ui/app.py"
```
**Time**: 2 minutes
**Success Rate**: ~80%

### Option D: Manual Port Changing
Run Streamlit on a different port:

```bash
# Kill stuck process (try multiple times):
sudo killall streamlit python3
sudo killall -9 streamlit python3

# Run on different port:
streamlit run ui/app.py --server.port 8502
# Then access via http://localhost:8502
```
**Time**: 30 seconds
**Success Rate**: ~60%

**Recommended Action for Now**:
```bash
# Try Option A (reboot) - most reliable
sudo reboot

# After reboot:
cd ~/football-predictor
./clear_port_8501.sh
streamlit run ui/app.py
```


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
