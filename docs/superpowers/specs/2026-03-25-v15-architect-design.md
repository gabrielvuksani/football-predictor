# v15 "Architect" — Prediction Engine Rebuild Spec

## Goal

Rebuild the prediction pipeline to fix 4 data leakage bugs, connect the disconnected self-learning system, wire up unused data sources, remove hardcoded weights in favor of learned parameters, and add new cutting-edge methods — producing genuinely accurate, self-improving, upset-aware predictions.

## Current State (v14 Apex)

- 50 experts, 115 features, CatBoost+XGBoost+HistGBM stacking ensemble
- Reported 56.4% accuracy — **inflated by 4 data leakage vectors**
- Self-learning system (Hedge, drift detection, per-league temperature) computes everything but **feeds nothing back into predictions**
- ~10 experts return flat 33/33/33 due to missing data or broken SQL joins
- BTTS/O2.5/Asian Handicap odds queried but never used
- News data ingested but never joined to matches
- ADWIN drift detector mathematically incomplete (variance discarded)
- All copula theta, HMM emission, GAS parameters hardcoded — should be learned

## Sub-Projects (ordered by dependency)

### Sub-Project 1: Fix Foundations (Critical — unlocks everything else)
1. Fix stacking meta-learner leakage (train on OOF predictions, not test set)
2. Fix temperature scaling leakage (separate calibration split)
3. Fix walk-forward CV leakage (re-run experts per fold with temporal cutoff)
4. Remove 2%/95% probability clipping (or raise to 1%/99%)
5. Prune dead experts (XPts, NewsSentiment with broken SQL, etc.) or fix their data
6. Fix V13_FEATURE_NAMES mismatch (97 names vs ~133 actual features)
7. Fix MarketExpert returning 0/0/0 instead of flat 1/3 when odds missing

### Sub-Project 2: Wire Up Unused Data
1. Join `news` table to `_MATCH_COLS` SQL → feed to NewsSentimentExpert
2. Feed BTTS odds as direct feature for BTTS head
3. Feed O2.5 odds as direct feature for O2.5 head
4. Feed Asian Handicap odds as features for main model
5. Feed FPL FDR (Fixture Difficulty Rating) to ContextExpert
6. Fix RefereeExpert SQL column alias mismatch
7. Wire half-time score patterns more deeply

### Sub-Project 3: Learn All Parameters (replace hardcoded with data-driven)
1. HMM: EM algorithm for emission rates and transition matrix
2. GAS: MLE for A, B, omega parameters per league
3. Copula: estimate theta from observed goal correlations per league
4. COM-Poisson: estimate nu from variance/mean ratio per league
5. Dixon-Coles: optimize half-life per league via cross-validation
6. Upset composite: learn weights via logistic regression on historical upsets
7. BTTS/O2.5 heads: separate hyperparameter tuning (currently identical)

### Sub-Project 4: Connect Self-Learning Loop
1. Store Hedge weights in DB, load at prediction time
2. Use expert Hedge weights to scale confidence in feature builder
3. Apply per-league temperature as post-processing
4. Fix ADWIN (add variance computation) or replace with Bayesian Online Changepoint Detection
5. Drift detection → automatic retraining trigger with higher time-decay
6. Expert auto-pruning: disable experts with sustained flat output
7. Persist self-learning state across restarts (currently lost)

### Sub-Project 5: New Stacking Architecture
1. Proper K-fold cross-validated base models producing OOF predictions
2. Stacking meta-learner trained only on OOF features
3. Held-out calibration split for temperature scaling
4. Conformal prediction sets for uncertainty quantification
5. Per-league model specialization (league-specific temperature + calibration)

### Sub-Project 6: Enhanced Upset Detection
1. Learn upset composite weights from historical data
2. Market-model disagreement signal (model vs bookmaker divergence)
3. Regime change signal from BOCD/HMM (team entering vulnerable state)
4. News disruption signal (GDELT negative sentiment spike pre-match)
5. Squad disruption signal (key player injury/suspension)
6. Conformal prediction set size as uncertainty indicator

## Architecture Decisions

### Stacking Without Leakage
```
Training:
  1. Split data: 70% train / 15% calibration / 15% test
  2. Run K-fold CV on train split with base models
  3. Collect out-of-fold predictions as stacking features
  4. Train meta-learner on OOF predictions only
  5. Optimize temperature on calibration split only
  6. Evaluate on test split (never seen by any component)

Prediction:
  1. Run all active experts → features
  2. Scale expert features by Hedge-learned weights
  3. Base models predict
  4. Meta-learner combines base model outputs
  5. Per-league temperature scaling applied
  6. Conformal prediction sets computed
  7. Results stored, self-learning loop scores when results arrive
```

### Self-Learning Feedback Loop
```
Match Played → Score Prediction → Hedge Update → Expert Weights Updated in DB
                                → Drift Detectors Updated
                                → If drift detected: trigger retrain with decay
                                → Per-league temperature gradient descent step
```

### Expert Gating
Instead of running all 50 experts (many producing noise), add a gating mechanism:
- Track per-expert signal quality (Brier score contribution)
- Experts below threshold automatically excluded from feature matrix
- Re-evaluate gating monthly based on rolling performance

## Success Criteria

1. **True accuracy** (after leakage fixes) measurably exceeds bookmaker baseline (53-55%)
2. Self-learning loop demonstrably improves predictions over time (not just tracking)
3. Upset detection precision > 30% at recall > 50% (currently unmeasured)
4. All expert parameters learned from data, not hardcoded
5. Conformal prediction sets achieve 90% empirical coverage
6. Walk-forward CV with proper temporal isolation shows consistent improvement

## Constraints

- Zero cost (no paid APIs for core functionality)
- Must work within GitHub Actions 60-minute timeout
- Must maintain backward compatibility with existing DB schema
- Alpine.js frontend not in scope (backend only)
- iOS app not in scope
