---
goal: Production Upgrade Plan for the Football Forecast Platform
version: 2.0
date_created: 2026-03-21
last_updated: 2026-03-21
owner: GitHub Copilot
status: In progress
tags: [upgrade, architecture, modeling, production, ui, reliability]
---

# Introduction

![Status: In progress](https://img.shields.io/badge/status-In%20progress-yellow)

This specification captures the current production upgrade direction for the football forecasting platform after the March 2026 hardening passes. It combines repo audit findings, external research on modern football forecasting methods, and implemented platform improvements into a single execution-oriented roadmap.

## 1. Requirements & Constraints

- **REQ-001**: Keep the stack zero-developer-cost for core operation using open-source tooling and free/public datasets where possible.
- **REQ-002**: Preserve the current FastAPI + DuckDB + static Pages architecture unless a change clearly improves reliability or maintainability.
- **REQ-003**: Maintain backward compatibility for core endpoints used by the live UI and static exports.
- **REQ-004**: Make model learning persistent, reproducible, and explainable.
- **REQ-005**: Expose one coherent prediction object that unifies outcome, totals, BTTS, scorelines, value edges, and uncertainty.
- **REQ-006**: Support incremental retraining and online updates without replaying the full world on every request.
- **REQ-007**: Keep all improvements test-covered and suitable for CI execution.
- **SEC-001**: Avoid secret-dependent production logic in CI or static export workflows.
- **SEC-002**: Treat all API error responses as user-safe and non-leaky.
- **CON-001**: Use only data sources already present in the project or freely available public sources.
- **CON-002**: Do not require a paid MLOps platform, hosted feature store, or proprietary model registry.
- **GUD-001**: Prefer typed Python interfaces and deterministic JSON payloads over implicit parsing of ad hoc blobs.
- **PAT-001**: Use cached score/probability grids as the source of truth for derived markets.
- **PAT-002**: Use sequential state-space updates and rolling evaluation rather than repeatedly fitting heavyweight static models from scratch.

## 2. Implementation Steps

### Implementation Phase 1

- GOAL-001: Stabilize the live learning loop and prediction serving path.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Persist `ensemble_weights`, `expert_performance`, and `expert_performance_by_comp` in `src/footy/db.py` core schema | ✅ | 2026-03-21 |
| TASK-002 | Reconstruct self-learning state from `prediction_scores`, `predictions`, and `expert_cache` in `src/footy/self_learning.py` | ✅ | 2026-03-21 |
| TASK-003 | Feed expert-level prediction traces into `score_finished_predictions()` in `src/footy/pipeline.py` | ✅ | 2026-03-21 |
| TASK-004 | Trigger persistent expert ranking and ensemble-weight refresh after scoring finished matches | ✅ | 2026-03-21 |
| TASK-005 | Add test coverage for self-learning DB reconstruction and live API endpoints | ✅ | 2026-03-21 |

### Implementation Phase 2

- GOAL-002: Make prediction APIs more coherent, faster, and more production-ready.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-006 | Add a reusable cached Bayesian engine loader in `web/api.py` to avoid full historical replay per request | ✅ | 2026-03-21 |
| TASK-007 | Upgrade `/api/unified-prediction/{match_id}` to combine council, Bayesian, statistical, and market signals when available | ✅ | 2026-03-21 |
| TASK-008 | Expand unified response payload to include prediction sets, intervals, and score probability summaries | ✅ | 2026-03-21 |
| TASK-009 | Align documentation/runtime language from 19 experts to 20 experts across core files | ✅ | 2026-03-21 |
| TASK-010 | Add GitHub Actions CI workflow for full pytest execution | ✅ | 2026-03-21 |
| TASK-035 | Export static JSON for Pages-only tabs (`btts-ou`, `accumulators`, `form-table`, `accuracy`, `round-preview`, `post-match-review`, `training/status`) in `src/footy/cli/pages_cmds.py` | ✅ | 2026-03-21 |
| TASK-036 | Add deterministic regression coverage for the expanded Pages export contract in `tests/test_pages_export.py` | ✅ | 2026-03-21 |

### Implementation Phase 3

- GOAL-003: Upgrade the modeling backbone using research-backed methods that extract more signal from available data.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-011 | Introduce dynamic forgetting-factor calibration for Bayesian state-space updates using season-aware tuning |  |  |
| TASK-012 | Add bivariate negative-binomial / shared-random-effect measurement option for overdispersion and positive goal correlation |  |  |
| TASK-013 | Add dynamic Skellam goal-difference layer for direct 1X2 calibration comparisons |  |  |
| TASK-014 | Add probability-grid service object so all markets derive from one normalized score grid |  |  |
| TASK-015 | Add ranked probability score (RPS) and expected calibration error (ECE) as first-class model-selection metrics |  |  |
| TASK-016 | Add season-transition priors for promoted/relegated teams using competition-aware shrinkage |  |  |

### Implementation Phase 4

- GOAL-004: Use richer football data to improve pre-match, shot-based, and contextual forecasting.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-017 | Extend `src/footy/xg.py` with a hierarchical Bayes-xG path using public shot/event data when available |  |  |
| TASK-018 | Add player-/position-adjusted finishing corrections for xG derived from open StatsBomb-style features |  |  |
| TASK-019 | Build a feature aggregation layer that fuses odds, xG, injuries, weather, referee, market value, and form into named feature groups |  |  |
| TASK-020 | Add missingness-aware confidence penalties when some provider enrichments are unavailable |  |  |
| TASK-021 | Add matchup context tags (derby, relegation fight, European congestion, injury crisis) for context-specific expert weighting |  |  |
| TASK-022 | Add residual-diagnostics tables for outlier scores, runaway matches, and calibration drift by competition |  |  |

### Implementation Phase 5

- GOAL-005: Reduce architectural drift and improve maintainability across web, static export, and deployment workflows.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-023 | Split `web/api.py` into domain routers (`matches`, `insights`, `learning`, `bayesian`, `training`) |  |  |
| TASK-024 | Unify `web/` and `docs/` frontend logic so the static UI is generated from the same product surface instead of a manual fork |  |  |
| TASK-025 | Change Pages deployment workflow to build fresh static JSON before publish when a database snapshot exists | ✅ | 2026-03-21 |
| TASK-026 | Add schema versioning / migration metadata instead of only best-effort column additions |  |  |
| TASK-027 | Introduce a worker/scheduler deployment mode separate from the web API process |  |  |
| TASK-028 | Reduce JSON blob parsing in `predictions.notes` by promoting stable fields into typed columns or materialized views |  |  |

### Implementation Phase 6

- GOAL-006: Lift product quality in UX, reliability, and operator workflows.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-029 | Add explicit loading/error/empty-state patterns to the static Pages UI to match the live UI quality bar |  |  |
| TASK-030 | Add proper ARIA tab semantics and keyboard navigation to the live and static navigation tabs | ✅ | 2026-03-21 |
| TASK-031 | Surface self-learning explanations in UI: why weights changed, which experts are hot/cold, drift reasons, retrain triggers |  |  |
| TASK-032 | Add operator diagnostics page for provider freshness, stale caches, and last successful scoring/training runs |  |  |
| TASK-033 | Add rate limiting and tighter CORS policy options for non-local deployment |  |  |
| TASK-034 | Add end-to-end CLI workflow tests for `footy go`, `footy refresh`, and `footy pages export` |  |  |
| TASK-037 | Add skip-link and error-toast support plus persisted static UI view preferences in `docs/index.html`, `docs/app.js`, and `docs/style.css` | ✅ | 2026-03-21 |

## 3. Alternatives

- **ALT-001**: Replace the current stack with a managed cloud warehouse and hosted serving layer. Rejected because it violates the zero-cost operational constraint.
- **ALT-002**: Move directly to deep neural sequence models for match prediction. Rejected for now because the current public-data regime and explainability goals still favor structured probabilistic models.
- **ALT-003**: Use only bookmaker-implied odds as the final prediction. Rejected because odds should be a calibration prior / market signal, not the whole product.
- **ALT-004**: Keep expert learning fully in-memory. Rejected because it is fragile across restarts and unsuitable for production diagnostics.

## 4. Dependencies

- **DEP-001**: DuckDB remains the primary persistence layer for model state, predictions, and performance tables.
- **DEP-002**: FastAPI remains the primary HTTP serving framework.
- **DEP-003**: Public football result/odds datasets remain necessary for historical and market signals.
- **DEP-004**: Optional public shot/event data (e.g. open StatsBomb-style sources) is required for richer xG upgrades.
- **DEP-005**: GitHub Actions provides no-cost CI for regression protection.

## 5. Files

- **FILE-001**: `src/footy/db.py` — core schema, persistence, indexes.
- **FILE-002**: `src/footy/self_learning.py` — learning reconstruction and expert weighting.
- **FILE-003**: `src/footy/pipeline.py` — scoring feedback loop and persistence triggers.
- **FILE-004**: `web/api.py` — cached Bayesian service and unified prediction API.
- **FILE-005**: `src/footy/prediction_aggregator.py` — unified prediction contract.
- **FILE-006**: `src/footy/models/bayesian_engine.py` — dynamic Bayesian state-space forecasting engine.
- **FILE-007**: `src/footy/models/council.py` — council prediction path and expert payloads.
- **FILE-008**: `.github/workflows/ci.yml` — automated test validation.
- **FILE-009**: `tests/test_self_learning.py` — self-learning regression coverage.
- **FILE-010**: `tests/test_api.py` — API regression coverage.
- **FILE-011**: `.github/workflows/static.yml` — GitHub Pages deployment with optional export regeneration.
- **FILE-012**: `src/footy/cli/pages_cmds.py` — static export parity for Pages tabs.
- **FILE-013**: `docs/index.html` — static UI shell accessibility and error surfacing.
- **FILE-014**: `docs/app.js` — static UI state, tab semantics, and toast behavior.
- **FILE-015**: `docs/style.css` — static accessibility and toast styling.
- **FILE-016**: `docs/404.html` — SPA redirect preserving query/hash state.
- **FILE-017**: `web/templates/index.html` — live tab semantics and keyboard navigation.
- **FILE-018**: `web/static/app.js` — live tab keyboard-navigation helpers.
- **FILE-019**: `tests/test_pages_export.py` — Pages export regression coverage.

## 6. Testing

- **TEST-001**: Full suite must pass locally with `.venv/bin/python -m pytest tests/ -q --tb=short`.
- **TEST-002**: Self-learning reconstruction must rebuild expert weights from `expert_cache` + `prediction_scores`.
- **TEST-003**: Unified prediction endpoint must expose multiple component probability sources and score probabilities.
- **TEST-004**: Bayesian endpoints must operate from cached state without per-request history replay regressions.
- **TEST-005**: CI must execute pytest on pushes and pull requests.
- **TEST-006**: Pages export must generate JSON payloads for static insights, review, and training tabs.

## 7. Risks & Assumptions

- **RISK-001**: The current `predictions.notes` payload still carries too much semantic load; further schema promotion is recommended.
- **RISK-002**: The static Pages UI still lags the live UI in feature breadth until the frontends are unified, even though the most visibly broken tabs now have exported payloads and better error handling.
- **RISK-003**: Bayesian/xG upgrades using richer public event data will increase implementation complexity and may require careful caching.
- **ASSUMPTION-001**: Existing public datasets remain available and legally usable under their current terms.
- **ASSUMPTION-002**: The platform should continue favoring explainable probabilistic methods over opaque end-to-end ML models.
- **ASSUMPTION-003**: FastAPI + DuckDB remains a valid scale target for the expected workload.

## 8. Related Specifications / Further Reading

- `UPGRADE_PLAN.md`
- `AUDIT_REPORT.md`
- Ridall, Titman, Pettitt (2025), *Bayesian state-space models for the modelling and prediction of the results of English Premier League football*.
- Ribeiro et al. (2025), *A Bayesian approach to predict performance in football: a case study*.
- Scholtes & Karakuş (2024), *Bayes-xG: player and position correction on expected goals (xG) using Bayesian hierarchical approach*.
- Karlis & Ntzoufras (2009), *Bayesian modelling of football outcomes using the Skellam distribution for goal difference*.
- PenaltyBlog model documentation on normalized football probability grids and internally consistent derived markets.
