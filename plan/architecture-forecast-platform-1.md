---
goal: Upgrade Footy Predictor into a research-backed, self-learning, production-ready football forecasting platform
version: 1.1
date_created: 2026-03-21
last_updated: 2026-03-21
owner: GitHub Copilot
status: In progress
tags: [architecture, upgrade, forecasting, ui, reliability, self-learning, bayesian]
---

# Introduction

![Status: In progress](https://img.shields.io/badge/status-In%20progress-yellow)

This plan captures the production upgrade path for the football forecasting platform after the current stabilization pass. It combines implemented work from this session with the next deterministic phases required to maximize predictive quality, improve UX, and harden the system for scalable zero-cost deployment.

## 1. Requirements & Constraints

- **REQ-001**: Preserve a zero-developer-cost operating model using free/open-source tooling and free data sources already integrated into the repository.
- **REQ-002**: Keep the primary prediction system probabilistic; all major outputs must expose calibrated probabilities rather than label-only predictions.
- **REQ-003**: Support continuous self-learning from scored predictions without requiring manual retraining each round.
- **REQ-004**: Maintain or improve full test-suite pass rate after every major change.
- **REQ-005**: Surface model health, drift, and expert performance in the web UI.
- **REQ-006**: Keep training and inference feature parity for upcoming-match predictions.
- **REQ-007**: Prevent batch contamination when scoring multiple upcoming matches in the same inference pass.
- **SEC-001**: Avoid secrets or paid APIs in the default runtime path.
- **SEC-002**: Keep API error responses safe and non-leaky.
- **CON-001**: The stack must remain runnable in the existing Python/FastAPI/DuckDB environment.
- **CON-002**: The implementation must remain compatible with the current project test harness.
- **GUD-001**: Prefer sequential or state-space updates over repeated global re-estimation when equivalent predictive quality can be achieved.
- **GUD-002**: Prefer interpretable, backtestable model fusion over opaque one-off heuristics.
- **PAT-001**: Treat every expert as a modular probability generator plus feature emitter.
- **PAT-002**: Treat bookmaker probabilities as calibration anchors and market-signal features, not as the sole source of truth.

## 2. Implementation Steps

### Implementation Phase 1

- GOAL-001: Stabilize the upgraded council stack and expose the self-learning system.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Fix expert inference contamination by preventing unfinished matches from mutating rolling expert state in `src/footy/models/experts/*`. | ✅ | 2026-03-21 |
| TASK-002 | Register and ship `NetworkStrengthExpert` in `src/footy/models/experts/network.py`. | ✅ | 2026-03-21 |
| TASK-003 | Register and ship `ZIPExpert` in `src/footy/models/experts/zip_expert.py`. | ✅ | 2026-03-21 |
| TASK-004 | Extend `src/footy/continuous_training.py` with expert-performance tracking and rankings. | ✅ | 2026-03-21 |
| TASK-005 | Expose expert rankings in `web/api.py` and surface them in the Training tab of `web/templates/index.html`. | ✅ | 2026-03-21 |
| TASK-006 | Repair hard-coded expert-count references in tests, README, and operational messaging. | ✅ | 2026-03-21 |

### Implementation Phase 2

- GOAL-002: Improve user-facing UX, accessibility, and model transparency.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-007 | Add keyboard-accessible skip link and focus-visible styles in `web/static/style.css` and `web/templates/index.html`. | ✅ | 2026-03-21 |
| TASK-008 | Add team search input and filter logic in `web/templates/index.html` and `web/static/app.js`. | ✅ | 2026-03-21 |
| TASK-009 | Keep Training tab aligned with drift, retraining history, and self-learning expert rankings. | ✅ | 2026-03-21 |
| TASK-010 | Add a dedicated Model Lab / Engine tab showing ensemble weights, top experts by competition, and latest drift verdicts. | ✅ | 2026-03-21 |
| TASK-011 | Add richer empty/error/loading states for API-dependent tabs and shared retry affordances. |  |  |
| TASK-012 | Add mobile-first responsive refinements for crowded nav states and large stat grids. | ✅ | 2026-03-21 |

### Implementation Phase 3

- GOAL-003: Upgrade the forecasting core using the strongest research-backed free methods.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-013 | Implement a Gamma-Poisson Bayesian state-space scorer in `src/footy/models/advanced_math.py` or `src/footy/models/state_space.py` using sequential conjugate updates and forgetting factors. |  |  |
| TASK-014 | Add a bivariate negative-binomial / shared-Gamma random-effect score model for overdispersion and positive score correlation. |  |  |
| TASK-015 | Add diagonal-inflated draw modelling or zero-inflated Skellam support for draw-heavy leagues. |  |  |
| TASK-016 | Learn league-specific and season-specific forgetting factors, home-advantage priors, and dispersion parameters from walk-forward RPS optimization. |  |  |
| TASK-017 | Add promoted-team cold-start priors using previous-season league strength, market value, and transfer delta. |  |  |
| TASK-018 | Add per-competition calibration layers (temperature / isotonic / Platt) and calibrate both 1X2 and ancillary markets such as BTTS and O2.5. |  |  |

### Implementation Phase 4

- GOAL-004: Unify all data into a disciplined multi-layer ensemble rather than ad hoc feature stacking.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-019 | Split model fusion into Layer A score models, Layer B rating models, Layer C context models, Layer D market calibration, and Layer E meta-learner in `src/footy/models/council.py`. |  |  |
| TASK-020 | Compute per-layer reliability scores and use them as meta-features during stacking. |  |  |
| TASK-021 | Add disagreement, entropy, and epistemic-uncertainty features to distinguish confident consensus from fragile consensus. |  |  |
| TASK-022 | Add competition-conditioned expert weights using the tracked `expert_performance_by_comp` table. | ✅ | 2026-03-21 |
| TASK-023 | Persist prediction decomposition snapshots so every predicted match can explain which models moved the final simplex and by how much. |  |  |

### Implementation Phase 5

- GOAL-005: Institutionalize model governance, backtesting, and reliability engineering.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-024 | Add an explicit backtest module that computes ROI, RPS, Brier, log loss, calibration, CLV proxies, and draw-specific diagnostics by competition and season. |  |  |
| TASK-025 | Add residual diagnostics for lopsided-score outliers, zero-misspecification, and score-correlation misspecification. |  |  |
| TASK-026 | Add regression tests for new experts, expert-ranking persistence, and training-status API payloads. | ✅ | 2026-03-21 |
| TASK-027 | Add data freshness and provider-health monitors to the scheduler and web health surfaces. | Partial | 2026-03-21 |
| TASK-028 | Add reproducible model cards documenting training window, hyperparameters, metrics, and deployment reason. |  |  |

### Implementation Phase 6

- GOAL-006: Prepare the product for scalable production review.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-029 | Add caching and pagination strategy for heavy API views in `web/api.py`. |  |  |
| TASK-030 | Split monolithic frontend state in `web/static/app.js` into domain modules or Alpine stores. |  |  |
| TASK-031 | Normalize docs and agent-facing architecture files to the current 19-expert council stack. | ✅ | 2026-03-21 |
| TASK-032 | Add lightweight deployment observability: structured logs, response-time histograms, and job-run summaries. |  |  |
| TASK-033 | Add snapshot exports for static docs/API examples after successful scheduled refreshes. |  |  |

### Implementation Phase 7

- GOAL-007: Eliminate correctness drift between analytics, APIs, and operational tooling.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-034 | Ensure analytics helpers in `src/footy/llm/insights.py` always filter by the active deployed model version. | ✅ | 2026-03-21 |
| TASK-035 | Replace the broken `CouncilPredictor` CLI backtest dependency with a supported `prepare_training_data()` pathway in `src/footy/models/council.py`. | ✅ | 2026-03-21 |
| TASK-036 | Fix training-history API usage of `training_date` and normalize UI/CLI improvement labels to percentage display. | ✅ | 2026-03-21 |
| TASK-037 | Prevent fallback expert-breakdown recomputation from contaminating sequential expert state using dummy 0-0 results. | ✅ | 2026-03-21 |

## 3. Alternatives

- **ALT-001**: Replace the current council with a single deep-learning model. Rejected because it increases operational cost, reduces interpretability, and is harder to keep robust on small league-specific datasets.
- **ALT-002**: Depend on paid event data or commercial odds feeds. Rejected because it violates the zero-cost requirement.
- **ALT-003**: Keep weighted-likelihood Poisson as the sole primary model. Rejected because recent state-space and dynamic ZIP literature suggests better calibration and update efficiency.
- **ALT-004**: Move immediately to Stan/HMC for the online path. Rejected for now because exact Bayesian inference is valuable offline but too expensive as the default production update loop.

## 4. Dependencies

- **DEP-001**: Existing Python runtime and `.venv` in repository root.
- **DEP-002**: DuckDB schema in `src/footy/db.py`.
- **DEP-003**: Existing council orchestration in `src/footy/models/council.py`.
- **DEP-004**: Existing expert result persistence in `expert_cache`.
- **DEP-005**: Free data providers already documented in `web/templates/index.html` and `README.md`.

## 5. Files

- **FILE-001**: `src/footy/models/experts/network.py` — network-based rating expert.
- **FILE-002**: `src/footy/models/experts/zip_expert.py` — zero-inflated Poisson expert.
- **FILE-003**: `src/footy/continuous_training.py` — self-learning ensemble and expert-performance tracking.
- **FILE-004**: `src/footy/scheduler.py` — scoring job integration for expert learning.
- **FILE-005**: `web/api.py` — training-status and expert-ranking API surfaces.
- **FILE-006**: `web/static/app.js` — UI filtering and tab state logic.
- **FILE-007**: `web/templates/index.html` — training UI and accessibility improvements.
- **FILE-008**: `web/static/style.css` — accessibility and search styles.
- **FILE-009**: `tests/test_continuous_training.py` — self-learning regression coverage.
- **FILE-010**: `tests/test_upgrades.py` — upgraded expert inventory assertions.
- **FILE-011**: `tests/test_v10_council.py` — council-wide integration assertions.

## 6. Testing

- **TEST-001**: Run `.venv/bin/python -m pytest tests/ -q --tb=short` from repository root and require full pass.
- **TEST-002**: Validate expert self-learning with database-backed tests in `tests/test_continuous_training.py`.
- **TEST-003**: Validate all experts compute valid outputs via `tests/test_v10_council.py`.
- **TEST-004**: Validate training-status API payload composition via API tests if endpoint coverage is expanded.
- **TEST-005**: Add walk-forward regression benchmarks before introducing new state-space or dispersion models.

## 7. Risks & Assumptions

- **RISK-001**: Adding many experts can increase correlation between model errors if fusion is not re-layered and recalibrated.
- **RISK-002**: Dynamic Bayesian models can overfit volatility unless forgetting factors are competition-specific and walk-forward tuned.
- **RISK-003**: Free scraped data sources may change structure and break silently without stronger provider-health checks.
- **RISK-004**: UI tab sprawl may reduce usability unless information architecture is consolidated.
- **ASSUMPTION-001**: The existing DuckDB-backed workflow remains acceptable for single-node production use.
- **ASSUMPTION-002**: Free data sources remain sufficient for baseline production value when combined with stronger modeling and calibration.
- **ASSUMPTION-003**: The current meta-learner remains a useful fusion layer while deeper state-space modules are phased in incrementally.

## 8. Related Specifications / Further Reading

- `MASTER_PLAN.md`
- `AUDIT_REPORT.md`
- Ridall, Titman, Pettitt (2025), Bayesian state-space models for English Premier League football.
- Ribeiro et al. (2025), dynamic zero-inflated Poisson for Brazilian Championship forecasting.
- `footBayes` rapid guide (2025) for practical Bayesian football-model comparisons.
- `penaltyblog` documentation for model families, ratings, backtesting, and implied-odds utilities.
