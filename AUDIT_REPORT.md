# Football Predictor — Full Codebase Audit Report

**Date:** 2025-07-17  
**Scope:** 38 source files, 14,913 lines · 15 test files, 1,168 lines · ~318 total files  
**Methodology:** Automated + manual review of every module, every endpoint, every template binding, every CLI command. Live server verified. Static export verified. Tests run. Dead code traced.

---

## A) Executive Summary

1. **The live web UI (`web/`) is solid.** All template bindings match the API response structures. 8 experts render correctly. H2H, form, odds, Elo, Poisson all display properly. Only 1 cosmetic bug (`predicted_score` renders as "1,0" instead of "1 - 0").

2. **The GitHub Pages static export (`pages_cmds.py`) is broken.** 5 bugs produce wrong or empty JSON: wrong column names (`breakdown` → `breakdown_json`, `home_wins` → `team_a_wins`), wrong JSON shapes for form/H2H/performance, and all failures silently swallowed by `except Exception: pass`.

3. **Dixon-Coles data leakage in `council.py`.** DC predictions are generated for all rows including training rows, using a model fit on those same rows. The meta-learner trains on train features that include "cheating" DC predictions, over-weighting DC signals that will be weaker at inference time.

4. **Brier score is computed two different ways.** `council.py`/`pipeline.py` sum the 3-class squared errors; `performance_tracker.py`/`scheduler.py` divide by 3. Cross-system metric comparisons are meaningless.

5. **xG off-target rate formula is algebraically broken.** `goals - sot * (goals/sot)` = 0 always. The off-target conversion rate always returns the floor clamp (0.005). Not currently impactful because the main pipeline uses hardcoded rates, but `compute_xg_advanced()` (called from the web API) is affected.

6. **Continuous training is dead infrastructure.** `auto_retrain()` is never called by the scheduler. The deploy gate accepts 0% improvement. The `check_and_retrain()` function never actually retrains. The configured `performance_threshold_improvement` is never consulted.

7. **1,607 lines of dead code confirmed**: `v5.py` (907), `v3.py` (406), `understat.py` (31), `fbref.py` (32), `stats_cmds.py` (231 — 11 non-functional CLI commands).

8. **12/141 tests fail** — all in `test_api.py` due to DuckDB "cannot launch in-memory database in read-only mode." The test setup doesn't mock `con()`.

9. **No dependency lock file.** Builds are non-reproducible. Two installs on different days may resolve different versions.

10. **Render free tier cannot persist data.** DuckDB file is baked into the Docker image and lost on every deploy. The scheduler can never fire reliably because the process sleeps after 15 minutes.

11. **No API auth, rate limiting, or CORS.** All endpoints are publicly accessible with no protection.

12. **Closing odds (B365CH/CD/CA) create train/serve skew.** The model trains with closing odds available for all historical matches, but at prediction time only opening odds (or none) exist.

---

## B) Prioritized Backlog

### P0 — Must Fix (Data Correctness / Broken Features)

| # | Issue | Impact | Effort | Files | PR Grouping |
|---|-------|--------|--------|-------|-------------|
| P0-1 | **pages_cmds.py: fix `breakdown` → `breakdown_json`** | Expert data export silently empty | 5 min | `cli/pages_cmds.py` | PR: "Fix static export" |
| P0-2 | **pages_cmds.py: fix h2h column names** (`home_wins` → `team_a_wins`, etc.) | H2H export silently empty | 10 min | `cli/pages_cmds.py` | Same PR |
| P0-3 | **pages_cmds.py: fix form JSON structure** (nested `{home: {results}}` → flat `{home_form: [], home_ppg: N}`) | Form shows blank on GH Pages | 15 min | `cli/pages_cmds.py` | Same PR |
| P0-4 | **pages_cmds.py: fix h2h JSON structure** (flat dict → `{stats: {}, recent_matches: []}`) | H2H panel broken on GH Pages | 15 min | `cli/pages_cmds.py` | Same PR |
| P0-5 | **pages_cmds.py: fix perf JSON structure** (`{models: []}` → `{metrics, calibration, by_competition, recent}`) | Perf page broken on GH Pages | 30 min | `cli/pages_cmds.py` | Same PR |
| P0-6 | **pages_cmds.py: add missing `elo`, `poisson`, `score` to match detail** | Match detail incomplete on GH Pages | 20 min | `cli/pages_cmds.py` | Same PR |
| P0-7 | **pages_cmds.py: add missing `implied_h/d/a`, `overround` to odds; remove raw `hs/as_/hst/ast`** | Odds panel broken on GH Pages | 15 min | `cli/pages_cmds.py` | Same PR |
| P0-8 | **pages_cmds.py: remove `except Exception: pass`** — at minimum log the error | Bugs silently swallowed | 10 min | `cli/pages_cmds.py` | Same PR |
| P0-9 | **Fix 12 API test failures** — mock `con()` in test_api.py to return in-memory DB (non-read-only) | Tests unreliable | 30 min | `tests/test_api.py` | PR: "Fix test suite" |
| P0-10 | **Standardize Brier score** — choose ÷3 (standard) or not, apply everywhere | Metrics comparison broken | 30 min | `council.py`, `pipeline.py`, `utils.py`, `performance_tracker.py`, `scheduler.py` | PR: "Standardize metrics" |

### P1 — Should Fix (Correctness / Reliability)

| # | Issue | Impact | Effort | Files | PR Grouping |
|---|-------|--------|--------|-------|-------------|
| P1-1 | **Dixon-Coles data leakage** — zero DC features for training rows, or use leave-one-out | Meta-learner over-weights DC | 2 hr | `council.py` L1428 | PR: "Fix DC leakage" |
| P1-2 | **xG off-target rate formula** — `off_target_rate = max(0.005, (goals - sot * on_target_rate) / off_target)` but `on_target_rate = goals/sot` makes numerator 0 | xG accuracy degraded (advanced path) | 20 min | `xg.py` L97-100 | PR: "Fix xG formula" |
| P1-3 | **api.py H2H connection leak** — `con()` called twice at L282 | DB connection leak | 5 min | `web/api.py` | PR: "API fixes" |
| P1-4 | **Continuous training deploy gate** — change `>= 0` to use `performance_threshold_improvement` | Regressions silently deployed | 15 min | `continuous_training.py` L381 | PR: "Fix deploy gate" |
| P1-5 | **Pipeline silent failures** — `try/except` on steps 1-3 and 6-8 catches all errors | Pipeline silently incomplete | 30 min | `pipeline.py` | PR: "Pipeline error handling" |
| P1-6 | **Closing odds train/serve skew** — add feature flag or use opening odds only | Prediction degradation | 1 hr | `council.py` MarketExpert | PR: "Fix odds features" |
| P1-7 | **predicted_score display** — join array with " - " | Cosmetic but visible | 5 min | `web/templates/index.html` L462 | PR: "UI polish" |
| P1-8 | **Wire auto_retrain to scheduler** — add `retrain` job type | Continuous training dead | 1 hr | `scheduler.py`, `continuous_training.py` | PR: "Enable auto-retrain" |
| P1-9 | **Generate lock file** — `pip-compile` or `uv lock` | Non-reproducible builds | 15 min | `requirements.lock` or `uv.lock` | PR: "Lock deps" |

### P2 — Nice to Have (Quality / Polish)

| # | Issue | Impact | Effort | Files | PR Grouping |
|---|-------|--------|--------|-------|-------------|
| P2-1 | **Delete dead code** — v5.py, v3.py, understat.py, fbref.py, cache.py dead functions, stats_cmds.py | 1,607 lines of noise | 30 min | 6 files | PR: "Dead code cleanup" |
| P2-2 | **Fix stale comments** — "6 experts" → "8 experts" (council.py L1751, L1253, L1275) | Misleading docs | 15 min | `council.py` | PR: "Comment cleanup" |
| P2-3 | **Add CORS middleware** | Browser security | 10 min | `web/api.py` | PR: "API security" |
| P2-4 | **Add API rate limiting** (slowapi) | Abuse protection | 30 min | `web/api.py` | Same PR |
| P2-5 | **Multi-stage Dockerfile** | Image size reduction | 30 min | `Dockerfile` | PR: "Docker improve" |
| P2-6 | **Remove data/ COPY from Dockerfile** — use volume only | Stale baked data | 10 min | `Dockerfile` | Same PR |
| P2-7 | **openMatch() race condition** — debounce or abort previous | UI glitch on fast clicks | 30 min | `web/static/app.js` | PR: "UI robustness" |
| P2-8 | **Add error states to UI** — show "Failed to load" instead of blank/misleading text | UX | 45 min | `web/templates/index.html`, `web/static/app.js` | Same PR |
| P2-9 | **Accessibility** — ARIA roles on tabs, skip-to-content, contrast fixes | a11y | 1 hr | `web/templates/index.html`, `web/static/style.css` | PR: "a11y" |
| P2-10 | **Standings CSS 9-col → 8-col** | Layout glitch | 5 min | `web/static/style.css` | PR: "UI polish" |
| P2-11 | **Add `/api/health` endpoint** checking DB connectivity | Better health checks | 15 min | `web/api.py` | PR: "API improvements" |
| P2-12 | **Stale static JSON cleanup** — delete old match dirs before export | Disk bloat on Pages | 15 min | `cli/pages_cmds.py` | PR: "Fix static export" |
| P2-13 | **Integrate pages export into pipeline** — call after `footy go`/`footy refresh` | Manual step forgotten | 10 min | `pipeline.py` | Same PR |
| P2-14 | **N+1 queries in match detail** — JOIN into 1-2 queries | API latency | 1 hr | `web/api.py` | PR: "API perf" |

---

## C) Bugs & Breakages

### Confirmed Bugs (repro steps + evidence)

| # | Bug | File:Line | Repro | Evidence | Fix Direction |
|---|-----|-----------|-------|----------|---------------|
| BUG-1 | Expert export uses wrong column `breakdown` | `pages_cmds.py:166` | `footy pages export` → `docs/api/matches/<id>/experts.json` not created | Table schema: column is `breakdown_json`, not `breakdown`. DuckDB throws error, caught by `except Exception: pass`. | Change `"SELECT breakdown"` → `"SELECT breakdown_json"` |
| BUG-2 | H2H export uses wrong columns | `pages_cmds.py:179-182` | Same as above → `h2h.json` not created | Table columns: `team_a_wins`, `team_b_wins`, `team_a_avg_goals_for`, etc. Code uses: `home_wins`, `away_wins`, `home_goals_avg`, `away_goals_avg`. | Update column names to match `h2h_stats` schema |
| BUG-3 | Form JSON structure mismatch | `pages_cmds.py:207-224` | `footy pages export` → form.json has `{home: {team, results, ppg}}` | Live API returns `{home_form: [], home_ppg: N}`. Template binds `matchForm.home_form` and `matchForm.home_ppg`. Static export creates `{home: {team, results, ppg}}` which doesn't match. | Restructure output to `{home_form: results, home_ppg: ppg, away_form: results, away_ppg: ppg}` |
| BUG-4 | H2H JSON structure mismatch | `pages_cmds.py:179-200` | Even if SQL fixed, produces flat `{total_matches, home_wins, recent: [...]}` | Live API returns `{stats: {...}, recent_matches: [...]}`. Template binds `matchH2H.stats.team_a_wins`. | Wrap in `{stats: {…}, recent_matches: [...]}` |
| BUG-5 | Performance JSON structure mismatch | `pages_cmds.py:_build_performance` | `footy pages export` → performance.json has `{models: [...]}` | Live API returns `{metrics: {}, calibration: [], by_competition: [], recent: []}`. Template binds those keys. | Rewrite `_build_performance` to match API output |
| BUG-6 | Match detail missing `elo`, `poisson`, `score` | `pages_cmds.py:131-160` | Static `<id>.json` vs live `/api/matches/<id>` | Live has `elo: {home, away, diff}`, `poisson: {btts, o25, lambda_home, lambda_away, predicted_score}`, `score: {outcome, logloss, brier, correct}`. Static has none. | Add Elo query from elo_ratings, Poisson from poisson_predictions, score from prediction_scores |
| BUG-7 | Match detail has raw `hs/as_/hst/ast` keys | `pages_cmds.py:145-146` | Leaks shot stats as top-level keys not used by template | Template doesn't reference these. | Remove from query or restructure into a nested object |
| BUG-8 | Match detail odds missing `implied_*`, `overround` | `pages_cmds.py:155-158` | Static odds: `{b365h, b365d, b365a}` | Live API odds: `{b365h, b365d, b365a, implied_h, implied_d, implied_a, overround}`. Template binds all of them. | Compute implied = 1/odds, overround = sum(implied) |
| BUG-9 | xG off-target rate always 0 | `xg.py:104` | `goals - sot * (goals/sot) = goals - goals = 0` | Formula: `on_target_rate = goals/sot`, then `goals - sot * on_target_rate = 0`. Clamped to 0.005. | Fix: compute off_target_goals = total_goals_from_off_target_shots separately, or use `sot * on_target_rate` and `(shots-sot) * off_target_rate = remaining_goals` solved for off_target_rate |
| BUG-10 | Brier score inconsistency | `council.py:1482` vs `performance_tracker.py:156` | Compare Brier from training log vs performance tracker | Training uses `np.mean(np.sum(...))` (≈0.6-0.7), perf tracker uses `sum(...)/3` (≈0.2). Same predictions, different "Brier". | Pick one formula, apply everywhere |
| BUG-11 | Dixon-Coles leakage | `council.py:1428-1435` | DC model fit on train rows, then predictions generated for same train rows → meta-learner trains on biased features | DC predictions on train data will be artificially accurate; meta-learner over-trusts DC at inference time. | Zero out DC features for train rows, or only generate for test rows |
| BUG-12 | api.py H2H double `con()` | `web/api.py:282` | Every H2H request leaks a DB connection | `db = con()` then `db = con()` again 4 lines later — first connection never closed. | Remove duplicate `con()` call |
| BUG-13 | Deploy gate too lenient | `continuous_training.py:381` | `improved = record.get("improvement_pct", 0) >= 0` | 0% improvement models deploy. `performance_threshold_improvement=0.01` (1%) is configured but never used. | Change `>= 0` → `>= self.config['retraining_schedules'][...]['performance_threshold_improvement']` |
| BUG-14 | Pipeline silent failures | `pipeline.py` steps 1-3, 6-8 | Any step fails → yellow warning printed, pipeline continues with stale data | `except Exception as e: console.print(f"[yellow]⚠ {step} skipped: {e}")` | Re-raise critical failures (at least log full traceback) |
| BUG-15 | 12 API tests fail | `tests/test_api.py` | `pytest tests/test_api.py` | `duckdb.InvalidInputException: Cannot launch in-memory database in read-only mode` | Mock `con()` to return non-read-only in-memory DB, or use temp file |
| BUG-16 | predicted_score renders "1,0" | `web/templates/index.html:462` | Open any match with Poisson data | JS `Array.toString()` on `[1, 0]` → `"1,0"` | Use `md.poisson.predicted_score?.join(' - ')` |

### Likely Issues (strong evidence, not directly reproduced)

| # | Issue | Risk | Evidence |
|---|-------|------|----------|
| LI-1 | Closing odds train/serve skew | Model trained on closing odds but predicts with opening odds (or none) | MarketExpert priority 4.0 (highest) for B365CH/CD/CA closing odds. At prediction time, only B365H/D/A (opening) may be available. |
| LI-2 | `check_and_retrain()` misleading name — never retrains | Developer confusion | Function only checks drift metrics and returns status dict. Docstring doesn't clarify. |
| LI-3 | DuckDB single-writer contention | Scheduler + pipeline could conflict | Both APScheduler jobs and CLI pipeline write to same DuckDB file. DuckDB allows only 1 writer at a time. |
| LI-4 | stats_cmds.py: 2 commands crash with KeyError | `footy stats provider-report`, `footy stats db-size` crash | Keys like `"api_football"`, `"football_data_co_uk"` don't exist in the returned dict |

---

## D) Refactor Plan

### Tier 1: Targeted Fixes (no architecture change)

| Module | What | Migration |
|--------|------|-----------|
| `cli/pages_cmds.py` | Rewrite `_export_match_detail` and `_build_performance` to match live API response shapes | Direct rewrite. Test by comparing `footy pages export` output with `curl` against live API. |
| `xg.py` L97-100 | Fix off-target rate: either use separate data source for off-target goals, or change formula to `off_target_rate = (total_goals - sot * conversion_rate) / max(off_target_shots, 1)` where conversion_rate is independently estimated | Algebraic fix. Verify with unit test. |
| `web/api.py` L282 | Remove duplicate `con()` call | 1-line fix. |

### Tier 2: Module-Level Improvements

| Module | What | Migration |
|--------|------|-----------|
| `council.py` | Fix DC leakage by masking DC features to 0.0 for training indices | Add `dc_probs[train_idx] = np.array([1/3, 1/3, 1/3])` before meta-learner training. Test: compare train vs test Brier with/without DC fix. |
| `continuous_training.py` | Wire to scheduler, fix deploy gate, rename `check_and_retrain` | Add `retrain` job type in scheduler.py. Change gate to `>= threshold`. Add docstring clarity. |
| `pipeline.py` | Replace bare `except Exception` with specific exception handling. Add `--strict` flag. | Keep try/except for non-critical steps (h2h, xg) but let core steps (ingest, train, predict) crash. Log full tracebacks. |
| `tests/test_api.py` | Mock `con()` with non-read-only in-memory DuckDB + seed schema | Use `@pytest.fixture` that creates temp DuckDB, creates all 16 tables, and patches `web.api.con`. |

### Tier 3: Structural Cleanup

| Module | What | Migration |
|--------|------|-----------|
| `v5.py` (907 lines) | Delete | Confirm zero callers ✓, no model file on disk ✓. Delete file + remove from `__init__.py` if referenced. |
| `v3.py` (406 lines) | Delete | Same as above. |
| `understat.py`, `fbref.py` | Delete or implement | Provider stubs that always return None. Either implement or delete. |
| `stats_cmds.py` (231 lines) | Delete or fix | 0/11 commands work. Either fix the data sources or remove the CLI group. |
| `cache.py` dead functions | Remove `cache_prediction_batch`, `lookup_cached_predictions` | Zero callers confirmed. |

### Tier 4: Architecture (future)

| Area | What | Notes |
|------|------|-------|
| Brier score | Standardize to `1/K * Σ(p-o)²` everywhere | Touch 5 files, but simple search-replace. Add a shared `brier_score()` util. |
| Walkforward validation | Create `walkforward.py` with expanding-window CV | Currently only single train/test split exists. |
| DB migration | Consider SQLite/PostgreSQL for multi-writer | DuckDB is single-writer; scheduler + pipeline conflict risk. |

---

## E) Documentation Gaps

| Gap | Where to Add | Priority |
|-----|-------------|----------|
| **API endpoint docs** — no OpenAPI/Swagger or endpoint reference | Add `summary` and `description` to every FastAPI route decorator. FastAPI auto-generates `/docs`. | P1 |
| **Architecture decision records** — why DuckDB? Why 8 experts? Why Brier/3? | Create `docs/architecture.md` or `ADR/` folder | P2 |
| **Data dictionary** — 16 tables, no schema docs | Auto-generate from DuckDB `DESCRIBE` into `docs/schema.md` | P1 |
| **Model card** — accuracy, calibration, known biases, training data range | Create `docs/model-card.md` per ML best practices | P2 |
| **Deployment guide** — how to actually deploy to Render/Railway/Docker | Expand README deployment section with step-by-step | P1 |
| **GitHub Pages export docs** — when to run, what it produces, how it works | Add docstring to `pages_cmds.py` + section in README | P1 |
| **CLI reference** — 60 commands, no reference doc | Auto-generate with `typer utils docs` or custom script | P2 |
| **MASTER_PLAN.md is stale** — references "v7 upgrade" as upcoming, v8 already shipped | Update to reflect current state | P1 |
| **`council.py` inline docs** — expert feature descriptions | Add docstrings to each Expert class explaining what features it contributes and what ranges they have | P2 |
| **Stale comments** — "6 experts" (L1751, L1253, L1275 in council.py), "Called by scheduler periodically" (continuous_training.py L69) | Fix inline | P1 |

---

## F) Quick Wins (each < 1 day)

| # | Win | Effort | Impact |
|---|-----|--------|--------|
| QW-1 | **Fix `breakdown` → `breakdown_json` in pages_cmds.py** | 5 min | Expert data starts appearing on GH Pages |
| QW-2 | **Fix h2h column names in pages_cmds.py** | 10 min | H2H data starts appearing on GH Pages |
| QW-3 | **Fix form JSON structure in pages_cmds.py** | 15 min | Form panel works on GH Pages |
| QW-4 | **Fix `predicted_score` display**: `md.poisson.predicted_score?.join?.(' - ') \|\| md.poisson.predicted_score \|\| '—'` | 5 min | Score shows "1 - 0" not "1,0" |
| QW-5 | **Remove duplicate `con()` in api.py H2H** | 5 min | Fix connection leak |
| QW-6 | **Delete v5.py and v3.py** | 5 min | Remove 1,313 lines of dead code |
| QW-7 | **Replace `except Exception: pass` with `except Exception as e: logger.warning(...)`** in pages_cmds.py | 10 min | Bugs become visible |
| QW-8 | **Fix standings CSS grid**: 9 → 8 columns | 5 min | Layout fix |
| QW-9 | **Add stale JSON cleanup** — delete `api/matches/` before export | 5 min | No accumulation |
| QW-10 | **Fix stale "6 experts" comments** in council.py | 10 min | Code accuracy |
| QW-11 | **Add CORS middleware** to api.py | 10 min | Browser security |
| QW-12 | **Generate `requirements.lock`** via `pip-compile` | 15 min | Reproducible builds |
| QW-13 | **Remove `hs/as_/hst/ast` from static match detail JSON** | 5 min | Cleaner export |
| QW-14 | **Add `implied_h/d/a` + `overround` to static match odds** | 10 min | Odds panel works on Pages |
| QW-15 | **Fix deploy gate**: `>= 0` → `> threshold` | 5 min | Prevent no-improvement deploys |

---

## G) Questions for the User

### GitHub Pages

1. **Is GH Pages the primary deployment target or just a demo?** If primary, the static export needs to be a first-class pipeline step. If demo-only, the broken exports are lower priority.

2. **Should `footy pages export` run automatically after `footy go` / `footy refresh`?** Currently it's a manual step that must be remembered.

3. **Do you want GH Pages to support match sub-pages (experts, h2h, form)?** Currently `docs/app.js` uses `_staticUrl()` which correctly maps URLs to JSON paths, but the JSON content is wrong. Once fixed, this will work. Confirm this is desired.

### Model & Predictions

4. **Are you aware of the Dixon-Coles data leakage?** The DC pseudo-expert's predictions on training rows are from a model fit on those same rows. This inflates training metrics. Fix requires zeroing DC features for train rows. Confirm you want this fixed.

5. **Which Brier score formula do you want?** Standard (÷K, where K=3) or raw sum? Both are defensible; the standard version gives values in [0, 2] range while the raw version gives [0, 6]. The key is consistency.

6. **Do you want the closing odds train/serve skew addressed?** Options: (a) always use opening odds only, (b) add a feature indicating which odds tier is available, (c) leave as-is and accept the skew.

7. **Do you want continuous training wired up?** Currently `auto_retrain` is dead code. Options: (a) wire to scheduler with proper thresholds, (b) keep as manual-only CLI command, (c) remove the infrastructure.

### Dead Code

8. **Can I delete `v5.py` (907 lines) and `v3.py` (406 lines)?** Both have zero callers and no model files on disk. They're legacy model versions superseded by v8_council.

9. **What about `understat.py` and `fbref.py`?** Both are stubs that always return None. Options: (a) delete, (b) implement (requires scraping infrastructure), (c) leave as placeholders for future work.

10. **What about `stats_cmds.py` (11 non-functional commands)?** 2 will crash with KeyError, 7 silently return "No data", 2 are broken. Options: (a) delete entire group, (b) fix with real data sources, (c) leave as-is.

### Deployment

11. **What's your target deployment platform?** Render free tier cannot persist DuckDB data or run the scheduler reliably. Options: (a) Render paid ($7/mo for persistent disk), (b) Railway ($5/mo), (c) VPS (Hetzner ~$4/mo), (d) just GH Pages + manual local pipeline.

12. **Do you need the API to be publicly accessible?** If yes, you need auth and rate limiting. If it's just for your personal use behind a bookmark, the current setup is fine.

### Architecture

13. **Would you consider migrating from DuckDB to PostgreSQL/SQLite?** DuckDB's single-writer limitation means the scheduler and CLI pipeline can't write simultaneously. PostgreSQL solves this but adds ops complexity. SQLite is a lateral move. DuckDB is fine if you avoid concurrent writes.

14. **Do you want a proper walkforward/backtesting module?** Currently there's only a single train/test split. A walkforward with expanding windows would give more robust model evaluation but requires 2-4 hours to implement and significant compute time to run.

---

## Appendix: Test Results

```
141 tests collected
129 passed ✅
12 failed ❌ (all in test_api.py — DuckDB read-only mode error)
0 errors
```

**Failing tests** (all same root cause):
- `test_index`, `test_matches_api`, `test_match_detail_api`, `test_value_bets_api`
- `test_standings_api`, `test_experts_api`, `test_h2h_api`, `test_form_api`
- `test_performance_api`, `test_narrative_api`, `test_search_api`, `test_last_updated_api`

**Root cause**: `web.api.con()` opens DuckDB with `read_only=True`. In test env, the `:memory:` database can't be opened read-only. Fix: mock `con()` in test fixtures.

---

## Appendix: Dead Code Inventory

| File | Lines | Status | Evidence |
|------|-------|--------|----------|
| `src/footy/models/v5.py` | 907 | Zero callers, no model file | `grep -r "v5" --include="*.py"` — only self-references |
| `src/footy/models/v3.py` | 406 | Zero callers, no model file | Same |
| `src/footy/understat.py` | 31 | Stub, always returns None | `return None` on every path |
| `src/footy/fbref.py` | 32 | Stub, always returns None/{} | `return None` / `return {}` |
| `src/footy/cli/stats_cmds.py` | 231 | 0/11 commands functional | 2 crash, 7 "no data", 2 broken |
| `src/footy/cache.py` (partial) | ~40 | 2 functions never called | `cache_prediction_batch`, `lookup_cached_predictions` |
| **Total** | **~1,647** | | |

---

## Appendix: File Complexity Hotspots

| File | Lines | Cyclomatic Concern |
|------|-------|-------------------|
| `council.py` | 1,939 | 8 expert classes + meta-learner + feature engineering. Very dense but well-structured. |
| `v5.py` | 907 | Dead code. Delete. |
| `api.py` | 867 | 16 endpoints. Some could be split into separate routers. |
| `pipeline.py` | 807 | Orchestrator with 8 steps. Each step is a function call. Reasonable. |
| `performance_tracker.py` | 718 | Analytics + tracking. Could split tracking from analysis. |
| `continuous_training.py` | 647 | 80% dead infrastructure (never called). Either wire up or delete. |
| `xg.py` | 546 | Two xG methods (basic + advanced). Advanced is broken (off-target rate). |
