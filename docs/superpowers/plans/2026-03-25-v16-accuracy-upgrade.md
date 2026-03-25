# v16 Accuracy Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix critical pipeline bugs, wire disconnected enrichment data, remove dead experts, and retrain to measurably improve prediction accuracy.

**Architecture:** Fix 4 bugs in council.py (OOF param mismatch, feature name desync, feature count instability, wrong feature builder in prepare_training_data), wire FBref/Transfermarkt/referee enrichment into go/refresh CLI commands, remove 5 dead experts, upgrade InjuryExpert for all leagues, add RPS metric, retrain and compare.

**Tech Stack:** Python 3.13, DuckDB, CatBoost, XGBoost, scikit-learn, FBref scraping, Transfermarkt scraping

---

## File Structure

**Modified files:**
- `src/footy/models/council.py` — Fix OOF params, feature names, auto-pruning, prepare_training_data
- `src/footy/models/experts/__init__.py` — Remove dead experts from ALL_EXPERTS
- `src/footy/models/experts/injury.py` — Read Transfermarkt injuries for non-PL leagues
- `src/footy/cli/pipeline_cmds.py` — Wire enrichment into go/refresh commands
- `src/footy/providers/data_ingest.py` — Add `refresh_all_enrichment()` convenience function

**No new files created.** All changes modify existing files.

---

### Task 1: Fix OOF Stacking Hyperparameter Mismatch

The OOF stacking meta-learner is trained on CatBoost predictions with different hyperparameters (iterations=1500, lr=0.037, depth=8, l2=9.55) than the final model (iterations=2638, lr=0.0397, depth=7, l2=5.90). This means the meta-learner learned from a different model than what it sees at inference. This is a train/serve skew bug.

**Files:**
- Modify: `src/footy/models/council.py:1415-1419`

- [ ] **Step 1: Fix OOF CatBoost params to match final model**

In `council.py`, lines 1415-1419, change the OOF fold CatBoost to use the same Optuna-optimized params as the final model (lines 1268-1283):

```python
# BEFORE (line 1415-1419):
fold_m = CatBoostClassifier(
    loss_function="MultiClass", iterations=1500,
    learning_rate=0.037, depth=8, l2_leaf_reg=9.55,
    random_seed=42, verbose=0,
)

# AFTER:
fold_m = CatBoostClassifier(
    loss_function="MultiClass",
    iterations=1500,  # fewer iters for speed in folds
    learning_rate=0.0397,
    depth=7,
    l2_leaf_reg=5.90,
    bootstrap_type="MVS",
    subsample=0.933,
    min_data_in_leaf=27,
    random_strength=5.80,
    random_seed=42,
    verbose=0,
)
```

Key: same architecture (depth, regularization, bootstrap) but fewer iterations for speed.

- [ ] **Step 2: Fix OOF XGBoost params similarly**

Lines 1423-1428, match the Optuna params from lines 1305-1321:

```python
# BEFORE:
fold_m = xgb.XGBClassifier(
    objective="multi:softprob", num_class=3,
    max_depth=5, learning_rate=0.025, n_estimators=1500,
    random_state=42, verbosity=0,
)

# AFTER:
fold_m = xgb.XGBClassifier(
    objective="multi:softprob", num_class=3,
    max_depth=5,
    min_child_weight=13,
    gamma=0.38,
    subsample=0.73,
    colsample_bytree=0.90,
    learning_rate=0.025,
    n_estimators=1500,
    reg_alpha=0.003,
    reg_lambda=8.46,
    max_delta_step=1,
    random_state=42,
    verbosity=0,
)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/ -x -q --tb=short -m "not slow and not smoke" --ignore=tests/test_enhanced_prediction.py -k "not test_from_db_reconstructs"`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add src/footy/models/council.py
git commit -m "fix: align OOF stacking hyperparams with final model params"
```

---

### Task 2: Fix V13_FEATURE_NAMES Desync

`V13_FEATURE_NAMES` (lines 1060-1106) lists ~100 feature names from an older version. The actual features built by `_build_v13_features()` are completely different. This makes SHAP analysis and debugging unreliable. Fix: generate feature names dynamically from the builder.

**Files:**
- Modify: `src/footy/models/council.py:725-1056, 1060-1106, 1704-1708`

- [ ] **Step 1: Make `_build_v13_features` return feature names alongside the matrix**

Change the function signature and return to also produce the ordered list of feature keys:

```python
def _build_v13_features(results: list[ExpertResult], experts: list[Expert] | None = None,
                        competitions: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
```

At the end (before the return), capture the numeric keys before building X:

```python
    numeric_keys = [k for k, v in features.items() if np.issubdtype(v.dtype, np.number)]
    blocks = [features[k][:, None] if features[k].ndim == 1 else features[k] for k in numeric_keys]

    X = np.hstack(blocks) if blocks else np.zeros((n, 0))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # v16: Automatic zero-feature pruning
    if X.shape[0] > 100:
        zero_rates = (X == 0).mean(axis=0)
        keep_mask = zero_rates < 0.95
        if keep_mask.sum() < X.shape[1]:
            n_pruned = X.shape[1] - keep_mask.sum()
            numeric_keys = [k for k, keep in zip(numeric_keys, keep_mask) if keep]
            X = X[:, keep_mask]
            log.debug("Auto-pruned %d features (>95%% zero), %d remaining", n_pruned, X.shape[1])

    return X, numeric_keys
```

- [ ] **Step 2: Update all callers of `_build_v13_features`**

In `train_and_save()` (around line 1245):
```python
# BEFORE:
X = _build_v13_features(results + dc_results, competitions=competitions)

# AFTER:
X, feature_names = _build_v13_features(results + dc_results, competitions=competitions)
```

In `predict_upcoming()` (around line 1890):
```python
# BEFORE:
X = _build_v13_features(results + dc_results, competitions=competitions)

# AFTER:
X, feature_names = _build_v13_features(results + dc_results, competitions=competitions)
```

- [ ] **Step 3: Save actual feature names in model artifact**

In `train_and_save()`, line 1708:
```python
# BEFORE:
"feature_names": V13_FEATURE_NAMES[:X.shape[1]],

# AFTER:
"feature_names": feature_names,
```

- [ ] **Step 4: Delete the stale V13_FEATURE_NAMES constant**

Remove lines 1060-1106 entirely (the `V13_FEATURE_NAMES = [...]` block). It's now dead code.

- [ ] **Step 5: Fix `prepare_training_data` to use correct builder**

Line 1137:
```python
# BEFORE:
X = _build_meta_X(results, competitions=competitions)

# AFTER:
X, _ = _build_v13_features(results, competitions=competitions)
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/ -x -q --tb=short -m "not slow and not smoke" --ignore=tests/test_enhanced_prediction.py -k "not test_from_db_reconstructs"`

- [ ] **Step 7: Commit**

```bash
git add src/footy/models/council.py
git commit -m "fix: generate feature names dynamically, fix prepare_training_data builder"
```

---

### Task 3: Fix Feature Count Instability

Auto-pruning can remove different features at training vs prediction time, causing silent column misalignment. Fix: save the pruning mask with the model and apply it consistently at prediction time.

**Files:**
- Modify: `src/footy/models/council.py` (train_and_save + predict_upcoming)

- [ ] **Step 1: Save feature names (already includes pruning info from Task 2)**

The feature_names list from Task 2 already reflects post-pruning state. Add it to the model artifact (done in Task 2 Step 3).

- [ ] **Step 2: At prediction time, align features to saved names**

In `predict_upcoming()`, after building X and feature_names, align to the model's saved feature names:

```python
# After: X, feature_names = _build_v13_features(...)
saved_names = obj.get("feature_names", [])
if saved_names and feature_names != saved_names:
    # Reorder/pad/drop to match training feature set
    saved_set = set(saved_names)
    new_X = np.zeros((X.shape[0], len(saved_names)))
    name_to_col = {name: i for i, name in enumerate(feature_names)}
    for j, sname in enumerate(saved_names):
        if sname in name_to_col:
            new_X[:, j] = X[:, name_to_col[sname]]
    X = new_X
    log.debug("Aligned %d prediction features to %d training features", len(feature_names), len(saved_names))
```

- [ ] **Step 3: Remove the old pad/truncate workaround**

Find and remove any existing code that does `X = X[:, :n_features]` or pads with zeros based on `obj["n_features"]`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -x -q --tb=short -m "not slow and not smoke" --ignore=tests/test_enhanced_prediction.py -k "not test_from_db_reconstructs"`

- [ ] **Step 5: Commit**

```bash
git add src/footy/models/council.py
git commit -m "fix: stable feature alignment between training and prediction"
```

---

### Task 4: Wire Enrichment into go/refresh Pipeline

The enrichment pipeline (FBref, Transfermarkt, referee, venue stats) exists but is never called by `go` or `refresh`. This is why all enrichment tables have 0 rows.

**Files:**
- Modify: `src/footy/cli/pipeline_cmds.py`
- Modify: `src/footy/providers/data_ingest.py`

- [ ] **Step 1: Add `refresh_all_enrichment()` to data_ingest.py**

At the end of data_ingest.py, add a convenience function that calls all enrichment steps:

```python
def refresh_all_enrichment(
    competitions: list[str] | tuple[str, ...] | None = None,
    season: str = "2025-2026",
) -> dict[str, int]:
    """Run all enrichment pipelines: FBref, Transfermarkt values, injuries, referee, venue.

    Each pipeline is isolated — one failure doesn't block the rest.
    """
    results = {}

    try:
        results["fbref"] = ingest_fbref_stats(competitions, season=season)
    except Exception as e:
        log.warning("enrichment: FBref failed: %s", e)
        results["fbref"] = 0

    try:
        results["transfermarkt_values"] = ingest_transfermarkt_values(competitions, season=season[:4])
    except Exception as e:
        log.warning("enrichment: Transfermarkt values failed: %s", e)
        results["transfermarkt_values"] = 0

    try:
        results["transfermarkt_injuries"] = ingest_transfermarkt_injuries(competitions)
    except Exception as e:
        log.warning("enrichment: Transfermarkt injuries failed: %s", e)
        results["transfermarkt_injuries"] = 0

    try:
        results["referee_stats"] = compute_referee_stats()
    except Exception as e:
        log.warning("enrichment: referee stats failed: %s", e)
        results["referee_stats"] = 0

    try:
        results["venue_stats"] = compute_venue_stats()
    except Exception as e:
        log.warning("enrichment: venue stats failed: %s", e)
        results["venue_stats"] = 0

    return results
```

- [ ] **Step 2: Add enrichment step to `go` command**

In `pipeline_cmds.py`, add enrichment between step 4 (new APIs) and step 5 (train Elo). Update step count from 11 to 12:

```python
    step(4, "Enriching data (FBref, Transfermarkt, referee, venue stats)...")
    try:
        from footy.providers.data_ingest import refresh_all_enrichment
        enrich = refresh_all_enrichment()
        console.print(f"  Enrichment: {enrich}")
    except Exception as e:
        console.print(f"[yellow]Enrichment warning:[/yellow] {e}")
```

- [ ] **Step 3: Add enrichment step to `refresh` command**

Same pattern, add between step 3 (new APIs) and step 4 (train Elo). Update step count from 9 to 10.

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -x -q --tb=short -m "not slow and not smoke" --ignore=tests/test_enhanced_prediction.py -k "not test_from_db_reconstructs"`

- [ ] **Step 5: Commit**

```bash
git add src/footy/cli/pipeline_cmds.py src/footy/providers/data_ingest.py
git commit -m "feat: wire FBref/Transfermarkt/referee/venue enrichment into go/refresh"
```

---

### Task 5: Remove Dead Experts

Remove experts that consistently produce zero signal due to missing data. This reduces compute time and noise in the consensus features.

**Experts to remove (5):**
- `NewsSentimentExpert` — reads columns that don't exist in `_MATCH_COLS`, always flat 1/3
- `OptaExpert` — depends on external Opta data that's never populated
- `CrowdMomentumExpert` — depends on stadiums table that has 0 rows, overlaps with VenueExpert
- `LineupExpert` — FPL-only (PL), 18/19 leagues get zero signal
- `TransferExpert` — depends on market_values which was 0 rows (now wired, but MarketValueExpert already covers this)

**Files:**
- Modify: `src/footy/models/experts/__init__.py`

- [ ] **Step 1: Remove dead experts from ALL_EXPERTS**

In `__init__.py`, remove these 5 entries from the `ALL_EXPERTS` list (lines 79-143):

```python
# Remove these lines:
    TransferExpert(),          # Transfer window activity impact
    NewsSentimentExpert(),     # News disruption and media sentiment
    LineupExpert(),            # Lineup strength and rotation detection
    OptaExpert(),              # Opta/TheAnalyst win probability predictions
    CrowdMomentumExpert(),     # Stadium atmosphere, home fortress, altitude, capacity
```

Keep the imports (for backward compat) but remove from `ALL_EXPERTS`. This reduces from 50 to 45 experts.

- [ ] **Step 2: Update the docstring**

Change the module docstring to reflect 45 experts instead of 50.

- [ ] **Step 3: Run tests**

Run: `pytest tests/ -x -q --tb=short -m "not slow and not smoke" --ignore=tests/test_enhanced_prediction.py -k "not test_from_db_reconstructs"`

- [ ] **Step 4: Commit**

```bash
git add src/footy/models/experts/__init__.py
git commit -m "refactor: remove 5 dead experts (news, opta, crowd, lineup, transfer)"
```

---

### Task 6: Upgrade InjuryAvailabilityExpert for All Leagues

Currently only reads FPL data (PL-only) and af_inj (API-Football, paid). Wire in Transfermarkt injuries which cover 18 leagues for free.

**Files:**
- Modify: `src/footy/models/experts/injury.py`
- Modify: `src/footy/models/council.py` (_MATCH_COLS and _MATCH_JOINS)

- [ ] **Step 1: Read current InjuryAvailabilityExpert**

Read `src/footy/models/experts/injury.py` to understand current implementation.

- [ ] **Step 2: Add Transfermarkt injury count to SQL query**

In council.py `_MATCH_COLS`, add after the existing market_values joins:

```sql
(SELECT COUNT(*) FROM transfermarkt_injuries ti WHERE ti.team = m.home_team) AS tm_inj_count_h,
(SELECT COUNT(*) FROM transfermarkt_injuries ti WHERE ti.team = m.away_team) AS tm_inj_count_a
```

- [ ] **Step 3: Update InjuryAvailabilityExpert to use Transfermarkt fallback**

In `injury.py`, after the FPL/AF logic, add a Transfermarkt fallback:

```python
# Transfermarkt injuries (available for all leagues)
tm_inj_h = _f(getattr(r, "tm_inj_count_h", None))
tm_inj_a = _f(getattr(r, "tm_inj_count_a", None))

# Use Transfermarkt if FPL/AF not available
if inj_h == 0 and tm_inj_h > 0:
    inj_h = tm_inj_h
    inj_a = tm_inj_a
    # Approximate injury_score from count (FPL scale is 0-1)
    injury_score_h = min(tm_inj_h / 10.0, 1.0)  # 10+ injuries = max severity
    injury_score_a = min(tm_inj_a / 10.0, 1.0)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -x -q --tb=short -m "not slow and not smoke" --ignore=tests/test_enhanced_prediction.py -k "not test_from_db_reconstructs"`

- [ ] **Step 5: Commit**

```bash
git add src/footy/models/experts/injury.py src/footy/models/council.py
git commit -m "feat: upgrade InjuryExpert with Transfermarkt fallback for all leagues"
```

---

### Task 7: Add RPS Metric to Training Evaluation

RPS (Ranked Probability Score) is the academically standard metric for ordered 3-class football outcomes. It's in the Optuna tuner but missing from the main training evaluation.

**Files:**
- Modify: `src/footy/models/council.py` (train_and_save evaluation section)

- [ ] **Step 1: Add RPS computation after accuracy/logloss/brier**

After line 1551 (where acc is computed), add:

```python
# RPS — proper scoring rule for ordered outcomes (H/D/A)
# Lower is better. Random baseline = 0.222, good model < 0.19
cum_pred = np.cumsum(P, axis=1)[:, :2]  # cumulative up to class 1
cum_true = np.cumsum(Y, axis=1)[:, :2]
rps = float(np.mean(np.sum((cum_pred - cum_true) ** 2, axis=1) / 2))
```

- [ ] **Step 2: Add RPS to output metrics**

In the `out` dict (line 1722-1734), add:
```python
"rps": round(rps, 5),
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/ -x -q --tb=short -m "not slow and not smoke" --ignore=tests/test_enhanced_prediction.py -k "not test_from_db_reconstructs"`

- [ ] **Step 4: Commit**

```bash
git add src/footy/models/council.py
git commit -m "feat: add RPS metric to training evaluation output"
```

---

### Task 8: Run Enrichment and Retrain

Execute the enrichment pipeline to populate empty tables, then retrain the model and compare metrics.

**Files:** None (execution only)

- [ ] **Step 1: Run enrichment pipeline**

```bash
source .venv/bin/activate
python -c "
from footy.providers.data_ingest import refresh_all_enrichment
results = refresh_all_enrichment()
print('Enrichment results:', results)
"
```

Expected: FBref populates team stats for 13+ leagues, referee stats derived from match_extras, venue stats computed.

- [ ] **Step 2: Verify enrichment data populated**

```bash
python -c "
import duckdb
con = duckdb.connect('data/footy.duckdb', read_only=True)
for t in ['fbref_team_stats', 'market_values', 'referee_stats', 'venue_stats', 'transfermarkt_injuries']:
    n = con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'{t}: {n} rows')
con.close()
"
```

- [ ] **Step 3: Retrain the model**

```bash
python -m footy.cli model train-meta
```

Record the output metrics (accuracy, logloss, brier, RPS, ECE, WF-CV results).

- [ ] **Step 4: Score existing predictions**

```bash
python -m footy.cli model predict
```

Then check scored predictions:
```bash
python -c "
import duckdb
con = duckdb.connect('data/footy.duckdb', read_only=True)
print(con.execute('SELECT COUNT(*) FROM prediction_scores').fetchone())
print(con.execute('SELECT model_version, COUNT(*), AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) FROM prediction_scores GROUP BY model_version').fetchdf().to_string())
con.close()
"
```

- [ ] **Step 5: Compare before/after metrics**

Document: accuracy, logloss, brier, RPS, ECE, WF-CV accuracy, WF-CV logloss.

- [ ] **Step 6: Commit trained model**

If metrics improved:
```bash
git add data/models/v15_architect.joblib
git commit -m "feat: retrained v15_architect with enrichment data and bug fixes"
```

---

### Task 9: Push and Release

- [ ] **Step 1: Run full lint check**

```bash
ruff check src/ web/ --select E,W,F --ignore E501,E702,E741,W293,W291
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -x -q --tb=short -m "not slow and not smoke" --ignore=tests/test_enhanced_prediction.py -k "not test_from_db_reconstructs"
```

- [ ] **Step 3: Push**

```bash
git push origin main
```
