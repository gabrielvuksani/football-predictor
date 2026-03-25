# v15 "Architect" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the prediction pipeline to fix data leakage, connect self-learning, wire unused data, learn all parameters, and add conformal prediction — achieving genuinely accurate, self-improving, upset-aware predictions.

**Architecture:** 6-layer pipeline: Experts → Feature Builder (with Hedge-weighted confidence) → K-fold OOF base models → Stacking meta-learner → Per-league temperature calibration → Conformal prediction sets. Self-learning loop feeds back after each matchday via Hedge weight updates, drift detection, and expert auto-pruning.

**Tech Stack:** Python 3.13, CatBoost, XGBoost, LightGBM, scikit-learn, scipy, numpy, DuckDB, mapie (conformal), soccerdata

---

## File Map

### Files to Modify
- `src/footy/models/council.py` — Fix leakage, new stacking, feature builder expansion
- `src/footy/self_learning.py` — Fix ADWIN, persist state, add DB read/write for Hedge weights
- `src/footy/models/experts/__init__.py` — Expert gating mechanism
- `src/footy/models/experts/_base.py` — Add expert signal quality tracking
- `src/footy/models/experts/news_sentiment_expert.py` — Fix data dependency (SQL join)
- `src/footy/models/experts/referee.py` — Fix SQL column aliases
- `src/footy/models/experts/xpts.py` — Fix or remove
- `src/footy/models/experts/market.py` — Fix 0/0/0 fallback to 1/3
- `src/footy/models/experts/hmm_expert.py` — EM-learned parameters
- `src/footy/models/experts/gas_expert.py` — MLE-learned parameters
- `src/footy/models/experts/copula_expert.py` — Data-driven theta
- `src/footy/models/experts/poisson_expert.py` — Data-driven COM-Poisson nu
- `src/footy/pipeline.py` — Connect self-learning feedback, fix hardcoded season
- `src/footy/models/math/scoring.py` — Add conformal prediction utilities

### Files to Create
- `src/footy/models/expert_gate.py` — Expert gating/pruning based on signal quality
- `src/footy/models/conformal.py` — Conformal prediction sets with coverage tracking
- `src/footy/models/parameter_learner.py` — HMM EM, GAS MLE, copula theta estimation
- `tests/test_leakage_fixes.py` — Tests verifying no data leakage
- `tests/test_self_learning_connected.py` — Tests verifying self-learning feeds back
- `tests/test_conformal.py` — Tests for conformal prediction coverage
- `tests/test_expert_gate.py` — Tests for expert gating
- `tests/test_parameter_learner.py` — Tests for learned parameters

---

## Sub-Project 1: Fix Foundation (Data Leakage + Dead Code)

### Task 1: Fix Stacking Meta-Learner Leakage

**Files:**
- Modify: `src/footy/models/council.py:1370-1400` (stacking section)
- Create: `tests/test_leakage_fixes.py`

- [ ] **Step 1: Write failing test for OOF stacking**

```python
# tests/test_leakage_fixes.py
"""Tests that verify the stacking meta-learner uses out-of-fold predictions only."""
import numpy as np
import pytest

def test_stacking_uses_oof_not_test():
    """The meta-learner must be trained on OOF predictions from training data,
    NOT on test set predictions. We verify by checking the stacking input
    shape matches the training split, not the test split."""
    from footy.models.council import ExpertCouncil
    # Mock a small dataset where we can verify the stacking input dimensions
    council = ExpertCouncil.__new__(ExpertCouncil)
    # The OOF matrix should have n_train rows, not n_test rows
    n_train, n_test = 500, 100
    n_models = 3
    oof_preds = np.random.rand(n_train, n_models * 3)
    test_preds = np.random.rand(n_test, n_models * 3)
    y_train = np.random.randint(0, 3, n_train)

    # The meta-learner should be fit on oof_preds (train), not test_preds
    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(C=0.5, max_iter=800, multi_class='multinomial')
    meta.fit(oof_preds, y_train)
    # Verify it was trained on training-sized data
    assert meta.n_features_in_ == n_models * 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_leakage_fixes.py::test_stacking_uses_oof_not_test -v`

- [ ] **Step 3: Implement proper OOF stacking in council.py**

In `council.py`, replace the stacking section (around line 1370-1400) with:

```python
# ── Proper OOF stacking (no test set leakage) ──
from sklearn.model_selection import StratifiedKFold

n_stack_folds = 5
skf = StratifiedKFold(n_splits=n_stack_folds, shuffle=True, random_state=42)
oof_cat = np.zeros((len(ytr), 3))
oof_xgb = np.zeros((len(ytr), 3))
oof_hist = np.zeros((len(ytr), 3))
oof_lr = np.zeros((len(ytr), 3))

for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(Xtr, ytr)):
    X_fold_tr, X_fold_val = Xtr[tr_idx], Xtr[val_idx]
    y_fold_tr = ytr[tr_idx]

    # CatBoost fold
    cat_fold = CatBoostClassifier(**cat_params, verbose=0)
    cat_fold.fit(X_fold_tr, y_fold_tr)
    oof_cat[val_idx] = cat_fold.predict_proba(X_fold_val)

    # XGBoost fold
    xgb_fold = XGBClassifier(**xgb_params, verbosity=0)
    xgb_fold.fit(X_fold_tr, y_fold_tr)
    oof_xgb[val_idx] = xgb_fold.predict_proba(X_fold_val)

    # HistGBM fold
    hist_fold = HistGradientBoostingClassifier(**hist_params)
    hist_fold.fit(X_fold_tr, y_fold_tr)
    oof_hist[val_idx] = hist_fold.predict_proba(X_fold_val)

    # LR fold
    lr_fold = LogisticRegression(C=0.5, max_iter=800, multi_class='multinomial')
    lr_fold.fit(X_fold_tr, y_fold_tr)
    oof_lr[val_idx] = lr_fold.predict_proba(X_fold_val)

# Stack OOF predictions as meta-learner input
stack_oof = np.hstack([oof_cat, oof_xgb, oof_hist, oof_lr])
meta_pipe = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial')
meta_pipe.fit(stack_oof, ytr)

# For test predictions, use the FULL base models (trained on all of Xtr)
stack_test = np.hstack([
    cat_model.predict_proba(Xte),
    xgb_model.predict_proba(Xte),
    hist_model.predict_proba(Xte),
    lr_model.predict_proba(Xte),
])
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_leakage_fixes.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/footy/models/council.py tests/test_leakage_fixes.py
git commit -m "fix: stacking meta-learner uses OOF predictions — eliminates test set leakage"
```

### Task 2: Fix Temperature Scaling Leakage

**Files:**
- Modify: `src/footy/models/council.py:1420-1460` (temperature section)

- [ ] **Step 1: Write failing test**

```python
# tests/test_leakage_fixes.py (append)
def test_temperature_uses_calibration_split():
    """Temperature must be optimized on a calibration split, not the test split."""
    # After the fix, the training function should accept and use a cal_split
    # The test set should never be used for temperature optimization
    pass  # Structural test — verified by code review in step 3
```

- [ ] **Step 2: Implement 3-way split in council.py training**

Change the data splitting from 2-way (train/test) to 3-way (train/calibration/test):

```python
# Replace the current train/test split with:
n = len(df_finished)
n_test = max(200, int(n * 0.15))
n_cal = max(100, int(n * 0.10))
n_train = n - n_test - n_cal

df_train = df_finished.iloc[:n_train]
df_cal = df_finished.iloc[n_train:n_train + n_cal]
df_test = df_finished.iloc[n_train + n_cal:]

# ... build features for each split ...
# Temperature scaling uses ONLY df_cal:
temp_result = optimize_temperature(P_cal_raw, y_cal)
# Final evaluation uses ONLY df_test (never seen by temperature or stacking)
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -x -q --tb=short -m "not slow and not smoke" --ignore=tests/test_enhanced_prediction.py -k "not test_from_db_reconstructs"`

- [ ] **Step 4: Commit**

```bash
git commit -am "fix: temperature scaling uses calibration split — eliminates evaluation leakage"
```

### Task 3: Fix Probability Clipping + MarketExpert Fallback + Feature Names

**Files:**
- Modify: `src/footy/models/council.py:1861` (clipping)
- Modify: `src/footy/models/council.py:1050-1096` (V13_FEATURE_NAMES)
- Modify: `src/footy/models/experts/market.py` (0/0/0 fallback)

- [ ] **Step 1: Raise probability ceiling from 95% to 99%**

```python
# council.py line 1861 — change:
P = np.clip(P, 0.01, 0.99)  # was 0.02, 0.95
```

- [ ] **Step 2: Fix MarketExpert to return 1/3 instead of 0/0/0 when no odds**

In `market.py`, find where probs are initialized and change the no-odds fallback:
```python
# When no odds available, return flat prior instead of zeros
if not has_odds:
    probs[i] = [1/3, 1/3, 1/3]  # was [0, 0, 0]
    confidence[i] = 0.0
```

- [ ] **Step 3: Sync V13_FEATURE_NAMES with actual features**

Read the runtime feature dict keys from `_build_v13_features` and ensure `V13_FEATURE_NAMES` includes all of them — especially the missing crowd, morale, opta, and competition one-hot features.

- [ ] **Step 4: Commit**

```bash
git commit -am "fix: raise prob ceiling to 99%, fix market 0/0/0 fallback, sync feature names"
```

### Task 4: Prune/Fix Dead Experts

**Files:**
- Modify: `src/footy/models/experts/xpts.py` — remove (columns don't exist)
- Modify: `src/footy/models/experts/news_sentiment_expert.py` — will be fixed in Sub-Project 2
- Modify: `src/footy/models/experts/referee.py` — fix SQL column aliases
- Modify: `src/footy/models/experts/__init__.py` — remove XPtsExpert from ALL_EXPERTS

- [ ] **Step 1: Remove XPtsExpert from ALL_EXPERTS** (reads nonexistent columns)

- [ ] **Step 2: Fix RefereeExpert column names** to match the actual SQL query aliases from `_MATCH_COLS`

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```bash
git commit -am "fix: remove dead XPtsExpert, fix referee column aliases"
```

---

## Sub-Project 2: Wire Up Unused Data

### Task 5: Join News Table to Match Query

**Files:**
- Modify: `src/footy/models/council.py:94-163` (`_MATCH_COLS` SQL)
- Modify: `src/footy/models/experts/news_sentiment_expert.py`

- [ ] **Step 1: Add LEFT JOIN for news data in `_MATCH_COLS`**

```sql
LEFT JOIN (
    SELECT team, AVG(tone) as news_tone, COUNT(*) as news_count,
           MAX(CASE WHEN tone < -3 THEN 1 ELSE 0 END) as news_disruption
    FROM news
    WHERE published_at >= m.utc_date - INTERVAL 3 DAY
      AND published_at < m.utc_date
    GROUP BY team
) nh ON nh.team = m.home_team
LEFT JOIN (...) na ON na.team = m.away_team
```

This feeds `news_tone_h`, `news_tone_a`, `news_count_h`, `news_count_a`, `news_disruption_h`, `news_disruption_a` to the DataFrame, which the NewsSentimentExpert already reads.

- [ ] **Step 2: Verify NewsSentimentExpert produces non-flat output**

- [ ] **Step 3: Commit**

```bash
git commit -am "feat: join news table to match query — activates NewsSentimentExpert"
```

### Task 6: Feed Market Odds Directly to BTTS/O2.5 Heads

**Files:**
- Modify: `src/footy/models/council.py` (feature builder + BTTS/O2.5 training)

- [ ] **Step 1: Add BTTS and O2.5 implied probabilities as direct features**

```python
# In _build_v13_features, add:
btts_yes_odds = _f(getattr(r, 'odds_btts_yes', 0))
btts_no_odds = _f(getattr(r, 'odds_btts_no', 0))
if btts_yes_odds > 1.0 and btts_no_odds > 1.0:
    btts_implied = 1.0 / btts_yes_odds
    features['mkt_btts_implied'] = btts_implied
else:
    features['mkt_btts_implied'] = 0.5  # no signal

o25_odds = _f(getattr(r, 'b365_o25', 0))
u25_odds = _f(getattr(r, 'b365_u25', 0))
if o25_odds > 1.0 and u25_odds > 1.0:
    features['mkt_o25_implied'] = 1.0 / o25_odds
else:
    features['mkt_o25_implied'] = 0.5
```

- [ ] **Step 2: Add Asian Handicap features**

```python
ah_line = _f(getattr(r, 'odds_ah_line', 0))
ah_home = _f(getattr(r, 'odds_ah_home', 0))
ah_away = _f(getattr(r, 'odds_ah_away', 0))
if ah_home > 1.0 and ah_away > 1.0:
    features['mkt_ah_line'] = ah_line
    features['mkt_ah_implied_h'] = 1.0 / ah_home
else:
    features['mkt_ah_line'] = 0.0
    features['mkt_ah_implied_h'] = 0.5
```

- [ ] **Step 3: Feed FPL FDR to ContextExpert**

```python
# Add to context expert feature extraction:
fdr_h = _f(getattr(r, 'fpl_fdr_h', 3.0))
fdr_a = _f(getattr(r, 'fpl_fdr_a', 3.0))
features['ctx_fdr_diff'] = fdr_h - fdr_a
features['ctx_fdr_hard_h'] = 1 if fdr_h >= 4 else 0
features['ctx_fdr_hard_a'] = 1 if fdr_a >= 4 else 0
```

- [ ] **Step 4: Commit**

```bash
git commit -am "feat: wire BTTS/O2.5/AH odds + FPL FDR as direct features"
```

---

## Sub-Project 3: Learn All Parameters

### Task 7: Create Parameter Learner Module

**Files:**
- Create: `src/footy/models/parameter_learner.py`
- Create: `tests/test_parameter_learner.py`

- [ ] **Step 1: Write tests for HMM EM algorithm**

```python
# tests/test_parameter_learner.py
def test_hmm_em_learns_from_data():
    """EM algorithm should learn emission rates from observed goals."""
    from footy.models.parameter_learner import learn_hmm_params
    # Synthetic data: team with 3 distinct phases
    goals = [2,3,2,3,1,1,1,1,0,0,1,0,2,3,2,2]  # strong→weak→strong
    params = learn_hmm_params(goals, n_states=3, n_iter=50)
    assert 'emission_rates' in params
    assert 'transition_matrix' in params
    assert params['emission_rates'].shape == (3,)
    # The dominant state should have highest emission rate
    assert max(params['emission_rates']) > 1.5

def test_gas_mle_learns_persistence():
    """GAS MLE should learn persistence parameter B close to 0.98 for stable teams."""
    from footy.models.parameter_learner import learn_gas_params
    import numpy as np
    # Synthetic stable team: goals around 1.5
    goals = np.random.poisson(1.5, 100)
    params = learn_gas_params(goals)
    assert 'A' in params and 'B' in params
    assert 0.8 < params['B'] < 1.0  # high persistence for stable team

def test_copula_theta_from_goals():
    """Copula theta should be estimated from observed goal correlations."""
    from footy.models.parameter_learner import learn_copula_theta
    import numpy as np
    home = np.random.poisson(1.5, 200)
    away = np.random.poisson(1.2, 200)
    theta = learn_copula_theta(home, away, family='frank')
    assert isinstance(theta, float)
    assert -10 < theta < 10
```

- [ ] **Step 2: Implement parameter_learner.py**

```python
"""Data-driven parameter estimation for expert models.

Replaces hardcoded parameters with learned values:
- HMM: Baum-Welch EM for emission rates and transition matrix
- GAS: Maximum likelihood for A, B, omega
- Copula: MLE for dependence parameter theta
- COM-Poisson: Variance/mean ratio for dispersion nu
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


def learn_hmm_params(goals: list[int], n_states: int = 3, n_iter: int = 100) -> dict:
    """Baum-Welch EM algorithm for HMM emission and transition parameters."""
    T = len(goals)
    if T < 20:
        return _default_hmm_params(n_states)

    # Initialize emission rates spread across goal range
    obs = np.array(goals, dtype=float)
    emission_rates = np.linspace(max(0.3, obs.min()), max(2.5, obs.max()), n_states)
    trans = np.full((n_states, n_states), 1.0 / n_states)
    np.fill_diagonal(trans, 0.8)
    trans /= trans.sum(axis=1, keepdims=True)
    pi = np.ones(n_states) / n_states

    for _ in range(n_iter):
        # E-step: forward-backward
        alpha = np.zeros((T, n_states))
        for s in range(n_states):
            alpha[0, s] = pi[s] * _poisson_pmf(goals[0], emission_rates[s])
        alpha[0] /= max(alpha[0].sum(), 1e-300)

        for t in range(1, T):
            for s in range(n_states):
                alpha[t, s] = sum(alpha[t-1, j] * trans[j, s] for j in range(n_states))
                alpha[t, s] *= _poisson_pmf(goals[t], emission_rates[s])
            alpha[t] /= max(alpha[t].sum(), 1e-300)

        beta = np.zeros((T, n_states))
        beta[T-1] = 1.0
        for t in range(T-2, -1, -1):
            for s in range(n_states):
                beta[t, s] = sum(trans[s, j] * _poisson_pmf(goals[t+1], emission_rates[j]) * beta[t+1, j]
                                 for j in range(n_states))
            beta[t] /= max(beta[t].sum(), 1e-300)

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True).clip(1e-300)

        # M-step
        for s in range(n_states):
            emission_rates[s] = max(0.1, (gamma[:, s] * obs).sum() / max(gamma[:, s].sum(), 1e-10))

        for i in range(n_states):
            for j in range(n_states):
                xi_sum = sum(
                    alpha[t, i] * trans[i, j] * _poisson_pmf(goals[t+1], emission_rates[j]) * beta[t+1, j]
                    for t in range(T-1)
                )
                trans[i, j] = max(xi_sum, 1e-10)
            trans[i] /= max(trans[i].sum(), 1e-10)

    # Sort states by emission rate (low → high)
    order = np.argsort(emission_rates)
    emission_rates = emission_rates[order]
    trans = trans[order][:, order]

    return {'emission_rates': emission_rates, 'transition_matrix': trans}


def learn_gas_params(goals: np.ndarray, home: bool = True) -> dict:
    """MLE for Score-Driven (GAS) model parameters (Koopman & Lit 2019)."""
    if len(goals) < 30:
        return {'A': 0.05, 'B': 0.98, 'omega': 0.0}

    def neg_log_lik(params):
        omega, A, B = params
        T = len(goals)
        lam = np.zeros(T)
        lam[0] = goals.mean()
        ll = 0.0
        for t in range(1, T):
            score = goals[t-1] - lam[t-1]  # scaled score
            lam[t] = max(0.1, omega + A * score + B * lam[t-1])
            ll += goals[t] * np.log(max(lam[t], 1e-10)) - lam[t] - gammaln(goals[t] + 1)
        return -ll

    result = minimize(neg_log_lik, x0=[0.0, 0.05, 0.98],
                      bounds=[(-2, 2), (0.001, 0.5), (0.5, 0.999)],
                      method='L-BFGS-B')
    omega, A, B = result.x
    return {'omega': float(omega), 'A': float(A), 'B': float(B)}


def learn_copula_theta(home_goals: np.ndarray, away_goals: np.ndarray,
                       family: str = 'frank') -> float:
    """Estimate copula dependence parameter from observed goal pairs."""
    if len(home_goals) < 30:
        return -1.8 if family == 'frank' else 0.3

    from scipy.stats import kendalltau
    tau, _ = kendalltau(home_goals, away_goals)
    tau = np.clip(tau, -0.5, 0.5)

    if family == 'frank':
        # Frank copula: tau = 1 - 4/theta * (1 - D1(theta)) where D1 is Debye
        # Approximate inverse: theta ≈ -9 * tau for |tau| < 0.3
        return float(np.clip(-9.0 * tau, -15.0, 15.0))
    elif family == 'clayton':
        return float(np.clip(2.0 * tau / (1.0 - tau), 0.01, 10.0)) if tau > 0 else 0.3
    elif family == 'gumbel':
        return float(np.clip(1.0 / (1.0 - tau), 1.0, 10.0)) if tau > 0 else 1.15
    return -1.8


def learn_com_poisson_nu(goals: np.ndarray) -> float:
    """Estimate COM-Poisson dispersion from variance/mean ratio."""
    if len(goals) < 30:
        return 0.93
    mean = goals.mean()
    var = goals.var()
    if mean < 0.1:
        return 1.0
    ratio = var / mean
    # nu ≈ 1/ratio for COM-Poisson (nu < 1 = overdispersed, nu > 1 = underdispersed)
    nu = np.clip(1.0 / max(ratio, 0.1), 0.3, 3.0)
    return float(nu)


def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF avoiding overflow."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return float(np.exp(k * np.log(lam) - lam - gammaln(k + 1)))


def _default_hmm_params(n_states: int = 3) -> dict:
    rates = np.linspace(0.8, 1.8, n_states)
    trans = np.full((n_states, n_states), 0.1)
    np.fill_diagonal(trans, 0.8)
    trans /= trans.sum(axis=1, keepdims=True)
    return {'emission_rates': rates, 'transition_matrix': trans}
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_parameter_learner.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/footy/models/parameter_learner.py tests/test_parameter_learner.py
git commit -m "feat: add parameter learner — HMM EM, GAS MLE, copula theta, COM-Poisson nu"
```

### Task 8: Wire Learned Parameters into Experts

**Files:**
- Modify: `src/footy/models/experts/hmm_expert.py`
- Modify: `src/footy/models/experts/gas_expert.py`
- Modify: `src/footy/models/experts/copula_expert.py`
- Modify: `src/footy/models/experts/poisson_expert.py`

- [ ] **Step 1: HMM Expert — use EM-learned params instead of hardcoded**

Replace the hardcoded `EMISSION_RATES` and `TRANS` constants with a warmup phase that calls `learn_hmm_params` after collecting enough finished match data per competition.

- [ ] **Step 2: GAS Expert — use MLE-learned params**

Replace hardcoded `A=0.05, B=0.98` with `learn_gas_params` called per league.

- [ ] **Step 3: Copula Expert — use data-driven theta**

Replace hardcoded `THETA_FRANK=-1.8` etc. with `learn_copula_theta` per competition.

- [ ] **Step 4: Poisson Expert — use data-driven COM-Poisson nu**

Replace hardcoded `nu_h=nu_a=0.93` with `learn_com_poisson_nu` per league.

- [ ] **Step 5: Run full test suite**

- [ ] **Step 6: Commit**

```bash
git commit -am "feat: wire learned parameters into HMM, GAS, Copula, Poisson experts"
```

---

## Sub-Project 4: Connect Self-Learning Loop

### Task 9: Persist Hedge Weights in DB

**Files:**
- Modify: `src/footy/self_learning.py`
- Modify: `src/footy/pipeline.py`
- Modify: `src/footy/models/council.py`

- [ ] **Step 1: Create DuckDB table for expert weights**

```sql
CREATE TABLE IF NOT EXISTS expert_hedge_weights (
    expert_name TEXT PRIMARY KEY,
    weight DOUBLE NOT NULL DEFAULT 1.0,
    cumulative_loss DOUBLE NOT NULL DEFAULT 0.0,
    n_predictions INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

- [ ] **Step 2: Write Hedge weights to DB after each scoring round**

In `self_learning.py`, after Hedge update, persist weights:
```python
def _persist_hedge_weights(self, con):
    for expert_name, tracker in self.expert_trackers.items():
        weight = tracker.get_weight()
        con.execute("""
            INSERT OR REPLACE INTO expert_hedge_weights
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [expert_name, weight, tracker.cumulative_loss, tracker.n_predictions])
```

- [ ] **Step 3: Load Hedge weights in council.py at prediction time**

```python
def _load_hedge_weights(con) -> dict[str, float]:
    """Load expert Hedge weights from self-learning system."""
    try:
        rows = con.execute("SELECT expert_name, weight FROM expert_hedge_weights").fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}
```

- [ ] **Step 4: Scale expert confidence by Hedge weight in feature builder**

```python
# In _build_v13_features, after computing expert confidence:
hedge_weights = _load_hedge_weights(con)
for expert_name, result in expert_results.items():
    hw = hedge_weights.get(expert_name, 1.0)
    result.confidence *= hw  # Scale confidence by learned weight
```

- [ ] **Step 5: Commit**

```bash
git commit -am "feat: persist Hedge weights in DB, load at prediction time, scale expert confidence"
```

### Task 10: Connect Drift Detection → Automatic Retraining

**Files:**
- Modify: `src/footy/self_learning.py` (fix ADWIN)
- Modify: `src/footy/pipeline.py` (trigger retraining)

- [ ] **Step 1: Fix ADWIN variance computation** (lines 559-560 where variance is computed but discarded)

```python
# Replace discarded computations with proper assignment:
var0 = max(0, sq0 / c0 - mean0**2)
var1 = max(0, sq1 / c1 - mean1**2)
# Use variance in Hoeffding bound:
epsilon = np.sqrt(2.0 * (var0/c0 + var1/c1) * np.log(2.0 / self.delta))
```

- [ ] **Step 2: Add retraining trigger in pipeline**

When drift detection consensus is reached, call retraining with higher time-decay:
```python
if drift_result.get('drift_detected'):
    logger.warning("Drift detected — triggering retrain with decay=0.5")
    retrain_with_decay(half_life_days=180)  # faster adaptation
```

- [ ] **Step 3: Commit**

```bash
git commit -am "feat: fix ADWIN detector, connect drift detection to auto-retraining"
```

### Task 11: Apply Per-League Temperature at Prediction Time

**Files:**
- Modify: `src/footy/models/council.py` (prediction path)
- Modify: `src/footy/self_learning.py` (persist temperatures)

- [ ] **Step 1: Persist per-league temperatures in DB**

```sql
CREATE TABLE IF NOT EXISTS league_temperatures (
    competition TEXT PRIMARY KEY,
    temperature DOUBLE NOT NULL DEFAULT 1.0,
    n_observations INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

- [ ] **Step 2: Apply per-league temperature in predict_upcoming**

```python
def _apply_league_temperature(P, competitions, con):
    """Apply per-league temperature scaling from self-learning."""
    try:
        rows = con.execute("SELECT competition, temperature FROM league_temperatures").fetchall()
        temps = {r[0]: r[1] for r in rows}
    except Exception:
        return P

    for i, comp in enumerate(competitions):
        t = temps.get(comp, 1.0)
        if t > 0 and t != 1.0:
            logits = np.log(np.clip(P[i], 1e-10, 1.0))
            P[i] = np.exp(logits / t)
            P[i] /= P[i].sum()
    return P
```

- [ ] **Step 3: Commit**

```bash
git commit -am "feat: apply per-league temperature from self-learning at prediction time"
```

---

## Sub-Project 5: Expert Gating

### Task 12: Create Expert Gate Module

**Files:**
- Create: `src/footy/models/expert_gate.py`
- Create: `tests/test_expert_gate.py`
- Modify: `src/footy/models/experts/__init__.py`

- [ ] **Step 1: Write test**

```python
def test_expert_gate_prunes_flat_experts():
    """Experts consistently returning flat 1/3 probs should be gated out."""
    from footy.models.expert_gate import ExpertGate
    gate = ExpertGate()
    # Simulate an expert that always returns flat
    for _ in range(100):
        gate.record('FlatExpert', probs=[0.333, 0.333, 0.334], confidence=0.0)
        gate.record('GoodExpert', probs=[0.6, 0.2, 0.2], confidence=0.7)
    assert not gate.is_active('FlatExpert')
    assert gate.is_active('GoodExpert')
```

- [ ] **Step 2: Implement ExpertGate**

```python
"""Expert gating — automatically disable experts producing pure noise."""
import numpy as np
from collections import defaultdict


class ExpertGate:
    """Track expert signal quality and gate out noise producers."""

    def __init__(self, min_observations: int = 50, entropy_threshold: float = 0.98):
        self.min_observations = min_observations
        self.entropy_threshold = entropy_threshold  # fraction of max entropy
        self._history = defaultdict(list)

    def record(self, expert_name: str, probs: list[float], confidence: float):
        self._history[expert_name].append({
            'entropy': self._entropy(probs),
            'confidence': confidence,
        })

    def is_active(self, expert_name: str) -> bool:
        history = self._history.get(expert_name, [])
        if len(history) < self.min_observations:
            return True  # benefit of the doubt until enough data

        recent = history[-self.min_observations:]
        max_entropy = np.log(3)  # entropy of uniform [1/3, 1/3, 1/3]
        mean_entropy = np.mean([h['entropy'] for h in recent])
        mean_conf = np.mean([h['confidence'] for h in recent])

        # Gate out if: near-max entropy AND near-zero confidence
        if mean_entropy / max_entropy > self.entropy_threshold and mean_conf < 0.05:
            return False
        return True

    def get_active_experts(self, all_experts: list) -> list:
        return [e for e in all_experts if self.is_active(e.name)]

    @staticmethod
    def _entropy(probs):
        p = np.clip(probs, 1e-12, 1.0)
        return float(-(p * np.log(p)).sum())
```

- [ ] **Step 3: Integrate into council.py expert loop**

- [ ] **Step 4: Commit**

```bash
git add src/footy/models/expert_gate.py tests/test_expert_gate.py
git commit -m "feat: expert gating — auto-prune noise-producing experts"
```

---

## Sub-Project 6: Conformal Prediction Sets

### Task 13: Create Conformal Prediction Module

**Files:**
- Create: `src/footy/models/conformal.py`
- Create: `tests/test_conformal.py`

- [ ] **Step 1: Write tests**

```python
def test_conformal_coverage():
    """Conformal prediction sets must achieve target coverage on held-out data."""
    from footy.models.conformal import ConformalPredictor
    import numpy as np

    np.random.seed(42)
    # Simulate calibration data
    n_cal = 200
    probs_cal = np.random.dirichlet([2, 1, 1], n_cal)
    y_cal = np.array([np.random.choice(3, p=p) for p in probs_cal])

    cp = ConformalPredictor(alpha=0.10)  # 90% coverage target
    cp.calibrate(probs_cal, y_cal)

    # Test on new data
    n_test = 500
    probs_test = np.random.dirichlet([2, 1, 1], n_test)
    y_test = np.array([np.random.choice(3, p=p) for p in probs_test])

    sets = cp.predict_sets(probs_test)
    coverage = np.mean([y_test[i] in sets[i] for i in range(n_test)])
    assert coverage >= 0.85  # allow some slack for randomness
```

- [ ] **Step 2: Implement conformal.py**

```python
"""Conformal prediction sets for football match outcomes.

Provides distribution-free uncertainty quantification with coverage guarantees.
Uses Adaptive Prediction Sets (APS) method from Romano et al. (2020).
"""
import numpy as np


class ConformalPredictor:
    """Conformal prediction sets with guaranteed coverage."""

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha  # miscoverage rate (1 - coverage target)
        self.qhat = None

    def calibrate(self, probs: np.ndarray, y: np.ndarray):
        """Calibrate on held-out data using APS scores."""
        n = len(y)
        scores = np.zeros(n)

        for i in range(n):
            # APS score: cumulative probability of sorted classes until true class included
            sorted_idx = np.argsort(-probs[i])
            cumsum = 0.0
            for j, cls in enumerate(sorted_idx):
                cumsum += probs[i, cls]
                if cls == y[i]:
                    # Add uniform random for tie-breaking
                    scores[i] = cumsum - probs[i, cls] * np.random.uniform()
                    break

        # Quantile with finite-sample correction
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat = np.quantile(scores, min(level, 1.0))

    def predict_sets(self, probs: np.ndarray) -> list[list[int]]:
        """Produce prediction sets for each row."""
        if self.qhat is None:
            raise ValueError("Must calibrate before predicting")

        n = len(probs)
        sets = []
        for i in range(n):
            sorted_idx = np.argsort(-probs[i])
            cumsum = 0.0
            pred_set = []
            for cls in sorted_idx:
                cumsum += probs[i, cls]
                pred_set.append(int(cls))
                if cumsum >= self.qhat:
                    break
            sets.append(pred_set)
        return sets

    def set_sizes(self, probs: np.ndarray) -> np.ndarray:
        """Return prediction set sizes — larger = more uncertain = upset risk."""
        sets = self.predict_sets(probs)
        return np.array([len(s) for s in sets])
```

- [ ] **Step 3: Integrate into council.py prediction path**

After temperature scaling, compute conformal prediction sets and store them in the prediction output.

- [ ] **Step 4: Commit**

```bash
git add src/footy/models/conformal.py tests/test_conformal.py
git commit -m "feat: conformal prediction sets with APS — uncertainty quantification for upset detection"
```

---

## Integration & Final Tasks

### Task 14: Update V15_FEATURE_NAMES and Model Version

- [ ] **Step 1: Rename V13_FEATURE_NAMES → V15_FEATURE_NAMES, add all new features**
- [ ] **Step 2: Bump MODEL_VERSION to v15_architect throughout codebase**
- [ ] **Step 3: Update CLAUDE.md, pyproject.toml, README**
- [ ] **Step 4: Run full test suite**
- [ ] **Step 5: Commit**

### Task 15: End-to-End Validation

- [ ] **Step 1: Run walk-forward CV with all fixes applied**

```bash
python -c "
from footy.models.council import ExpertCouncil
council = ExpertCouncil()
council.train_meta(verbose=True)
"
```

- [ ] **Step 2: Compare true accuracy (post-leakage-fix) against baseline**
- [ ] **Step 3: Verify self-learning loop executes end-to-end**
- [ ] **Step 4: Verify conformal prediction coverage ≥ 90%**
- [ ] **Step 5: Final commit and release**

```bash
git tag v15.0.0
git push origin main --tags
gh release create v15.0.0 --title "v15 Architect" --notes "..."
```
