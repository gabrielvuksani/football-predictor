# Calibration Pipeline & Enhanced Drift Detection Upgrades

## Overview

This document describes two major upgrades to the football prediction system:

1. **Comprehensive Calibration Pipeline** (`src/footy/models/calibration.py`)
2. **Enhanced Drift Detection** (updates to `src/footy/self_learning.py`)
3. **Enhanced Self-Learning Weights** (updates to `src/footy/self_learning.py`)

## 1. Calibration Pipeline

### Location
`src/footy/models/calibration.py`

### Purpose
Ensures predicted probabilities match empirical frequencies, critical for:
- Reliable uncertainty quantification
- Better-calibrated ensemble predictions
- Detecting systematic biases in predictions
- Improved decision-making in betting/trading systems

### Implemented Methods

#### 1.1 Platt Scaling
```python
from footy.models.calibration import PlattCalibrator

calibrator = PlattCalibrator(n_classes=3)
calibrator.fit(training_probs, training_outcomes)
calibrated_probs = calibrator.calibrate(test_probs)
```

**Characteristics:**
- Simple logistic regression on model outputs
- Fits P(y=1|f) = sigmoid(Af + B)
- Fast, interpretable, works well for 2-class problems
- Trains one-vs-rest classifiers for multi-class

**When to use:**
- When you want simplicity and interpretability
- Limited calibration data available
- Real-time inference speed is critical

#### 1.2 Isotonic Regression
```python
from footy.models.calibration import IsotonicCalibrator

calibrator = IsotonicCalibrator(n_classes=3)
calibrator.fit(training_probs, training_outcomes)
calibrated_probs = calibrator.calibrate(test_probs)
```

**Characteristics:**
- Non-parametric monotonic calibration
- Fits monotonically increasing function
- More flexible than Platt scaling
- Requires more data to prevent overfitting

**When to use:**
- Flexible calibration needed
- Sufficient calibration data (100+ samples per class)
- Want to preserve monotonicity

#### 1.3 Temperature Scaling
```python
from footy.models.calibration import TemperatureCalibrator

calibrator = TemperatureCalibrator()
calibrator.fit(training_probs, training_outcomes)
calibrated_probs = calibrator.calibrate(test_probs)
print(f"Optimal temperature: {calibrator.temperature:.4f}")
```

**Characteristics:**
- Single-parameter method: divides logits by temperature T
- P_calib = softmax(logits / T)
- Highly efficient
- Excellent for deep neural networks

**When to use:**
- Single scalar temperature scales all classes uniformly
- Very fast inference
- Modern neural networks commonly need this

#### 1.4 Beta Calibration
```python
from footy.models.calibration import BetaCalibrator

calibrator = BetaCalibrator(n_classes=3)
calibrator.fit(training_probs, training_outcomes)
calibrated_probs = calibrator.calibrate(test_probs)
```

**Characteristics:**
- Parametric method using Beta distribution
- Fits class-specific parameters
- Theoretically grounded in probability theory
- Good for severely miscalibrated classifiers

**When to use:**
- Parametric assumptions acceptable
- Systematic calibration bias exists
- Want theoretical guarantees

#### 1.5 Venn-ABERS Calibration
```python
from footy.models.calibration import VennAbersCalibrator

calibrator = VennAbersCalibrator(n_classes=3)
calibrator.fit(training_probs, training_outcomes)
calibrated_probs = calibrator.calibrate(test_probs)
```

**Characteristics:**
- Non-parametric transductive calibration
- Provides provably valid confidence intervals
- Uses non-conformity scores
- Slower but more theoretically sound

**When to use:**
- Formal validity guarantees needed
- Can afford transductive approach
- Risk-sensitive applications

#### 1.6 Auto-Calibrator (Ensemble Selection)
```python
from footy.models.calibration import AutoCalibrator

auto_cal = AutoCalibrator(n_classes=3, cv_folds=5)
auto_cal.fit(training_probs, training_outcomes)
calibrated_probs = auto_cal.calibrate(test_probs)
print(f"Selected: {auto_cal.params['selected_method']}")
print(f"Scores: {auto_cal.method_scores}")
```

**Characteristics:**
- Tests all methods via cross-validation
- Selects method with lowest ECE (Expected Calibration Error)
- Automatically handles model selection
- Best for production when method is unknown

**When to use:**
- Unsure which method works best
- Want to automate calibration method selection
- Comfortable with CV overhead

### Calibration Diagnostics

#### Expected Calibration Error (ECE)
```python
from footy.models.calibration import expected_calibration_error

ece = expected_calibration_error(predicted_probs, actual_outcomes, n_bins=10)
# Returns: float in [0, 1], lower is better
# Measures average |predicted_prob - empirical_frequency|
```

**Interpretation:**
- 0.0: Perfect calibration
- < 0.05: Well-calibrated
- 0.05-0.10: Acceptable calibration
- > 0.10: Poor calibration

#### Maximum Calibration Error (MCE)
```python
from footy.models.calibration import maximum_calibration_error

mce = maximum_calibration_error(predicted_probs, actual_outcomes, n_bins=10)
# Returns: float in [0, 1]
# Worst-case calibration gap
```

#### Reliability Diagrams
```python
from footy.models.calibration import reliability_diagram_data

diagram_data = reliability_diagram_data(predicted_probs, actual_outcomes, n_bins=15)
# Returns:
# {
#   'bin_centers': [0.07, 0.12, 0.17, ...],
#   'bin_accuracies': [0.05, 0.10, 0.15, ...],
#   'bin_sizes': [32, 48, 55, ...],
#   'ece': 0.0347,
#   'perfect_calibration_line': [0.0, 0.01, 0.02, ...]
# }
```

### Calibration Manager (Persistence)

```python
from footy.models.calibration import CalibrationManager

# Initialize with database connection (optional)
manager = CalibrationManager(db_con=duckdb_connection)

# Fit and save a calibrator
selected_method = manager.fit_and_save(
    name="council_v2",
    probs=training_probs,
    outcomes=training_outcomes,
    method="auto"  # or "platt", "isotonic", "temperature", "beta"
)

# Later: apply calibrator
calibrated = manager.calibrate("council_v2", test_probs)

# Calibrators are persisted in DuckDB calibrators table:
# CREATE TABLE calibrators (
#   name VARCHAR PRIMARY KEY,
#   method VARCHAR,
#   params_json VARCHAR,
#   created_at TIMESTAMP
# )
```

## 2. Enhanced Drift Detection

### Location
`src/footy/self_learning.py` - DriftDetector class

### Purpose
Detects when model predictions become unreliable due to:
- Sudden changes in data distribution (sudden drift)
- Gradual shifts in performance (gradual drift)
- Recurring patterns (concept recurrence)
- Feature distribution changes

### Implemented Methods

#### 2.1 Page-Hinkley Test (Original)
Cumulative sum test detecting changes in mean error rate.
- Fast, simple
- Effective for sudden changes
- Less sensitive to gradual drift

#### 2.2 ADWIN (Adaptive Windowing)
Maintains variable-length window, detects change points automatically.
- Adaptive window size
- Detects both sudden and gradual drift
- More computationally intensive

#### 2.3 DDM (Drift Detection Method)
Monitors error rate for significant increases.
- Tracks warning and drift zones
- Good for binary error signals
- Fast and simple

#### 2.4 EDDM (Early Drift Detection Method)
Monitors distance between consecutive errors.
- Detects drift earlier than DDM
- More sensitive to early indicators
- Better for early warning systems

#### 2.5 Ensemble Approach (Default)
Combines all methods, alerts when consensus reached (2+ detectors agree).

```python
from footy.self_learning import DriftDetector

detector = DriftDetector(use_ensemble=True)

# Feed loss values
for loss in loss_values:
    drift_detected = detector.update(loss, is_error=0 if correct else 1)
    if drift_detected:
        severity = detector.drift_severity  # "sudden", "gradual", "recurring"
        print(f"Drift detected: {severity}")
```

### Drift Severity Classification

The ensemble detector classifies drift type:

- **"sudden"**: Large rapid increase in error rate (> 15% change in 30 predictions)
- **"gradual"**: Slow, steady increase (5-15% change)
- **"recurring"**: Oscillating pattern (< 5% change, non-monotonic)
- **"none"**: No drift

```python
if detector.is_drifting:
    severity = detector.drift_severity
    if severity == "sudden":
        # Urgent retrain needed
        trigger_emergency_retrain()
    elif severity == "gradual":
        # Plan scheduled retrain
        schedule_retrain()
    elif severity == "recurring":
        # Monitor closely, may be seasonal
        increase_monitoring()
```

## 3. Enhanced Self-Learning Weights

### Location
`src/footy/self_learning.py` - ExpertTracker.get_weight() and SelfLearningLoop.get_optimal_expert_weights()

### Enhancements

#### 3.1 Inverse-Variance Weighting
```python
# Experts with consistent, low-loss predictions get higher weight
# weight ∝ 1 / (log_loss + variance)
```

#### 3.2 Bayesian Shrinkage (Hierarchical Model)
```python
# Blend league/context specific weights with global using:
# weight_blended = shrinkage_factor * specific_weight + (1 - shrinkage_factor) * global_weight
# shrinkage_factor = n_specific / (n_specific + 30)  # Prior strength = 30
```

#### 3.3 Recency Weighting (Exponential Decay)
```python
# Recent predictions weighted more heavily:
# recency_weight = 1 - exp(-0.02 * n_predictions)
# Configurable halflife (default 50 predictions)
```

#### 3.4 Per-Context Weights
```python
# Track weights separately for different contexts:
# - League contexts (EPL, Ligue1, Bundesliga, etc.)
# - Match contexts (derby, relegation battle, title race, etc.)

weights = loop.get_optimal_expert_weights(
    league="EPL",
    context="derby"
)
# Returns weights optimized for EPL derbies specifically
```

#### 3.5 Uncertainty Estimates
```python
result = loop.get_optimal_expert_weights(
    league="EPL",
    context=None,
    return_uncertainties=True
)

# Returns:
# {
#   "weights": {"expert_a": 0.35, "expert_b": 0.30, ...},
#   "uncertainties": {"expert_a": 0.08, "expert_b": 0.12, ...},
#   "meta": {
#     "league": "EPL",
#     "context": None,
#     "n_experts": 18,
#     "avg_weight_uncertainty": 0.095
#   }
# }
```

**Uncertainty interpretation:**
- Lower uncertainty = higher confidence in weight
- High uncertainty (> 0.15) = weight should be trusted less
- Use uncertainty for adaptive confidence intervals

### Usage Examples

#### Example 1: Record Prediction Result
```python
from footy.self_learning import get_learning_loop

loop = get_learning_loop()

result = loop.record_prediction_result(
    match_id=12345,
    predicted_probs=[0.45, 0.30, 0.25],  # [Home, Draw, Away]
    actual_outcome=0,  # Home won
    expert_predictions={
        "elo": [0.48, 0.28, 0.24],
        "poisson": [0.42, 0.32, 0.26],
        "bayesian": [0.44, 0.31, 0.25],
    },
    league="EPL",
    context="derby",
    model_version="council_v2",
)

print(result)
# {
#   "correct": True,
#   "log_loss": 0.7981,
#   "prob_of_actual": 0.45,
#   "drift_detected": False,
#   "drift_severity": "none",
#   "retrain_recommended": False,
#   "retrain_reasons": [],
#   "overall_accuracy": 0.5230,
#   "overall_ece": 0.0432,
#   "n_predictions": 125,
# }
```

#### Example 2: Get Context-Specific Weights
```python
# Get weights optimized for EPL matches
epl_weights = loop.get_optimal_expert_weights(league="EPL")

# Get weights optimized for EPL derbies specifically
derby_weights = loop.get_optimal_expert_weights(league="EPL", context="derby")

# Get weights with uncertainty estimates
weights_with_unc = loop.get_optimal_expert_weights(
    league="EPL",
    return_uncertainties=True
)

print(weights_with_unc["weights"])
# {"elo": 0.32, "poisson": 0.25, "bayesian": 0.43}

print(weights_with_unc["uncertainties"])
# {"elo": 0.08, "poisson": 0.12, "bayesian": 0.06}
```

#### Example 3: Monitor Drift
```python
loop = get_learning_loop()

for match in recent_matches:
    result = loop.record_prediction_result(...)

    if result["drift_detected"]:
        severity = result["drift_severity"]

        if severity == "sudden":
            log.warning(f"URGENT: Sudden drift detected in {result['league']}")
            trigger_emergency_retrain()

        elif severity == "gradual":
            log.warning(f"Gradual drift detected, scheduling retrain")
            schedule_retrain(delay_minutes=60)

        elif severity == "recurring":
            log.info(f"Recurring pattern detected, monitor closely")
```

#### Example 4: Performance Report
```python
report = loop.get_performance_report()

print(f"Overall Accuracy: {report['overall']['accuracy']:.4f}")
print(f"Log Loss: {report['overall']['log_loss']:.4f}")
print(f"Calibration Error: {report['overall']['calibration_error']:.4f}")
print(f"Retrain Recommended: {report['retrain_recommended']}")

for expert in report['expert_rankings'][:5]:
    print(f"  {expert['name']}: {expert['weight']:.3f} (acc={expert['accuracy']:.3f})")
```

## Integration with Database

Both modules are designed to work with DuckDB for persistence:

### Calibration Tables
```sql
CREATE TABLE IF NOT EXISTS calibrators (
    name VARCHAR PRIMARY KEY,
    method VARCHAR,
    params_json VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Self-Learning Tables (Already Exist)
- `expert_performance`: Global expert stats
- `expert_performance_by_comp`: Per-league expert stats
- `ensemble_weights`: Expert weights (legacy)

## API Reference

### calibration.py

```python
# Classes
PlattCalibrator(n_classes=3)
IsotonicCalibrator(n_classes=3)
TemperatureCalibrator()
BetaCalibrator(n_classes=3)
VennAbersCalibrator(n_classes=3)
AutoCalibrator(n_classes=3, cv_folds=5)
CalibrationManager(db_con=None)

# Functions
expected_calibration_error(probs, outcomes, n_bins=10, class_idx=None) -> float
maximum_calibration_error(probs, outcomes, n_bins=10) -> float
reliability_diagram_data(probs, outcomes, n_bins=15) -> dict
calibrate(probs, method='temperature') -> np.ndarray
fit_calibrator(train_probs, train_outcomes, method='platt') -> BaseCalibrator
```

### self_learning.py

```python
# Enhanced DriftDetector
DriftDetector(
    delta=0.005,
    lambda_threshold=50.0,
    min_instances=30,
    use_ensemble=True
)
detector.update(value, is_error=None) -> bool
detector.drift_severity -> str

# Enhanced SelfLearningLoop
SelfLearningLoop()
loop.record_prediction_result(
    match_id, predicted_probs, actual_outcome,
    expert_predictions=None, league="", context="", model_version=""
) -> dict

loop.get_optimal_expert_weights(
    league="", context="", return_uncertainties=False
) -> dict

loop.get_performance_report() -> dict
```

## Performance Characteristics

| Method | Speed | Memory | Accuracy | Flexibility | Use Case |
|--------|-------|--------|----------|-------------|----------|
| Platt | Fast | Low | Good | Low | Real-time, simple |
| Isotonic | Fast | Medium | Excellent | High | When data available |
| Temperature | Fastest | Low | Very Good | Very Low | NN calibration |
| Beta | Medium | Low | Good | Medium | Severe miscalibration |
| Venn-ABERS | Slow | Medium | Excellent | High | Risk-critical |
| Auto | Slow (CV) | Medium | Excellent | High | Production unknown |

## References

1. Guo et al. (2017) "On Calibration of Modern Neural Networks" - Temperature scaling
2. Niculescu-Mizil & Caruana (2005) "Predicting Good Probabilities" - Platt & isotonic
3. DeGroot & Fienberg (1983) "The Comparison and Evaluation of Forecasters" - ECE/reliability
4. Gama et al. (2014) "A survey on concept drift adaptation" - Drift detection overview
5. Bifet & Gavalda (2007) "Learning from Time-Changing Data" - ADWIN
6. Baena-García et al. (2006) "Early Drift Detection Method" - EDDM

## Testing

The modules include comprehensive error handling and graceful fallbacks:
- scipy imports are optional (fallbacks provided)
- Network issues don't break persistence (warnings logged)
- Invalid data is handled with sensible defaults

## Future Enhancements

1. **Histogram Binning Calibration** - Similar to isotonic but more stable
2. **Matrix Scaling** - For ordered multi-class problems
3. **Spiegelhalter Z-statistic** - Statistical significance testing
4. **Adaptive Calibration** - Online calibration as new data arrives
5. **Calibration Curves** - ROC-like curves for calibration assessment
6. **Seasonal Drift Detection** - Distinguish seasonal from true drift
7. **Expert Combination Methods** - Dynamic combination weights based on calibration

## Troubleshooting

### High ECE after calibration?
- Check if you have enough calibration data (ideally 200+ samples)
- Try different n_bins parameter (10-20 typically good)
- Auto-calibrator may select wrong method; try all manually

### Drift detected constantly?
- Check if threshold parameters are too sensitive
- Ensure you're passing is_error correctly (1=error, 0=correct)
- Increasing min_instances reduces false positives

### Weights always near uniform?
- Experts may be equally good (check individual accuracies)
- May need more data (weights stabilize with n>100)
- Context may be too granular (merge related contexts)
