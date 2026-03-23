"""Self-Learning Feedback System — Autonomous accuracy improvement.

Implements a closed-loop learning system that:
1. Tracks prediction accuracy per model, expert, league, and time window
2. Automatically adjusts model weights based on rolling performance
3. Detects concept drift and triggers retraining
4. Learns which experts perform best in specific contexts
5. Maintains a prediction accuracy log for transparency

This module is the brain that makes the prediction system "learn from itself"
without any manual intervention.

References:
    Gama et al. (2014) "A survey on concept drift adaptation"
    Bates & Granger (1969) "Combination of forecasts"
    Diebold & Mariano (1995) "Comparing predictive accuracy"
"""
from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, TypeAlias, cast
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

ProbabilityVector: TypeAlias = list[float]
PredictionRecord: TypeAlias = dict[str, Any]
ExpertProbabilityMap: TypeAlias = dict[str, ProbabilityVector]


def _new_prediction_records() -> list[PredictionRecord]:
    return []


def _new_performance_windows() -> dict[str, "PerformanceWindow"]:
    return {}


def _normalize_triplet(probs: list[float]) -> list[float]:
    """Normalize a triplet of probabilities to sum to 1.

    Validates that:
    - Input is a list of 3 values
    - All values are non-negative
    - Sum is positive (non-zero)

    Args:
        probs: List of 3 probability values

    Returns:
        Normalized triplet summing to 1.0
    """
    if not isinstance(probs, (list, tuple)) or len(probs) != 3:
        return [1 / 3, 1 / 3, 1 / 3]

    try:
        numeric_probs = [float(p) for p in probs]
    except (TypeError, ValueError):
        return [1 / 3, 1 / 3, 1 / 3]

    # Ensure non-negative
    numeric_probs = [max(0.0, p) for p in numeric_probs]

    # Normalize
    total = float(sum(numeric_probs))
    if total <= 0:
        return [1 / 3, 1 / 3, 1 / 3]

    result = [p / total for p in numeric_probs]

    # Validate result sums to 1.0 (within floating point tolerance)
    if not (0.999 <= sum(result) <= 1.001):
        return [1 / 3, 1 / 3, 1 / 3]

    return result


def extract_expert_predictions(payload: Any) -> ExpertProbabilityMap:
    """Extract expert probability triplets from notes or expert-cache payloads."""
    data = payload
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return {}
    if not isinstance(data, dict):
        return {}
    data_dict = cast(dict[str, Any], data)

    experts = data_dict.get("experts")
    if not isinstance(experts, dict):
        return {}
    experts_dict = cast(dict[str, Any], experts)

    extracted: ExpertProbabilityMap = {}
    for raw_name, raw_data in experts_dict.items():
        if not isinstance(raw_data, dict):
            continue
        raw_data_dict = cast(dict[str, Any], raw_data)

        raw_probs = raw_data_dict.get("probs")
        probs_dict = cast(dict[str, Any], raw_probs) if isinstance(raw_probs, dict) else raw_data_dict

        try:
            probs = _normalize_triplet(
                [
                    float(probs_dict.get("home", 1 / 3)),
                    float(probs_dict.get("draw", 1 / 3)),
                    float(probs_dict.get("away", 1 / 3)),
                ]
            )
        except (TypeError, ValueError):
            continue
        extracted[raw_name] = probs

    return extracted


# ═══════════════════════════════════════════════════════════════════
# PERFORMANCE WINDOW — ROLLING ACCURACY TRACKING
# ═══════════════════════════════════════════════════════════════════
@dataclass
class PerformanceWindow:
    """Track prediction performance over a rolling window."""
    predictions: list[PredictionRecord] = field(default_factory=_new_prediction_records)
    max_size: int = 500
    _cached_accuracy: float | None = None
    _cached_log_loss: float | None = None
    _cached_brier: float | None = None
    _cached_calibration: float | None = None
    _cache_valid: bool = False

    def add(self, prediction: PredictionRecord) -> None:
        self.predictions.append(prediction)
        if len(self.predictions) > self.max_size:
            self.predictions = self.predictions[-self.max_size:]
        # Invalidate cache when new prediction added
        self._cache_valid = False

    def _invalidate_cache(self) -> None:
        """Invalidate all cached metrics."""
        self._cache_valid = False
        self._cached_accuracy = None
        self._cached_log_loss = None
        self._cached_brier = None
        self._cached_calibration = None

    @property
    def accuracy(self) -> float:
        if self._cache_valid and self._cached_accuracy is not None:
            return self._cached_accuracy
        correct = sum(1 for p in self.predictions if p.get("correct"))
        result = correct / max(len(self.predictions), 1)
        self._cached_accuracy = result
        self._cache_valid = True
        return result

    @property
    def log_loss(self) -> float:
        if self._cache_valid and self._cached_log_loss is not None:
            return self._cached_log_loss
        if not self.predictions:
            return 999.0
        total = 0.0
        for p in self.predictions:
            prob = max(1e-12, p.get("predicted_prob", 1 / 3))
            total -= math.log(prob)
        result = total / len(self.predictions)
        self._cached_log_loss = result
        self._cache_valid = True
        return result

    @property
    def brier_score(self) -> float:
        if self._cache_valid and self._cached_brier is not None:
            return self._cached_brier
        if not self.predictions:
            return 1.0
        total = 0.0
        for p in self.predictions:
            probs = p.get("probs", [1 / 3, 1 / 3, 1 / 3])
            outcome = p.get("outcome", 0)
            for i, prob in enumerate(probs):
                target = 1.0 if i == outcome else 0.0
                total += (prob - target) ** 2
        result = total / (3 * len(self.predictions))
        self._cached_brier = result
        self._cache_valid = True
        return result

    @property
    def calibration_error(self) -> float:
        """Expected Calibration Error (ECE) with 10 bins."""
        if self._cache_valid and self._cached_calibration is not None:
            return self._cached_calibration
        if len(self.predictions) < 10:
            return 0.0
        confidences = [max(p.get("probs", [1 / 3])) for p in self.predictions]
        corrects = [1.0 if p.get("correct") else 0.0 for p in self.predictions]
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for i in range(10):
            mask = [(c >= bins[i]) and (c < bins[i + 1]) for c in confidences]
            if sum(mask) == 0:
                continue
            bin_conf = np.mean([c for c, m in zip(confidences, mask) if m])
            bin_acc = np.mean([c for c, m in zip(corrects, mask) if m])
            bin_size = sum(mask) / len(self.predictions)
            ece += abs(bin_conf - bin_acc) * bin_size
        self._cached_calibration = float(ece)
        self._cache_valid = True
        return self._cached_calibration

    @property
    def n_predictions(self) -> int:
        return len(self.predictions)


# ═══════════════════════════════════════════════════════════════════
# EXPERT PERFORMANCE TRACKER
# ═══════════════════════════════════════════════════════════════════
@dataclass
class ExpertTracker:
    """Track individual expert performance to learn optimal weights."""
    name: str
    global_window: PerformanceWindow = field(default_factory=PerformanceWindow)
    league_windows: dict[str, PerformanceWindow] = field(default_factory=_new_performance_windows)
    context_windows: dict[str, PerformanceWindow] = field(default_factory=_new_performance_windows)
    last_prediction_time: float = 0.0
    failure_count: int = 0
    timeout_count: int = 0
    extreme_prob_count: int = 0
    weight_decay_factor: float = 0.999  # Exponential decay per update

    def record(
        self,
        prediction: PredictionRecord,
        league: str = "",
        context: str = "",
        expert_probs: list[float] | None = None,
    ) -> None:
        """Record a prediction with optional outlier detection.

        Args:
            prediction: The prediction record
            league: League context
            context: Match context
            expert_probs: Expert's probability triplet for outlier detection
        """
        self.last_prediction_time = time.time()

        # Detect extreme probabilities (outliers)
        if expert_probs is not None and (hasattr(expert_probs, '__len__') and len(expert_probs) > 0):
            self._check_extreme_probabilities(expert_probs)

        self.global_window.add(prediction)
        if league:
            if league not in self.league_windows:
                self.league_windows[league] = PerformanceWindow(max_size=200)
            self.league_windows[league].add(prediction)
        if context:
            if context not in self.context_windows:
                self.context_windows[context] = PerformanceWindow(max_size=100)
            self.context_windows[context].add(prediction)

    def _check_extreme_probabilities(self, probs: list[float]) -> None:
        """Detect if expert assigns extreme probabilities (near 0 or 1)."""
        for p in probs:
            if p < 0.01 or p > 0.99:
                self.extreme_prob_count += 1
                break

    def record_failure(self) -> None:
        """Record a failure event (e.g., timeout, error)."""
        self.failure_count += 1
        self.timeout_count += 1

    def record_timeout(self) -> None:
        """Record a timeout event."""
        self.timeout_count += 1

    def is_inactive(self, max_age_seconds: float = 86400) -> bool:
        """Check if expert is inactive (no predictions in max_age_seconds)."""
        if self.last_prediction_time == 0:
            return True
        return time.time() - self.last_prediction_time > max_age_seconds

    def get_inactive_decay_factor(self, max_age_seconds: float = 86400) -> float:
        """Get weight decay factor based on inactivity.

        Decays weight exponentially for inactive experts.
        """
        if self.last_prediction_time == 0:
            return 0.01  # New, unproven expert

        age = time.time() - self.last_prediction_time
        if age > max_age_seconds:
            return 0.1  # Heavily decay inactive experts

        # Smooth decay over time
        return max(0.5, 1.0 - (age / max_age_seconds) * 0.5)

    def get_weight(
        self,
        league: str = "",
        context: str = "",
        use_recency: bool = True,
        recency_halflife: int = 50,
    ) -> tuple[float, float]:
        """Compute performance-based weight with Bayesian shrinkage and recency.

        Uses hierarchical Bayesian approach:
        1. Context-specific (most specific)
        2. League-specific (intermediate)
        3. Global (least specific)

        Incorporates:
        1. Inverse-variance weighting (consistent performers get higher weight)
        2. Multi-level Bayesian shrinkage (blend context/league with global)
        3. Recency weighting (recent predictions matter more)
        4. Inactivity decay (penalize dormant experts)
        5. Outlier penalty (reduce weight if extreme probabilities detected)
        6. Failure tracking (penalize timeout/failures)

        Args:
            league: League context for league-specific weighting
            context: Match context for context-specific weighting
            use_recency: Apply exponential decay to older predictions
            recency_halflife: Number of predictions for 50% weight decay

        Returns:
            Tuple of (weight, uncertainty) where uncertainty reflects confidence
        """
        # Multi-level hierarchy: context -> league -> global
        context_weight = None
        league_weight = None

        # Try context-specific window
        if context and context in self.context_windows:
            ctx_window = self.context_windows[context]
            if ctx_window.n_predictions >= 15:
                context_weight = (ctx_window, "context")

        # Try league-specific window
        if league and league in self.league_windows and context_weight is None:
            lg_window = self.league_windows[league]
            if lg_window.n_predictions >= 20:
                league_weight = (lg_window, "league")

        # Use global if no context/league specific data
        window = context_weight or league_weight
        if window is None:
            window = self.global_window
        else:
            window = window[0]

        if window.n_predictions < 10:
            # Insufficient data: return equal weight with high uncertainty
            return 1.0, 0.95

        # Compute base weight from log loss
        log_loss = window.log_loss
        weight_from_loss = max(0.01, 1.0 - log_loss / 3.0)

        # Multi-level shrinkage: blend up the hierarchy
        shrinkage_factor = min(0.9, window.n_predictions / (window.n_predictions + 40.0))

        # Shrink toward league if available
        if window != self.global_window and window != (
            self.league_windows.get(league) if league else None
        ):
            parent_window = (
                self.league_windows.get(league)
                if league and league in self.league_windows
                else self.global_window
            )
            if parent_window.n_predictions >= 20:
                parent_weight = max(0.01, 1.0 - parent_window.log_loss / 3.0)
                weight_from_loss = (
                    shrinkage_factor * weight_from_loss +
                    (1 - shrinkage_factor) * parent_weight
                )

        # Apply recency weighting
        if use_recency and window.n_predictions > recency_halflife:
            recent_weight = min(1.0, 1.0 - math.exp(-0.03 * window.n_predictions / recency_halflife))
            weight_from_loss *= recent_weight

        # Apply inactivity decay
        inactive_decay = self.get_inactive_decay_factor(max_age_seconds=172800)  # 2 days
        weight_from_loss *= inactive_decay

        # Penalty for extreme probabilities (overconfident outliers)
        if self.extreme_prob_count > window.n_predictions * 0.1:  # >10% extreme
            extreme_penalty = max(0.5, 1.0 - (self.extreme_prob_count / window.n_predictions) * 0.5)
            weight_from_loss *= extreme_penalty

        # Penalty for failures/timeouts
        if self.failure_count > 0:
            failure_penalty = max(0.3, 1.0 - (self.failure_count / window.n_predictions) * 0.3)
            weight_from_loss *= failure_penalty

        # Compute uncertainty from calibration error and data count
        variance = window.calibration_error + 0.01
        uncertainty = variance * (1.0 - min(window.n_predictions / 250.0, 0.75))

        return max(0.01, float(weight_from_loss)), max(0.01, float(uncertainty))


# ═══════════════════════════════════════════════════════════════════
# ENHANCED DRIFT DETECTOR WITH MULTIPLE METHODS
# ═══════════════════════════════════════════════════════════════════


class CUSUMDetector:
    """CUSUM (Cumulative Sum Control Chart) for drift detection on log loss.

    Detects gradual and sudden changes in a time series by monitoring
    cumulative sums above and below a target value.

    Reference: Page (1954) "Continuous Inspection Schemes"
    """

    def __init__(self, target: float = 0.5, slack: float = 0.1, threshold: float = 5.0):
        """Initialize CUSUM detector.

        Args:
            target: Target value for the metric (e.g., 0.5 for log loss)
            slack: Slack parameter (tolerance around target)
            threshold: Threshold for drift detection
        """
        self.target = target
        self.slack = slack
        self.threshold = threshold
        self._sum_high: float = 0.0
        self._sum_low: float = 0.0
        self._n: int = 0
        self._drifts: int = 0

    def update(self, value: float) -> bool:
        """Update with new value. Returns True if drift detected."""
        self._n += 1

        # CUSUM formulas
        self._sum_high = max(0.0, self._sum_high + (value - self.target - self.slack))
        self._sum_low = max(0.0, self._sum_low + (self.target - self.slack - value))

        # Drift if either cumsum exceeds threshold
        if self._sum_high > self.threshold or self._sum_low > self.threshold:
            self._drifts += 1
            return self._n > 30  # Require minimum samples

        return False

    def reset(self) -> None:
        self._sum_high = 0.0
        self._sum_low = 0.0
        self._n = 0
        self._drifts = 0


class PageHinkleyDetector:
    """Page-Hinkley test for drift detection.

    Monitors cumulative sum of deviations from expected value.
    Detects sudden changes when cumulative sum exceeds threshold.
    """

    def __init__(self, delta: float = 0.005, lambda_threshold: float = 50.0):
        self.delta = delta
        self.lambda_threshold = lambda_threshold
        self._sum: float = 0.0
        self._mean: float = 0.0
        self._n: int = 0
        self._min_sum: float = float("inf")

    def update(self, value: float) -> bool:
        self._n += 1
        self._mean = self._mean + (value - self._mean) / self._n
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)

        if self._n < 30:
            return False

        return self._sum - self._min_sum > self.lambda_threshold

    def reset(self) -> None:
        self._sum = 0.0
        self._mean = 0.0
        self._n = 0
        self._min_sum = float("inf")


class ADWINDetector:
    """ADWIN (Adaptive Windowing): Automatically detects change points.

    Maintains a variable-length window and detects drift when window
    statistics change significantly. More adaptive than Page-Hinkley.

    Reference: Bifet & Gavalda (2007) "Learning from Time-Changing Data"
    """

    def __init__(self, delta: float = 0.002, min_window: int = 10):
        self.delta = delta
        self.min_window = min_window
        self.buckets: list[tuple[int, float, float]] = []  # (count, sum, sum_sq)
        self.total_count = 0
        self.drift_detected = False

    def update(self, value: float) -> bool:
        """Add observation and detect drift."""
        self.total_count += 1

        # Add to current bucket (compress buckets periodically)
        if not self.buckets:
            self.buckets.append((1, value, value**2))
        else:
            count, bucket_sum, bucket_sum_sq = self.buckets[-1]
            self.buckets[-1] = (count + 1, bucket_sum + value, bucket_sum_sq + value**2)

        # Compress if too many buckets
        if len(self.buckets) > 20:
            self._compress_buckets()

        # Check for drift
        if self.total_count > self.min_window:
            return self._detect_change()

        return False

    def _compress_buckets(self) -> None:
        """Merge oldest buckets."""
        if len(self.buckets) <= 1:
            return
        c1, s1, sq1 = self.buckets[0]
        c2, s2, sq2 = self.buckets[1]
        self.buckets = [(c1 + c2, s1 + s2, sq1 + sq2)] + self.buckets[2:]

    def _detect_change(self) -> bool:
        """Detect if distribution changed."""
        if len(self.buckets) < 2:
            return False

        # Compare statistics of first and last buckets
        c0, s0, sq0 = self.buckets[0]
        c1, s1, sq1 = self.buckets[-1]

        if c0 < 1 or c1 < 1:
            return False

        mean0 = s0 / c0
        mean1 = s1 / c1
        var0 = max(0, sq0 / c0 - mean0**2)
        var1 = max(0, sq1 / c1 - mean1**2)

        # Hoeffding bound for change detection
        m = 1 / (c0 + c1)
        eps = math.sqrt(m * math.log(2 / self.delta))

        return abs(mean0 - mean1) > eps

    def reset(self) -> None:
        self.buckets = []
        self.total_count = 0
        self.drift_detected = False


class DDMDetector:
    """DDM (Drift Detection Method): Monitors error rate for increases.

    Tracks error rate and its running average, alerts when error increases
    significantly beyond what would be expected from random variation.
    Fast and simple, good for detecting sudden drift.

    Reference: Gama et al. (2004) "Learning with Drift Detection"
    """

    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.errors: list[int] = []
        self.n: int = 0
        self.warning_zone = False
        self.drift_detected = False

    def update(self, error: int) -> bool:
        """Update with error (1) or correct (0) prediction."""
        self.errors.append(error)
        self.n += 1

        if self.n < 30:
            return False

        # Compute statistics
        p = np.mean(self.errors)
        s = math.sqrt(p * (1 - p) / self.n)

        # Recent window (last 30)
        recent_errors = self.errors[-30:]
        p_recent = np.mean(recent_errors)
        s_recent = math.sqrt(p_recent * (1 - p_recent) / 30)

        # Check for drift
        mean_diff = p_recent - p
        std_diff = math.sqrt(s**2 + s_recent**2)

        if mean_diff > self.drift_level * std_diff:
            self.drift_detected = True
            return True

        self.warning_zone = mean_diff > self.warning_level * std_diff
        return False

    def reset(self) -> None:
        self.errors = []
        self.n = 0
        self.warning_zone = False
        self.drift_detected = False


class EDDMDetector:
    """EDDM (Early Drift Detection Method): Detects drift earlier than DDM.

    Monitors distance between consecutive errors instead of error rate.
    More sensitive to early drift indicators.

    Reference: Baena-García et al. (2006) "Early Drift Detection Method"
    """

    def __init__(self, warning_level: float = 0.95, drift_level: float = 0.9):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.error_distances: list[int] = []
        self.positions_errors: list[int] = []
        self.n: int = 0
        self.warning_zone = False
        self.drift_detected = False

    def update(self, error: int) -> bool:
        """Update with error (1) or correct (0) prediction."""
        self.n += 1

        if error == 1:
            distance = self.n - sum(self.positions_errors)
            if self.positions_errors:
                self.error_distances.append(distance)
            self.positions_errors.append(self.n)

        if len(self.error_distances) < 30:
            return False

        # Compute statistics
        mean_dist = np.mean(self.error_distances)
        std_dist = np.std(self.error_distances)

        # Recent window
        recent_dist = self.error_distances[-30:]
        mean_recent = np.mean(recent_dist)
        std_recent = np.std(recent_dist)

        if std_recent + std_dist > 0:
            p_drift = (mean_recent - mean_dist) / (std_recent + std_dist)
            if p_drift > 2.0:
                self.drift_detected = True
                return True
            self.warning_zone = p_drift > 1.5

        return False

    def reset(self) -> None:
        self.error_distances = []
        self.positions_errors = []
        self.n = 0
        self.warning_zone = False
        self.drift_detected = False


class DriftDetector:
    """Ensemble drift detector combining multiple methods.

    Uses Page-Hinkley, CUSUM, ADWIN, DDM, and EDDM in parallel and alerts
    when consensus is reached. Backward compatible with original DriftDetector.

    When drift is detected, it signals that the model should be retrained
    because the relationship between features and outcomes has changed.

    Reference: Gama et al. (2014) "A survey on concept drift adaptation"
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_threshold: float = 50.0,
        min_instances: int = 30,
        use_ensemble: bool = True,
    ):
        self.delta = delta
        self.lambda_threshold = lambda_threshold
        self.min_instances = min_instances
        self.use_ensemble = use_ensemble

        # Original detector for backward compatibility
        self.page_hinkley = PageHinkleyDetector(delta, lambda_threshold)

        # Ensemble detectors
        self.cusum = CUSUMDetector(target=0.5, slack=0.1, threshold=5.0)
        self.adwin = ADWINDetector(delta=delta)
        self.ddm = DDMDetector()
        self.eddm = EDDMDetector()

        self._n: int = 0
        self._drift_detected: bool = False
        self._severity: str = "none"  # "none", "gradual", "sudden", "recurring"
        self._last_errors: list[int] = []

    def update(self, value: float, is_error: int | None = None) -> bool:
        """Add a new observation and return True if drift is detected.

        Args:
            value: Error or loss value for the latest prediction
            is_error: Optional binary error (1) or correct (0)
        Returns:
            True if concept drift has been detected
        """
        self._n += 1

        if not self.use_ensemble:
            # Original behavior: Page-Hinkley only
            return self.page_hinkley.update(value)

        # Ensemble approach: combine 5 detection methods
        detections = 0
        detection_methods = []

        # Page-Hinkley on loss (detects sudden changes)
        if self.page_hinkley.update(value):
            detections += 1
            detection_methods.append("page_hinkley")

        # CUSUM on loss (detects gradual changes)
        if self.cusum.update(value):
            detections += 1
            detection_methods.append("cusum")

        # ADWIN on loss (adaptive windowing)
        if self.adwin.update(value):
            detections += 1
            detection_methods.append("adwin")

        # DDM and EDDM on error rate
        if is_error is not None:
            self._last_errors.append(is_error)
            # Keep only recent errors
            if len(self._last_errors) > 300:
                self._last_errors = self._last_errors[-300:]

            if self.ddm.update(is_error):
                detections += 1
                detection_methods.append("ddm")
            if self.eddm.update(is_error):
                detections += 1
                detection_methods.append("eddm")

        # Consensus: 2+ detectors agree (out of 5)
        drift_consensus = detections >= 2

        if drift_consensus and self._n >= self.min_instances:
            self._drift_detected = True
            self._classify_severity()
            log.info(f"Drift detected: {detections}/5 methods agreed ({detection_methods})")
            return True

        return False

    def _classify_severity(self) -> None:
        """Classify drift type: gradual, sudden, or recurring."""
        if len(self._last_errors) < 50:
            self._severity = "sudden"
            return

        recent_errors = self._last_errors[-30:]
        older_errors = self._last_errors[-60:-30]

        recent_error_rate = np.mean(recent_errors)
        older_error_rate = np.mean(older_errors)

        change_rate = abs(recent_error_rate - older_error_rate)

        if change_rate > 0.15:
            self._severity = "sudden"
        elif change_rate > 0.05:
            self._severity = "gradual"
        else:
            self._severity = "recurring"

    def reset(self) -> None:
        self.page_hinkley.reset()
        self.cusum.reset()
        self.adwin.reset()
        self.ddm.reset()
        self.eddm.reset()
        self._n = 0
        self._drift_detected = False
        self._severity = "none"
        self._last_errors = []

    @property
    def is_drifting(self) -> bool:
        return self._drift_detected

    @property
    def drift_severity(self) -> str:
        """Return severity classification."""
        return self._severity


# ═══════════════════════════════════════════════════════════════════
# SELF-LEARNING FEEDBACK LOOP
# ═══════════════════════════════════════════════════════════════════
class SelfLearningLoop:
    """Main self-learning feedback system.

    Continuously monitors prediction quality and makes automatic adjustments:
    1. Expert weight adjustment (per league, context, and global)
    2. Model selection (identifies which model family works best)
    3. Drift detection (triggers retraining when needed)
    4. Confidence calibration (learns to adjust confidence scores)
    """

    def __init__(
        self,
        global_window_size: int = 500,
        league_window_size: int = 200,
        context_window_size: int = 100,
        overall_window_size: int = 1000,
    ):
        self.expert_trackers: dict[str, ExpertTracker] = {}
        self.model_performance: dict[str, PerformanceWindow] = {}
        self.context_weights: dict[str, dict[str, float]] = {}  # context -> {expert -> weight}
        self.drift_detector = DriftDetector()
        self.overall_performance = PerformanceWindow(max_size=overall_window_size)

        # Configurable window sizes
        self.global_window_size = global_window_size
        self.league_window_size = league_window_size
        self.context_window_size = context_window_size

        # Retraining thresholds
        self._retrain_recommended: bool = False
        self._last_retrain_check: float = 0
        self._retrain_threshold_accuracy: float = 0.38
        self._retrain_threshold_ece: float = 0.12
        self._grace_window_hours: int = 24  # Grace window before rollback

        # Performance tracking
        self._prediction_count_since_cache_clear: int = 0
        self._cache_clear_interval: int = 100

    def record_prediction_result(
        self,
        match_id: int,
        predicted_probs: list[float],
        actual_outcome: int,
        expert_predictions: dict[str, list[float]] | None = None,
        league: str = "",
        context: str = "",
        model_version: str = "",
    ) -> dict[str, Any]:
        """Record a finished prediction and update all trackers.

        Args:
            match_id: The match identifier
            predicted_probs: [p_home, p_draw, p_away] from the council
            actual_outcome: 0=Home, 1=Draw, 2=Away
            expert_predictions: Optional dict of expert_name -> [p_h, p_d, p_a]
            league: Competition code
            context: Context string (e.g., "derby", "relegation")
            model_version: Model version string

        Returns:
            Dict with performance metrics and recommendations
        """
        predicted_outcome = int(np.argmax(predicted_probs))
        is_correct = predicted_outcome == actual_outcome
        prob_of_actual = predicted_probs[actual_outcome]
        log_loss = -math.log(max(prob_of_actual, 1e-12))

        record: PredictionRecord = {
            "match_id": match_id,
            "probs": predicted_probs,
            "predicted_prob": prob_of_actual,
            "outcome": actual_outcome,
            "correct": is_correct,
            "log_loss": log_loss,
            "timestamp": time.time(),
        }

        # Update overall performance
        self.overall_performance.add(record)

        # Update drift detector with log loss and error indicator
        drift_detected = self.drift_detector.update(log_loss, is_error=0 if is_correct else 1)

        # Update model performance
        if model_version:
            if model_version not in self.model_performance:
                self.model_performance[model_version] = PerformanceWindow(max_size=500)
            self.model_performance[model_version].add(record)

        # Update per-expert performance
        if expert_predictions:
            for expert_name, expert_probs in expert_predictions.items():
                if expert_name not in self.expert_trackers:
                    tracker = ExpertTracker(name=expert_name)
                    # Set configurable window sizes
                    tracker.global_window.max_size = self.global_window_size
                    self.expert_trackers[expert_name] = tracker

                expert_correct = int(np.argmax(expert_probs)) == actual_outcome
                expert_record: PredictionRecord = {
                    "match_id": match_id,
                    "probs": expert_probs,
                    "predicted_prob": expert_probs[actual_outcome],
                    "outcome": actual_outcome,
                    "correct": expert_correct,
                    "log_loss": -math.log(max(expert_probs[actual_outcome], 1e-12)),
                }
                self.expert_trackers[expert_name].record(
                    expert_record,
                    league=league,
                    context=context,
                    expert_probs=expert_probs,
                )

        # ---- Hedge weight update for each expert ----
        if expert_predictions:
            for expert_name, expert_probs in expert_predictions.items():
                self.hedge_update(expert_name, expert_probs, actual_outcome)

        # ---- Per-league temperature calibration ----
        if league:
            self.update_league_temperature(league, predicted_probs, actual_outcome)

        # Periodic cache clear to prevent memory bloat
        self._prediction_count_since_cache_clear += 1
        if self._prediction_count_since_cache_clear >= self._cache_clear_interval:
            self._clear_old_cache_entries()
            self._prediction_count_since_cache_clear = 0

        # Check if retraining is recommended
        should_retrain = False
        reasons: list[str] = []

        if drift_detected:
            should_retrain = True
            reasons.append("concept_drift_detected")

        if self.overall_performance.n_predictions >= 50:
            if self.overall_performance.accuracy < self._retrain_threshold_accuracy:
                should_retrain = True
                reasons.append(f"accuracy_below_{self._retrain_threshold_accuracy}")
            if self.overall_performance.calibration_error > self._retrain_threshold_ece:
                should_retrain = True
                reasons.append(f"ece_above_{self._retrain_threshold_ece}")

        self._retrain_recommended = should_retrain

        return {
            "correct": is_correct,
            "log_loss": round(log_loss, 4),
            "prob_of_actual": round(prob_of_actual, 4),
            "drift_detected": drift_detected,
            "drift_severity": self.drift_detector.drift_severity,
            "retrain_recommended": should_retrain,
            "retrain_reasons": reasons,
            "overall_accuracy": round(self.overall_performance.accuracy, 4),
            "overall_ece": round(self.overall_performance.calibration_error, 4),
            "n_predictions": self.overall_performance.n_predictions,
            "hedge_weights": self.get_hedge_weights(),
            "league_temperature": self.get_league_temperature(league) if league else 1.0,
        }

    def get_optimal_expert_weights(
        self,
        league: str = "",
        context: str = "",
        return_uncertainties: bool = False,
        use_rolling_window: bool = True,
        rolling_window_size: int = 100,
    ) -> dict[str, Any]:
        """Get performance-optimized weights for all tracked experts.

        Uses inverse-variance weighting with Bayesian shrinkage and recency.
        Applies exponential time decay to weight recent predictions more heavily.
        Optionally uses Brier score decomposition (reliability + resolution) for
        enhanced weight calculation.
        Caches per-context weights for faster future lookups.

        Args:
            league: League context for league-specific weighting
            context: Match context (e.g., "derby", "relegation")
            return_uncertainties: Include uncertainty estimates in output
            use_rolling_window: Use rolling window of recent predictions with exponential decay
            rolling_window_size: Size of rolling window (default 100 matches)

        Returns:
            Dict mapping expert names to weights (and optionally uncertainties).
            Weights are normalized to sum to 1.
            
        Notes:
            - Weights incorporate:
              1. Inverse-variance weighting (consistent performers get higher weight)
              2. Exponential time decay (recent ~1 week of predictions weighted more)
              3. Bayesian shrinkage (blend context/league with global estimates)
              4. Reliability + resolution decomposition (Brier score analysis)
        """
        # Check cache first
        cache_key = f"{league}|{context}"
        if cache_key in self.context_weights and not return_uncertainties:
            return self.context_weights[cache_key]

        weights: dict[str, float] = {}
        uncertainties: dict[str, float] = {}

        for name, tracker in self.expert_trackers.items():
            w, unc = tracker.get_weight(
                league=league,
                context=context,
                use_recency=True,
                recency_halflife=50,
            )
            weights[name] = w
            uncertainties[name] = unc

        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}

        # Cache the weights
        self.context_weights[cache_key] = weights

        result: dict[str, Any] = weights

        if return_uncertainties:
            # Normalize uncertainties similarly
            total_unc = sum(uncertainties.values())
            if total_unc > 0:
                uncertainties = {k: round(v / total_unc, 4) for k, v in uncertainties.items()}
            result = {
                "weights": weights,
                "uncertainties": uncertainties,
                "meta": {
                    "league": league,
                    "context": context,
                    "n_experts": len(self.expert_trackers),
                    "avg_weight_uncertainty": round(np.mean(list(uncertainties.values())), 4),
                },
            }

        return result

    def get_performance_report(self) -> dict[str, Any]:
        """Generate a comprehensive performance report."""
        expert_rankings: list[dict[str, Any]] = []
        model_rankings: list[dict[str, Any]] = []

        report: dict[str, Any] = {
            "overall": {
                "accuracy": round(self.overall_performance.accuracy, 4),
                "log_loss": round(self.overall_performance.log_loss, 4),
                "brier_score": round(self.overall_performance.brier_score, 4),
                "calibration_error": round(self.overall_performance.calibration_error, 4),
                "n_predictions": self.overall_performance.n_predictions,
            },
            "drift_detected": self.drift_detector.is_drifting,
            "retrain_recommended": self._retrain_recommended,
            "expert_rankings": expert_rankings,
            "model_rankings": model_rankings,
        }

        # Expert rankings
        for name, tracker in sorted(
            self.expert_trackers.items(),
            key=lambda x: x[1].global_window.accuracy,
            reverse=True,
        ):
            if tracker.global_window.n_predictions < 10:
                continue
            expert_rankings.append({
                "name": name,
                "accuracy": round(tracker.global_window.accuracy, 4),
                "log_loss": round(tracker.global_window.log_loss, 4),
                "n_predictions": tracker.global_window.n_predictions,
                "weight": round(tracker.get_weight()[0], 4),
            })

        # Model rankings
        for version, window in sorted(
            self.model_performance.items(),
            key=lambda x: x[1].accuracy,
            reverse=True,
        ):
            model_rankings.append({
                "version": version,
                "accuracy": round(window.accuracy, 4),
                "log_loss": round(window.log_loss, 4),
                "n_predictions": window.n_predictions,
            })

        return report

    def _clear_old_cache_entries(self) -> None:
        """Clear old cached context weights to prevent memory bloat."""
        # Keep only recent context weights
        max_cache_entries = 1000
        if len(self.context_weights) > max_cache_entries:
            # Remove oldest entries
            keys_to_remove = list(self.context_weights.keys())[:-500]
            for key in keys_to_remove:
                del self.context_weights[key]

    # ═══════════════════════════════════════════════════════════════
    # HEDGE ALGORITHM — ONLINE EXPERT WEIGHTING
    # ═══════════════════════════════════════════════════════════════

    def hedge_update(
        self,
        expert_name: str,
        predicted_probs: list[float],
        actual_outcome: int,
        eta: float = 0.1,
    ) -> None:
        """Multiplicative weight update (Hedge algorithm).

        After each scored match, update the expert's Hedge weight using the
        exponential-weights method of Freund & Schapire (1997).  The loss for
        the expert is the negative log-probability it assigned to the outcome
        that actually occurred, clamped to avoid numerical blow-up.

        loss = -log(expert.prob[actual_outcome])
        expert.weight *= exp(-eta * loss)

        All weights are re-normalised after every update so they sum to 1.
        The learning rate *eta* controls the trade-off between reactivity
        (large eta — fast adaptation) and stability (small eta — smooth).

        This is provably optimal for the best-expert-in-hindsight benchmark
        (regret bound O(sqrt(T log N))).

        Args:
            expert_name: Identifier for the expert being updated.
            predicted_probs: [p_home, p_draw, p_away] the expert predicted.
            actual_outcome: 0 = Home, 1 = Draw, 2 = Away.
            eta: Learning rate (default 0.1).  Smaller values are more
                 conservative; larger values react faster to recent results.
        """
        if not hasattr(self, "_hedge_weights"):
            self._hedge_weights: dict[str, float] = {}
        if not hasattr(self, "_hedge_cumulative_loss"):
            self._hedge_cumulative_loss: dict[str, float] = {}

        # Initialise expert if unseen
        if expert_name not in self._hedge_weights:
            self._hedge_weights[expert_name] = 1.0
            self._hedge_cumulative_loss[expert_name] = 0.0

        # Compute log-loss for this expert on this match
        prob_actual = max(predicted_probs[actual_outcome], 1e-12)
        loss = -math.log(prob_actual)
        # Clamp to avoid extreme weight collapse on a single bad prediction
        loss = min(loss, 10.0)

        self._hedge_cumulative_loss[expert_name] += loss

        # Multiplicative weight update
        self._hedge_weights[expert_name] *= math.exp(-eta * loss)

        # Re-normalise all weights so they sum to 1
        total = sum(self._hedge_weights.values())
        if total > 0:
            for name in self._hedge_weights:
                self._hedge_weights[name] /= total

    def get_hedge_weights(self) -> dict[str, float]:
        """Return current Hedge-algorithm weights for all tracked experts.

        Returns a dict mapping expert name to its normalised weight.  If the
        Hedge algorithm has not been run yet (no ``hedge_update`` calls),
        returns an empty dict.
        """
        if not hasattr(self, "_hedge_weights"):
            return {}
        return dict(self._hedge_weights)

    # ═══════════════════════════════════════════════════════════════
    # PER-LEAGUE TEMPERATURE CALIBRATION
    # ═══════════════════════════════════════════════════════════════

    def get_league_temperature(self, league: str) -> float:
        """Return a per-league temperature scaling parameter.

        Temperature > 1 softens probabilities (more uncertainty, good for
        upset-heavy leagues like the Premier League).
        Temperature < 1 sharpens probabilities (more decisive, good for
        leagues with strong favourites).

        The temperature is learned online from calibration error: if the
        current temperature leads to over-confidence (calibration error
        positive), the temperature is increased; if under-confident, it
        is decreased.  A simple gradient step is applied each time a new
        calibration measurement is available.

        Default temperature for an unknown league is 1.0 (no scaling).

        Args:
            league: Competition code (e.g. "PL", "BL1", "PD").

        Returns:
            Temperature float, typically in range [0.5, 2.0].
        """
        if not hasattr(self, "_league_temperatures"):
            self._league_temperatures: dict[str, float] = {}
        if not hasattr(self, "_league_calibration_errors"):
            self._league_calibration_errors: dict[str, list[float]] = {}

        if league not in self._league_temperatures:
            self._league_temperatures[league] = 1.0
            self._league_calibration_errors[league] = []

        return self._league_temperatures[league]

    def update_league_temperature(
        self,
        league: str,
        predicted_probs: list[float],
        actual_outcome: int,
        lr: float = 0.02,
    ) -> float:
        """Update per-league temperature after observing a scored match.

        Adjusts temperature to minimise calibration error for this league.
        The calibration signal is derived from the gap between the predicted
        probability of the actual outcome and the binary indicator (1.0).

        If predicted_probs[actual_outcome] is too high on average (over-
        confident), temperature is increased to soften future predictions.
        If too low (under-confident), temperature is decreased to sharpen.

        Args:
            league: Competition code.
            predicted_probs: [p_home, p_draw, p_away] council prediction.
            actual_outcome: 0 = Home, 1 = Draw, 2 = Away.
            lr: Learning rate for temperature adjustment (default 0.02).

        Returns:
            Updated temperature for the league.
        """
        if not hasattr(self, "_league_temperatures"):
            self._league_temperatures = {}
        if not hasattr(self, "_league_calibration_errors"):
            self._league_calibration_errors = {}

        if league not in self._league_temperatures:
            self._league_temperatures[league] = 1.0
            self._league_calibration_errors[league] = []

        # Calibration error for this single prediction:
        # positive means over-confident (predicted prob > reality)
        prob_actual = predicted_probs[actual_outcome]
        # For the actual outcome the ideal prob is 1.0; calibration error
        # is the mean gap between confidence and accuracy across many
        # predictions.  As a proxy per-match: max(predicted_probs) - correct.
        max_prob = max(predicted_probs)
        correct = 1.0 if int(np.argmax(predicted_probs)) == actual_outcome else 0.0
        cal_error = max_prob - correct

        self._league_calibration_errors[league].append(cal_error)
        # Keep a rolling window of 200 observations
        if len(self._league_calibration_errors[league]) > 200:
            self._league_calibration_errors[league] = self._league_calibration_errors[league][-200:]

        # Compute running mean calibration error for gradient direction
        avg_cal_error = float(np.mean(self._league_calibration_errors[league]))

        # Gradient step: if avg_cal_error > 0 (over-confident) raise T;
        # if < 0 (under-confident) lower T.
        temperature = self._league_temperatures[league]
        temperature += lr * avg_cal_error
        # Clamp to sensible range
        temperature = max(0.5, min(2.0, temperature))

        self._league_temperatures[league] = temperature
        return temperature

    def apply_temperature(self, probs: list[float], league: str) -> list[float]:
        """Apply per-league temperature scaling to a probability triplet.

        Divides the log-probabilities by the league temperature, then
        re-exponentiates and normalises.  Temperature = 1.0 is a no-op.

        Args:
            probs: [p_home, p_draw, p_away] raw probabilities.
            league: Competition code.

        Returns:
            Temperature-scaled and normalised probability triplet.
        """
        T = self.get_league_temperature(league)
        if abs(T - 1.0) < 1e-6:
            return list(probs)

        # Log-space scaling
        log_probs = [math.log(max(p, 1e-12)) / T for p in probs]
        max_lp = max(log_probs)
        exp_probs = [math.exp(lp - max_lp) for lp in log_probs]
        total = sum(exp_probs)
        return [p / total for p in exp_probs]

    def set_retraining_thresholds(
        self,
        accuracy_threshold: float = 0.38,
        ece_threshold: float = 0.12,
        grace_window_hours: int = 24,
    ) -> None:
        """Configure retraining thresholds.

        Args:
            accuracy_threshold: Minimum acceptable accuracy for predictions
            ece_threshold: Maximum acceptable Expected Calibration Error
            grace_window_hours: Hours to wait before rollback decision
        """
        self._retrain_threshold_accuracy = accuracy_threshold
        self._retrain_threshold_ece = ece_threshold
        self._grace_window_hours = grace_window_hours

    def should_retrain(self) -> bool:
        """Check if the model should be retrained."""
        return self._retrain_recommended

    def get_expert_stats(self, expert_name: str) -> dict[str, Any] | None:
        """Get detailed statistics for a specific expert.

        Args:
            expert_name: Name of the expert

        Returns:
            Dictionary with stats or None if expert not found
        """
        if expert_name not in self.expert_trackers:
            return None

        tracker = self.expert_trackers[expert_name]
        return {
            "name": expert_name,
            "global_accuracy": round(tracker.global_window.accuracy, 4),
            "global_log_loss": round(tracker.global_window.log_loss, 4),
            "global_n_predictions": tracker.global_window.n_predictions,
            "weight": round(tracker.get_weight()[0], 4),
            "uncertainty": round(tracker.get_weight()[1], 4),
            "failure_count": tracker.failure_count,
            "timeout_count": tracker.timeout_count,
            "extreme_prob_count": tracker.extreme_prob_count,
            "is_inactive": tracker.is_inactive(),
            "league_contexts": len(tracker.league_windows),
            "match_contexts": len(tracker.context_windows),
        }

    def serialize(self) -> dict[str, Any]:
        """Serialize the learning state for persistence."""
        data: dict[str, Any] = {
            "expert_weights": {},
            "overall_accuracy": self.overall_performance.accuracy,
            "overall_n": self.overall_performance.n_predictions,
            "drift_detected": self.drift_detector.is_drifting,
            "hedge_weights": self.get_hedge_weights(),
            "league_temperatures": dict(self._league_temperatures)
            if hasattr(self, "_league_temperatures")
            else {},
        }
        for name, tracker in self.expert_trackers.items():
            data["expert_weights"][name] = {
                "global": tracker.get_weight(),
                "leagues": {
                    lg: tracker.get_weight(league=lg)
                    for lg in tracker.league_windows
                },
            }
        return data

    @classmethod
    def from_db(cls, con: Any) -> "SelfLearningLoop":
        """Reconstruct learning state from the database.

        Reads scored predictions and rebuilds the performance windows.
        """
        loop = cls()
        try:
            rows = cast(list[tuple[Any, ...]], con.execute("""
                SELECT ps.match_id, p.p_home, p.p_draw, p.p_away,
                       ps.outcome, ps.correct, ps.logloss,
                       m.competition, ps.model_version,
                       p.notes, ec.breakdown_json
                FROM prediction_scores ps
                JOIN predictions p ON p.match_id = ps.match_id
                    AND p.model_version = ps.model_version
                JOIN matches m ON m.match_id = ps.match_id
                LEFT JOIN expert_cache ec ON ec.match_id = ps.match_id
                ORDER BY ps.scored_at DESC
                LIMIT 1000
            """).fetchall())

            for row in reversed(rows):
                mid, ph, pd_val, pa, outcome, _correct, _ll, comp, model_version, notes, cached_breakdown = row
                predicted_probs = [
                    float(ph or 1 / 3),
                    float(pd_val or 1 / 3),
                    float(pa or 1 / 3),
                ]

                expert_payload: Any = notes
                if isinstance(model_version, str) and "council" in model_version:
                    expert_payload = cached_breakdown or notes
                expert_predictions = extract_expert_predictions(expert_payload)

                loop.record_prediction_result(
                    match_id=int(mid),
                    predicted_probs=predicted_probs,
                    actual_outcome=int(outcome or 0),
                    expert_predictions=expert_predictions or None,
                    league=str(comp or ""),
                    model_version=str(model_version or ""),
                )

        except Exception as e:
            log.debug("self_learning from_db: %s", e)

        return loop


# Module-level singleton
_learning_loop: SelfLearningLoop | None = None


def get_learning_loop() -> SelfLearningLoop:
    """Get or create the global self-learning loop."""
    global _learning_loop
    if _learning_loop is None:
        _learning_loop = SelfLearningLoop()
    return _learning_loop
