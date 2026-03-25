"""Conformal prediction sets for football match outcomes.

Provides distribution-free uncertainty quantification with finite-sample coverage
guarantees. Uses Adaptive Prediction Sets (APS) method from Romano et al. (2020).

Key insight for football: when the prediction set contains all 3 outcomes
(Home, Draw, Away), the model is genuinely uncertain — this is the strongest
upset risk indicator.
"""
import numpy as np


class ConformalPredictor:
    """Conformal prediction sets with guaranteed coverage.

    Usage:
        cp = ConformalPredictor(alpha=0.10)  # 90% target coverage
        cp.calibrate(cal_probs, cal_labels)  # calibrate on held-out data
        sets = cp.predict_sets(new_probs)    # get prediction sets
        sizes = cp.set_sizes(new_probs)      # 1=confident, 3=uncertain/upset risk
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: Miscoverage rate. 0.10 = 90% coverage target.
        """
        self.alpha = alpha
        self.qhat = None
        self._calibrated = False

    def calibrate(self, probs: np.ndarray, y: np.ndarray):
        """Calibrate on held-out data using APS nonconformity scores.

        Args:
            probs: (n, 3) probability matrix from the model.
            y: (n,) true labels (0=Home, 1=Draw, 2=Away).
        """
        n = len(y)
        if n < 10:
            raise ValueError(f"Need at least 10 calibration samples, got {n}")

        scores = np.zeros(n)
        rng = np.random.default_rng(42)

        for i in range(n):
            sorted_idx = np.argsort(-probs[i])
            cumsum = 0.0
            for cls in sorted_idx:
                cumsum += probs[i, cls]
                if cls == y[i]:
                    # Randomized APS score for exact coverage
                    u = rng.uniform()
                    scores[i] = cumsum - probs[i, cls] * u
                    break

        # Quantile with finite-sample correction (Vovk 2005)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat = float(np.quantile(scores, min(level, 1.0)))
        self._calibrated = True

    def predict_sets(self, probs: np.ndarray) -> list[list[int]]:
        """Produce prediction sets for each row.

        Args:
            probs: (n, 3) probability matrix.

        Returns:
            List of prediction sets. Each set is a list of class indices.
            Set size 1 = confident prediction.
            Set size 3 = maximum uncertainty (upset risk indicator).
        """
        if not self._calibrated:
            raise ValueError("Must calibrate before predicting")

        sets = []
        for i in range(len(probs)):
            sorted_idx = np.argsort(-probs[i])
            cumsum = 0.0
            pred_set = []
            for cls in sorted_idx:
                pred_set.append(int(cls))
                cumsum += probs[i, cls]
                if cumsum >= self.qhat:
                    break
            sets.append(pred_set)
        return sets

    def set_sizes(self, probs: np.ndarray) -> np.ndarray:
        """Return prediction set sizes. Larger = more uncertain = upset risk.

        Args:
            probs: (n, 3) probability matrix.

        Returns:
            (n,) array of set sizes (1, 2, or 3).
        """
        sets = self.predict_sets(probs)
        return np.array([len(s) for s in sets])

    def upset_risk_scores(self, probs: np.ndarray) -> np.ndarray:
        """Compute upset risk scores based on prediction set analysis.

        Combines set size with probability distribution shape.
        Higher score = more likely upset.

        Args:
            probs: (n, 3) probability matrix.

        Returns:
            (n,) array of upset risk scores in [0, 1].
        """
        sizes = self.set_sizes(probs)
        max_probs = probs.max(axis=1)
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        max_entropy = np.log(3)

        # Combine: large set + low max prob + high entropy = upset
        risk = (
            0.4 * (sizes - 1) / 2.0 +          # set size contribution
            0.3 * (1.0 - max_probs) +            # uncertainty from max prob
            0.3 * (entropy / max_entropy)         # entropy contribution
        )
        return np.clip(risk, 0.0, 1.0)

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated
