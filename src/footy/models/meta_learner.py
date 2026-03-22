"""Meta-learner for ensemble stacking.

Combines predictions from multiple expert models using a gradient boosting
meta-learner trained on walk-forward cross-validation splits.

Architecture:
- Input: Expert predictions matrix (N x (num_experts * 3))
  Each expert contributes 3 probabilities: [p_home, p_draw, p_away]
- Meta-learner: HistGradientBoosting with isotonic calibration
- Output: Calibrated probabilities for home/draw/away
"""
from __future__ import annotations

import numpy as np
import logging
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

log = logging.getLogger(__name__)


class EnsembleMetaLearner:
    """Stacking meta-learner that combines expert predictions."""
    
    MODEL_PATH = Path("data/models/meta_learner.joblib")
    
    def __init__(self):
        self._model = None
        self._expert_names = []
        self._is_fitted = False
    
    def fit(self, expert_predictions: np.ndarray, outcomes: np.ndarray, 
            expert_names: list[str]):
        """Train meta-learner on expert prediction matrix.
        
        Args:
            expert_predictions: shape (n_matches, n_experts * 3) - each expert contributes
                               [p_home, p_draw, p_away]
            outcomes: shape (n_matches,) - 0=home, 1=draw, 2=away
            expert_names: list of expert names for feature tracking
        """
        self._expert_names = expert_names
        
        # Validate inputs
        if len(expert_predictions) < 20:
            log.warning("Meta-learner: insufficient training data (%d < 20)", 
                       len(expert_predictions))
            return
        
        if expert_predictions.shape[1] != len(expert_names) * 3:
            log.warning("Meta-learner: feature count mismatch. "
                       "Expected %d, got %d", 
                       len(expert_names) * 3, expert_predictions.shape[1])
            return
        
        try:
            # Train HistGradientBoosting with calibration
            base = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42,
            )
            self._model = CalibratedClassifierCV(base, cv=5, method="isotonic")
            self._model.fit(expert_predictions, outcomes)
            self._is_fitted = True
            
            # Save model
            self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "model": self._model,
                "expert_names": self._expert_names,
            }, self.MODEL_PATH)
            log.info("Meta-learner trained on %d matches with %d experts", 
                     len(outcomes), len(expert_names))
        except Exception as e:
            log.error("Meta-learner training failed: %s", e)
            self._is_fitted = False
    
    def predict(self, expert_predictions: np.ndarray) -> np.ndarray | None:
        """Predict match outcome probabilities.
        
        Args:
            expert_predictions: shape (1, n_experts * 3) or (n, n_experts * 3)
            
        Returns:
            shape (n, 3) - [p_home, p_draw, p_away] calibrated probabilities,
            or None if not fitted
        """
        if not self._is_fitted:
            self._load()
        if not self._is_fitted:
            return None
        
        try:
            probs = self._model.predict_proba(expert_predictions)
            return probs
        except Exception as e:
            log.warning("Meta-learner prediction failed: %s", e)
            return None
    
    def _load(self):
        """Load saved model from disk."""
        if self.MODEL_PATH.exists():
            try:
                data = joblib.load(self.MODEL_PATH)
                self._model = data["model"]
                self._expert_names = data["expert_names"]
                self._is_fitted = True
            except Exception as e:
                log.warning("Failed to load meta-learner: %s", e)
    
    def feature_importance(self) -> dict[str, float]:
        """Get feature importance for each expert's contribution.
        
        Note: HistGradientBoosting doesn't provide built-in feature importance.
        This placeholder allows for future extension with permutation importance
        or other analysis methods.
        
        Returns:
            Dictionary mapping feature names to importance scores (currently empty)
        """
        if not self._is_fitted:
            return {}
        
        # Return empty dict - HistGradientBoosting doesn't support feature_importances_
        # Users can compute permutation importance if needed:
        # from sklearn.inspection import permutation_importance
        # perm = permutation_importance(self._model, X_test, y_test)
        return {}
    
    def is_fitted(self) -> bool:
        """Check if meta-learner has been trained."""
        return self._is_fitted
