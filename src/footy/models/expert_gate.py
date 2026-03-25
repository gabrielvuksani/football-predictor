"""Expert gating — automatically disable experts producing pure noise.

Tracks per-expert signal quality (entropy of probability outputs and confidence)
and gates out experts that consistently return near-uniform (1/3, 1/3, 1/3)
distributions with zero confidence. This prevents noise injection into the
feature matrix from data-starved experts.
"""
import numpy as np
from collections import defaultdict


class ExpertGate:
    """Track expert signal quality and gate out noise producers."""

    def __init__(self, min_observations: int = 50, entropy_threshold: float = 0.98):
        """
        Args:
            min_observations: Minimum predictions before gating decision.
            entropy_threshold: Fraction of max entropy above which expert is noise.
        """
        self.min_observations = min_observations
        self.entropy_threshold = entropy_threshold
        self._history = defaultdict(list)

    def record(self, expert_name: str, probs: list[float], confidence: float):
        """Record one prediction from an expert for quality tracking."""
        self._history[expert_name].append({
            'entropy': self._entropy(probs),
            'confidence': confidence,
        })

    def record_batch(self, expert_name: str, probs: np.ndarray, confidence: np.ndarray):
        """Record a batch of predictions for quality tracking."""
        for i in range(len(probs)):
            self._history[expert_name].append({
                'entropy': self._entropy(probs[i]),
                'confidence': float(confidence[i]),
            })

    def is_active(self, expert_name: str) -> bool:
        """Check if an expert should be included in the feature matrix."""
        history = self._history.get(expert_name, [])
        if len(history) < self.min_observations:
            return True  # benefit of the doubt

        recent = history[-self.min_observations:]
        max_entropy = np.log(3)  # entropy of uniform [1/3, 1/3, 1/3]
        mean_entropy = np.mean([h['entropy'] for h in recent])
        mean_conf = np.mean([h['confidence'] for h in recent])

        # Gate out if: near-max entropy AND near-zero confidence
        if mean_entropy / max_entropy > self.entropy_threshold and mean_conf < 0.05:
            return False
        return True

    def get_active_experts(self, all_experts: list) -> list:
        """Filter expert list to only active (signal-producing) experts."""
        return [e for e in all_experts if self.is_active(getattr(e, 'name', str(e)))]

    def get_report(self) -> dict:
        """Return quality report for all tracked experts."""
        report = {}
        max_entropy = np.log(3)
        for name, history in self._history.items():
            if not history:
                continue
            recent = history[-min(len(history), self.min_observations):]
            mean_ent = np.mean([h['entropy'] for h in recent])
            mean_conf = np.mean([h['confidence'] for h in recent])
            report[name] = {
                'n_observations': len(history),
                'mean_entropy_ratio': round(mean_ent / max_entropy, 4),
                'mean_confidence': round(mean_conf, 4),
                'active': self.is_active(name),
            }
        return report

    @staticmethod
    def _entropy(probs):
        """Shannon entropy of a probability distribution."""
        p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
        return float(-(p * np.log(p)).sum())
