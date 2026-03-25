"""Tests for ExpertGate — expert signal quality gating."""
import numpy as np
import pytest

from footy.models.expert_gate import ExpertGate


class TestExpertGate:
    """Tests for the ExpertGate class."""

    def test_expert_gate_prunes_flat_experts(self):
        """Simulate 100 flat (1/3, 1/3, 1/3) predictions — expert should be gated."""
        gate = ExpertGate(min_observations=50, entropy_threshold=0.98)
        flat_probs = [1 / 3, 1 / 3, 1 / 3]

        for _ in range(100):
            gate.record("flat_expert", flat_probs, confidence=0.0)

        assert gate.is_active("flat_expert") is False

    def test_expert_gate_keeps_good_experts(self):
        """Simulate 100 non-flat predictions — expert should stay active."""
        gate = ExpertGate(min_observations=50, entropy_threshold=0.98)
        good_probs = [0.7, 0.2, 0.1]

        for _ in range(100):
            gate.record("good_expert", good_probs, confidence=0.6)

        assert gate.is_active("good_expert") is True

    def test_expert_gate_benefit_of_doubt(self):
        """Fewer than min_observations should always return active."""
        gate = ExpertGate(min_observations=50, entropy_threshold=0.98)
        flat_probs = [1 / 3, 1 / 3, 1 / 3]

        # Only record 30 observations (below the 50 threshold)
        for _ in range(30):
            gate.record("new_expert", flat_probs, confidence=0.0)

        assert gate.is_active("new_expert") is True

        # An expert with zero history should also be active
        assert gate.is_active("unknown_expert") is True

    def test_expert_gate_batch_recording(self):
        """Test record_batch method processes arrays correctly."""
        gate = ExpertGate(min_observations=50, entropy_threshold=0.98)

        n = 100
        flat_probs = np.full((n, 3), 1 / 3)
        zero_conf = np.zeros(n)

        gate.record_batch("batch_flat", flat_probs, zero_conf)

        assert len(gate._history["batch_flat"]) == n
        assert gate.is_active("batch_flat") is False

        # Now test with good predictions
        good_probs = np.array([[0.7, 0.2, 0.1]] * n)
        good_conf = np.full(n, 0.6)

        gate.record_batch("batch_good", good_probs, good_conf)

        assert len(gate._history["batch_good"]) == n
        assert gate.is_active("batch_good") is True

    def test_expert_gate_report(self):
        """Test get_report returns correct structure and values."""
        gate = ExpertGate(min_observations=50, entropy_threshold=0.98)

        # Record data for two experts
        for _ in range(100):
            gate.record("noisy", [1 / 3, 1 / 3, 1 / 3], confidence=0.0)
            gate.record("sharp", [0.8, 0.1, 0.1], confidence=0.7)

        report = gate.get_report()

        # Both experts should appear in the report
        assert "noisy" in report
        assert "sharp" in report

        # Check report structure for each expert
        for name in ("noisy", "sharp"):
            entry = report[name]
            assert "n_observations" in entry
            assert "mean_entropy_ratio" in entry
            assert "mean_confidence" in entry
            assert "active" in entry
            assert entry["n_observations"] == 100

        # Noisy expert should be gated
        assert report["noisy"]["active"] is False
        assert report["noisy"]["mean_entropy_ratio"] > 0.98
        assert report["noisy"]["mean_confidence"] == 0.0

        # Sharp expert should be active
        assert report["sharp"]["active"] is True
        assert report["sharp"]["mean_entropy_ratio"] < 0.98
        assert report["sharp"]["mean_confidence"] == 0.7

    def test_get_active_experts_filters_correctly(self):
        """Test get_active_experts filters a list of expert-like objects."""
        gate = ExpertGate(min_observations=50, entropy_threshold=0.98)

        # Create mock expert objects with a name attribute
        class MockExpert:
            def __init__(self, name):
                self.name = name

        experts = [MockExpert("good"), MockExpert("bad"), MockExpert("new")]

        # Record enough data to gate "bad", keep "good" active
        for _ in range(100):
            gate.record("good", [0.7, 0.2, 0.1], confidence=0.5)
            gate.record("bad", [1 / 3, 1 / 3, 1 / 3], confidence=0.0)
        # "new" has no history — benefit of the doubt

        active = gate.get_active_experts(experts)
        active_names = [e.name for e in active]

        assert "good" in active_names
        assert "new" in active_names
        assert "bad" not in active_names

    def test_entropy_calculation(self):
        """Verify the entropy static method produces correct values."""
        max_entropy = np.log(3)

        # Uniform distribution should have maximum entropy
        uniform_entropy = ExpertGate._entropy([1 / 3, 1 / 3, 1 / 3])
        assert abs(uniform_entropy - max_entropy) < 1e-6

        # Certain distribution should have near-zero entropy
        certain_entropy = ExpertGate._entropy([1.0, 0.0, 0.0])
        assert certain_entropy < 1e-6

        # Intermediate distribution
        mid_entropy = ExpertGate._entropy([0.5, 0.3, 0.2])
        assert 0 < mid_entropy < max_entropy
