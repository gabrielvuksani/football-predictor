"""Comprehensive tests for conformal prediction sets."""
import numpy as np
import pytest

from footy.models.conformal import ConformalPredictor


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def calibrated_predictor(rng):
    """Return a ConformalPredictor calibrated on Dirichlet(2,1,1) samples."""
    cp = ConformalPredictor(alpha=0.10)
    cal_probs = rng.dirichlet([2, 1, 1], size=200)
    cal_labels = np.array([
        np.argmax(rng.multinomial(1, p)) for p in cal_probs
    ])
    cp.calibrate(cal_probs, cal_labels)
    return cp


def test_conformal_coverage(rng, calibrated_predictor):
    """Calibrate with 200 Dirichlet(2,1,1) samples, test on 500.

    Empirical coverage should be >= 0.85 (allowing slack for small sample).
    """
    cp = calibrated_predictor

    test_probs = rng.dirichlet([2, 1, 1], size=500)
    test_labels = np.array([
        np.argmax(rng.multinomial(1, p)) for p in test_probs
    ])

    pred_sets = cp.predict_sets(test_probs)

    covered = sum(
        test_labels[i] in pred_sets[i] for i in range(len(test_labels))
    )
    coverage = covered / len(test_labels)
    assert coverage >= 0.85, f"Coverage {coverage:.3f} < 0.85"


def test_conformal_set_sizes(rng, calibrated_predictor):
    """All prediction set sizes must be 1, 2, or 3."""
    cp = calibrated_predictor
    test_probs = rng.dirichlet([2, 1, 1], size=100)
    sizes = cp.set_sizes(test_probs)

    assert sizes.shape == (100,)
    for s in sizes:
        assert s in (1, 2, 3), f"Invalid set size: {s}"


def test_conformal_requires_calibration():
    """Predicting before calibration must raise ValueError."""
    cp = ConformalPredictor(alpha=0.10)
    probs = np.array([[0.6, 0.2, 0.2]])

    with pytest.raises(ValueError, match="Must calibrate"):
        cp.predict_sets(probs)

    assert not cp.is_calibrated


def test_conformal_upset_risk_scores(rng, calibrated_predictor):
    """Upset risk scores must have correct shape and lie in [0, 1]."""
    cp = calibrated_predictor
    test_probs = rng.dirichlet([2, 1, 1], size=50)
    risk = cp.upset_risk_scores(test_probs)

    assert risk.shape == (50,)
    assert np.all(risk >= 0.0)
    assert np.all(risk <= 1.0)


def test_conformal_confident_predictions(calibrated_predictor):
    """High-confidence predictions (one prob near 1) should yield set size 1."""
    cp = calibrated_predictor
    confident_probs = np.array([
        [0.95, 0.03, 0.02],
        [0.02, 0.96, 0.02],
        [0.01, 0.02, 0.97],
    ])
    sizes = cp.set_sizes(confident_probs)
    for i, s in enumerate(sizes):
        assert s == 1, (
            f"Row {i} with probs {confident_probs[i]} got set size {s}, expected 1"
        )


def test_conformal_uncertain_predictions(calibrated_predictor):
    """Near-uniform predictions should yield set size 3 (maximum uncertainty)."""
    cp = calibrated_predictor
    uniform_probs = np.array([
        [0.34, 0.33, 0.33],
        [0.33, 0.34, 0.33],
        [0.33, 0.33, 0.34],
    ])
    sizes = cp.set_sizes(uniform_probs)
    for i, s in enumerate(sizes):
        assert s == 3, (
            f"Row {i} with probs {uniform_probs[i]} got set size {s}, expected 3"
        )
