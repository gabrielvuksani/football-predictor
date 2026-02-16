"""Unit tests — Expert Council model (v8)."""
import numpy as np
import pytest


class TestCouncilHelpers:
    """Pure helper functions from council module."""

    def test_f_none(self):
        from footy.models.council import _f
        assert _f(None) == 0.0

    def test_f_nan(self):
        from footy.models.council import _f
        assert _f(float("nan")) == 0.0

    def test_f_inf(self):
        from footy.models.council import _f
        assert _f(float("inf")) == 0.0

    def test_f_normal(self):
        from footy.models.council import _f
        assert _f(3.14) == 3.14

    def test_norm3_sums_to_one(self):
        from footy.models.council import _norm3
        a, b, c = _norm3(2.0, 1.0, 1.0)
        assert abs(a + b + c - 1.0) < 1e-9

    def test_norm3_zeros_returns_uniform(self):
        from footy.models.council import _norm3
        a, b, c = _norm3(0, 0, 0)
        assert abs(a - 1 / 3) < 1e-9
        assert abs(b - 1 / 3) < 1e-9

    def test_entropy3_uniform(self):
        from footy.models.council import _entropy3
        h = _entropy3([1 / 3, 1 / 3, 1 / 3])
        assert abs(h - np.log(3)) < 0.01

    def test_entropy3_certain(self):
        from footy.models.council import _entropy3
        h = _entropy3([1.0, 0.0, 0.0])
        assert h < 0.01

    def test_label_home(self):
        from footy.models.council import _label
        assert _label(2, 1) == 0

    def test_label_draw(self):
        from footy.models.council import _label
        assert _label(1, 1) == 1

    def test_label_away(self):
        from footy.models.council import _label
        assert _label(0, 2) == 2

    def test_pts(self):
        from footy.models.council import _pts
        assert _pts(3, 0) == 3
        assert _pts(1, 1) == 1
        assert _pts(0, 2) == 0

    def test_implied_valid_odds(self):
        from footy.models.council import _implied
        ih, id_, ia, over = _implied(2.0, 3.5, 4.0)
        assert abs(ih + id_ + ia - 1.0) < 1e-9
        assert over > 0  # overround

    def test_implied_bad_odds(self):
        from footy.models.council import _implied
        ih, id_, ia, over = _implied(0, 0, 0)
        assert ih == 0.0


class TestExpertResult:
    def test_expert_result_fields(self):
        from footy.models.council import ExpertResult
        er = ExpertResult(
            probs=np.array([[0.4, 0.3, 0.3]]),
            confidence=np.array([0.8]),
            features={"feat_a": np.array([1.0]), "feat_b": np.array([2.0])},
        )
        assert er.probs.shape == (1, 3)
        assert isinstance(er.features, dict)
        assert abs(er.probs.sum() - 1.0) < 1e-6


class TestModelPath:
    def test_model_version_constant(self):
        from footy.models.council import MODEL_VERSION
        assert MODEL_VERSION == "v8_council"

    def test_model_path_extension(self):
        from footy.models.council import MODEL_PATH
        assert str(MODEL_PATH).endswith(".joblib")
