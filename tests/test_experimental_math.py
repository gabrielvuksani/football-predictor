from __future__ import annotations

import numpy as np

from footy.models.experimental_math import (
    KalmanStrengthState,
    expected_points_from_score_matrix,
    kalman_attack_defence_update,
    recency_attention_weights,
    variance_adjusted_blend,
    zero_inflated_poisson_pmf,
    build_all_score_matrices,
    bayesian_model_average,
    score_matrix_entropy,
    aggregate_model_predictions,
    negative_binomial_pmf,
    weibull_count_pmf,
    double_poisson_pmf,
    ordered_probit_1x2,
    skellam_match_probs,
    bradley_terry_davidson,
    pythagorean_win_pct,
    regression_to_mean_signal,
    match_state_transition_probs,
    markov_chain_simulate,
    bivariate_normal_goal_diff,
    goal_timing_intensity,
    SelfCalibratingEnsemble,
    ensemble_disagreement,
    expected_calibration_error,
    unified_prediction,
)


def test_zero_inflated_poisson_zero_mass():
    assert zero_inflated_poisson_pmf(0, 1.2, 0.1) > 0.1


def test_kalman_update_changes_state():
    prev = KalmanStrengthState()
    nxt = kalman_attack_defence_update(prev, observed_for=2, observed_against=0, expected_for=1.1, expected_against=1.0)
    assert nxt.attack != prev.attack
    assert nxt.defence != prev.defence


def test_expected_points_from_score_matrix():
    mat = np.array([[0.2, 0.1], [0.3, 0.4]], dtype=float)
    hp, ap = expected_points_from_score_matrix(mat)
    assert hp > ap


def test_recency_attention_weights_sum_to_one():
    w = recency_attention_weights(6)
    assert np.isclose(w.sum(), 1.0)
    assert w[-1] > w[0]


def test_variance_adjusted_blend_normalized():
    p1 = np.array([[0.5, 0.3, 0.2]])
    p2 = np.array([[0.2, 0.3, 0.5]])
    v1 = np.array([[0.1, 0.1, 0.1]])
    v2 = np.array([[0.2, 0.2, 0.2]])
    out = variance_adjusted_blend([p1, p2], [v1, v2])
    assert np.isclose(out.sum(), 1.0)
    assert out[0, 0] > out[0, 2]


def test_build_all_score_matrices_12_models():
    matrices = build_all_score_matrices(1.5, 1.1)
    assert len(matrices) == 12
    for name, mat in matrices.items():
        assert mat.shape == (9, 9), f"{name} shape wrong"
        assert np.isclose(mat.sum(), 1.0, atol=0.01), f"{name} doesn't sum to 1"


def test_bayesian_model_average_uniform():
    m1 = np.ones((9, 9)) / 81.0
    m2 = np.ones((9, 9)) / 81.0
    bma = bayesian_model_average([m1, m2])
    assert np.isclose(bma.sum(), 1.0)


def test_score_matrix_entropy_positive():
    mat = np.ones((9, 9)) / 81.0
    e = score_matrix_entropy(mat)
    assert e > 0


def test_aggregate_model_predictions_keys():
    result = aggregate_model_predictions(1.3, 0.9)
    assert "bma_p_home" in result
    assert "bma_entropy" in result
    assert "model_agreement" in result
    assert result["bma_p_home"] + result["bma_p_draw"] + result["bma_p_away"] < 1.05


def test_negative_binomial_pmf_sums_to_one():
    total = sum(negative_binomial_pmf(k, 1.5, 3.0) for k in range(15))
    assert abs(total - 1.0) < 0.05


def test_weibull_count_pmf_positive():
    assert weibull_count_pmf(0, 1.2, 1.0) > 0
    assert weibull_count_pmf(2, 1.2, 1.08) > 0


def test_double_poisson_pmf_basic():
    p0 = double_poisson_pmf(0, 1.0, 1.0)
    assert 0 < p0 < 1


def test_ordered_probit_valid_probs():
    probs = ordered_probit_1x2(0.5)
    assert abs(probs["p_home"] + probs["p_draw"] + probs["p_away"] - 1.0) < 1e-6
    assert probs["p_home"] > probs["p_away"]  # positive mu favors home


def test_skellam_valid_probs():
    probs = skellam_match_probs(1.8, 1.0)
    assert abs(probs["p_home"] + probs["p_draw"] + probs["p_away"] - 1.0) < 1e-6
    assert probs["p_home"] > probs["p_away"]
    assert probs["expected_gd"] > 0


def test_bradley_terry_davidson_1x2():
    probs = bradley_terry_davidson(1.5, 1.0)
    assert abs(probs["p_home"] + probs["p_draw"] + probs["p_away"] - 1.0) < 1e-6
    assert probs["p_home"] > probs["p_away"]


def test_pythagorean_win_pct_basic():
    # Team scoring more should win more
    assert pythagorean_win_pct(50, 30) > 0.5
    assert pythagorean_win_pct(30, 50) < 0.5
    assert abs(pythagorean_win_pct(40, 40) - 0.5) < 0.01


def test_regression_to_mean_signal_overperformer():
    result = regression_to_mean_signal(45.0, 38.0, 20)
    assert result["overperformance"] > 0
    assert result["direction"] == "negative"  # expects regression down


def test_regression_to_mean_signal_underperformer():
    result = regression_to_mean_signal(30.0, 38.0, 20)
    assert result["overperformance"] < 0
    assert result["direction"] == "positive"


def test_match_state_transition_probs():
    probs = match_state_transition_probs(1.5, 1.0)
    assert 0 < probs["p_home_scores_next"] < 1
    assert 0 < probs["p_away_scores_next"] < 1


def test_markov_chain_simulate_valid():
    r = markov_chain_simulate(1.4, 1.2, 1.0, 1.0, n_sims=1000)
    total = r["mc_p_home"] + r["mc_p_draw"] + r["mc_p_away"]
    assert abs(total - 1.0) < 0.01
    assert r["mc_eg_h"] > 0
    assert r["mc_eg_a"] > 0


def test_bivariate_normal_goal_diff():
    r = bivariate_normal_goal_diff(1.6, 1.1)
    assert abs(r["p_home"] + r["p_draw"] + r["p_away"] - 1.0) < 1e-6
    assert r["gd_mean"] > 0


def test_goal_timing_intensity_sums():
    intensities = goal_timing_intensity(2.5)
    assert len(intensities) == 90
    assert abs(sum(intensities) - 2.5) < 0.5  # should approximately match total lambda


# ═══════ Self-Calibrating Ensemble Tests ═══════

def test_self_calibrating_ensemble_init():
    ens = SelfCalibratingEnsemble()
    assert len(ens.model_names) == 12
    weights = ens.get_weights()
    assert len(weights) == 12
    # Without data, all weights should be equal
    values = list(weights.values())
    assert all(abs(v - values[0]) < 0.01 for v in values)


def test_self_calibrating_ensemble_update():
    ens = SelfCalibratingEnsemble(model_names=["poisson", "dixon_coles"])
    # Simulate: poisson predicts well, DC poorly
    ens.update(
        {"poisson": {"p_home": 0.7, "p_draw": 0.2, "p_away": 0.1},
         "dixon_coles": {"p_home": 0.3, "p_draw": 0.3, "p_away": 0.4}},
        "home",
    )
    weights = ens.get_weights()
    assert weights["poisson"] > weights["dixon_coles"]


def test_self_calibrating_ensemble_serialization():
    ens = SelfCalibratingEnsemble(model_names=["poisson", "dixon_coles"])
    ens.update(
        {"poisson": {"p_home": 0.6, "p_draw": 0.2, "p_away": 0.2},
         "dixon_coles": {"p_home": 0.4, "p_draw": 0.3, "p_away": 0.3}},
        "home",
    )
    data = ens.to_dict()
    ens2 = SelfCalibratingEnsemble.from_dict(data)
    assert ens2.records["poisson"].n_predictions == 1
    assert ens2.records["poisson"].total_log_loss > 0


def test_ensemble_disagreement_identical():
    preds = [{"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}] * 3
    result = ensemble_disagreement(preds)
    assert result["diversity_score"] == 0.0


def test_ensemble_disagreement_varied():
    preds = [
        {"p_home": 0.8, "p_draw": 0.1, "p_away": 0.1},
        {"p_home": 0.1, "p_draw": 0.1, "p_away": 0.8},
    ]
    result = ensemble_disagreement(preds)
    assert result["diversity_score"] > 0.1


def test_expected_calibration_error_perfect():
    preds = np.array([0.8, 0.2, 0.7, 0.3])
    actuals = np.array([1.0, 0.0, 1.0, 0.0])
    ece = expected_calibration_error(preds, actuals)
    assert ece < 0.3  # Well-calibrated predictions


def test_unified_prediction_produces_valid_probs():
    result = unified_prediction(1.5, 1.0)
    probs_sum = result["p_home"] + result["p_draw"] + result["p_away"]
    assert abs(probs_sum - 1.0) < 0.01
    assert 0 < result["confidence"] < 1
    assert result["entropy"] > 0
