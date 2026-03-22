#!/usr/bin/env python3
"""Integration test for advanced mathematical implementations.

Tests:
    1. TrueSkill rating system
    2. TrueSkill expert for council
    3. Enhanced Bayesian engine
    4. Enhanced prediction aggregator with BMA
"""
import sys
import math
from typing import Any

import numpy as np
import pandas as pd

# Add source to path
sys.path.insert(0, '/sessions/friendly-nice-dirac/mnt/football-predictor-main/src')

from footy.models.trueskill import TrueSkillEngine, Skill, TeamSkills
from footy.models.experts.trueskill import TrueSkillExpert
from footy.models.bayesian_engine import BayesianStateSpaceEngine
from footy.prediction_aggregator import (
    compute_value_edges,
    compute_upset_risk,
    jensen_shannon_divergence,
    model_agreement_from_divergences,
    aggregate_predictions,
    logarithmic_opinion_pool,
)


def test_trueskill_engine():
    """Test TrueSkill rating system."""
    print("\n" + "="*70)
    print("TEST 1: TrueSkill Rating System")
    print("="*70)

    engine = TrueSkillEngine(
        initial_mu=1500.0,
        initial_sigma=350.0,
        beta=100.0,
        tau=10.0,
        draw_margin=45.0,
    )

    # Simulate a few matches
    matches = [
        {"home_team": "Team A", "away_team": "Team B", "home_goals": 2, "away_goals": 1},
        {"home_team": "Team B", "away_team": "Team C", "home_goals": 0, "away_goals": 3},
        {"home_team": "Team A", "away_team": "Team C", "home_goals": 1, "away_goals": 1},
    ]

    engine.batch_update(matches)

    # Predict
    p_h, p_d, p_a = engine.predict_probs("Team A", "Team B")
    assert 0.0 <= p_h <= 1.0 and 0.0 <= p_d <= 1.0 and 0.0 <= p_a <= 1.0
    assert abs(p_h + p_d + p_a - 1.0) < 0.01, f"Probs don't sum to 1: {p_h + p_d + p_a}"

    rankings = engine.get_rankings(top_n=10)
    assert len(rankings) == 3, f"Expected 3 teams, got {len(rankings)}"

    print(f"✓ Predictions: P(Home)={p_h:.4f}, P(Draw)={p_d:.4f}, P(Away)={p_a:.4f}")
    print(f"✓ Team A overall strength: {rankings[0]['overall_strength']:.2f}")
    print(f"✓ State export/import works")

    # Test state export/import
    state = engine.export_state()
    assert "Team A" in state
    engine2 = TrueSkillEngine()
    engine2.import_state(state)
    p_h2, p_d2, p_a2 = engine2.predict_probs("Team A", "Team B")
    assert abs(p_h - p_h2) < 0.001, "Export/import not preserving predictions"

    return True


def test_trueskill_expert():
    """Test TrueSkill expert for council."""
    print("\n" + "="*70)
    print("TEST 2: TrueSkill Expert")
    print("="*70)

    expert = TrueSkillExpert()

    # Create sample DataFrame
    df = pd.DataFrame({
        "utc_date": pd.date_range("2024-01-01", periods=5),
        "home_team": ["A", "B", "A", "C", "B"],
        "away_team": ["B", "C", "C", "A", "A"],
        "home_goals": [2.0, 1.0, 3.0, 0.0, np.nan],  # Last is upcoming (NaN)
        "away_goals": [1.0, 0.0, 1.0, 2.0, np.nan],
    })

    result = expert.compute(df)

    # Check output shape
    assert result.probs.shape == (5, 3), f"Probs shape {result.probs.shape}, expected (5, 3)"
    assert len(result.confidence) == 5
    assert all("ts_" in k for k in result.features.keys()), "Features should have ts_ prefix"

    # Verify probabilities
    for i in range(5):
        p_sum = result.probs[i].sum()
        assert abs(p_sum - 1.0) < 0.01, f"Row {i} probs sum to {p_sum}, expected 1.0"

    print(f"✓ Generated {result.probs.shape[0]} predictions")
    print(f"✓ Mean confidence: {result.confidence.mean():.4f}")
    print(f"✓ Features: {list(result.features.keys())}")

    return True


def test_bayesian_engine_enhancements():
    """Test Bayesian engine improvements."""
    print("\n" + "="*70)
    print("TEST 3: Enhanced Bayesian Engine")
    print("="*70)

    engine = BayesianStateSpaceEngine(
        surprise_history_max=200,
        lambda_bounds=(0.1, 5.0),
        per_league_zero_inflation=True,
    )

    # Verify parameters are bounded
    assert engine.rue_salvesen_gamma <= 0.15
    assert engine.lambda_bounds[0] >= 0.05
    assert engine.lambda_bounds[1] <= 15.0
    assert engine.surprise_history_max <= 500

    # Update with matches
    matches = [
        {
            "home_team": "A",
            "away_team": "B",
            "home_goals": 2,
            "away_goals": 1,
            "league": "Premier League",
        },
        {
            "home_team": "C",
            "away_team": "D",
            "home_goals": 0,
            "away_goals": 0,
            "league": "Premier League",
        },
    ]

    for match in matches:
        engine.update(
            home_team=match["home_team"],
            away_team=match["away_team"],
            home_goals=match["home_goals"],
            away_goals=match["away_goals"],
            league=match["league"],
        )

    # Verify surprise history is bounded
    assert len(engine._surprise_history) <= 200

    # Test prediction
    pred = engine.predict("A", "B")
    assert 0.0 <= pred.lambda_home <= 5.0, f"Lambda bounds violated: {pred.lambda_home}"
    assert 0.0 <= pred.lambda_away <= 5.0, f"Lambda bounds violated: {pred.lambda_away}"
    assert 0.0 <= pred.zero_inflation <= 0.15

    print(f"✓ Surprise history bounded: {len(engine._surprise_history)} ≤ 200")
    print(f"✓ Lambda home: {pred.lambda_home:.3f} (within bounds [0.1, 5.0])")
    print(f"✓ Zero-inflation: {pred.zero_inflation:.4f} (calibrated)")

    return True


def test_value_edge_computation():
    """Test enhanced value edge computation."""
    print("\n" + "="*70)
    print("TEST 4: Value Edge Computation")
    print("="*70)

    # Simple case: model thinks home is 60%, market thinks 50%
    model_probs = (0.60, 0.25, 0.15)
    market_probs = (0.50, 0.30, 0.20)

    edge_h, edge_d, edge_a = compute_value_edges(
        model_probs,
        market_probs,
        model_confidence=0.8,
    )

    # Edge should be positive for home (0.60 - 0.50 = 0.10, weighted by 0.8 = 0.08)
    assert edge_h > 0, f"Expected positive home edge, got {edge_h}"
    assert abs(edge_h - 0.08) < 0.01, f"Expected ~0.08, got {edge_h}"

    print(f"✓ Home edge: {edge_h:.4f} (model favors home over market)")
    print(f"✓ Draw edge: {edge_d:.4f}")
    print(f"✓ Away edge: {edge_a:.4f}")

    # Test with odds
    market_odds = (2.0, 3.5, 4.0)  # Implied: 0.50, 0.29, 0.25 (with margin)
    edge_h2, edge_d2, edge_a2 = compute_value_edges(
        model_probs,
        market_probs,
        market_odds=market_odds,
        model_confidence=0.8,
    )

    print(f"✓ With odds: home edge={edge_h2:.4f}")

    return True


def test_upset_risk_scoring():
    """Test multi-signal upset risk."""
    print("\n" + "="*70)
    print("TEST 5: Upset Risk Scoring")
    print("="*70)

    # Strong favorite: 70% home
    strong_fav = (0.70, 0.20, 0.10)
    market_agrees = (0.65, 0.25, 0.10)

    risk_strong = compute_upset_risk(
        strong_fav,
        market_agrees,
        model_agreement=0.9,
        prediction_spread=0.05,
        confidence=0.9,
    )

    # Weak favorite: 55% home with model disagreement
    weak_fav = (0.55, 0.30, 0.15)
    market_disagrees = (0.50, 0.35, 0.15)

    risk_weak = compute_upset_risk(
        weak_fav,
        market_disagrees,
        model_agreement=0.5,  # Low agreement
        prediction_spread=0.15,  # High spread
        confidence=0.5,  # Low confidence
    )

    assert risk_weak > risk_strong, "Weak favorite should have higher upset risk"
    assert 0.0 <= risk_strong <= 1.0 and 0.0 <= risk_weak <= 1.0

    print(f"✓ Strong favorite upset risk: {risk_strong:.4f}")
    print(f"✓ Weak favorite upset risk: {risk_weak:.4f}")
    print(f"✓ Multi-signal fusion working correctly")

    return True


def test_jensen_shannon_divergence():
    """Test Jensen-Shannon divergence for model agreement."""
    print("\n" + "="*70)
    print("TEST 6: Jensen-Shannon Divergence & Model Agreement")
    print("="*70)

    # Identical distributions
    p1 = (0.5, 0.3, 0.2)
    p2 = (0.5, 0.3, 0.2)
    div = jensen_shannon_divergence(p1, p2)
    assert div < 0.001, f"Identical distributions should have JS≈0, got {div}"

    # Different distributions
    p3 = (0.7, 0.2, 0.1)
    div2 = jensen_shannon_divergence(p1, p3)
    assert div2 > div, "Different distributions should have higher JS"

    print(f"✓ Identical dists JS divergence: {div:.6f}")
    print(f"✓ Different dists JS divergence: {div2:.4f}")

    # Model agreement from multiple sets
    prob_sets = [
        (0.50, 0.30, 0.20),
        (0.51, 0.29, 0.20),  # Close
        (0.52, 0.28, 0.20),  # Close
    ]
    agreement = model_agreement_from_divergences(prob_sets)
    assert 0.0 <= agreement <= 1.0
    assert agreement > 0.8, f"Close models should have high agreement, got {agreement}"

    print(f"✓ Model agreement (close): {agreement:.4f}")

    # Low agreement
    prob_sets_low = [
        (0.60, 0.25, 0.15),
        (0.40, 0.40, 0.20),  # Very different
    ]
    agreement_low = model_agreement_from_divergences(prob_sets_low)
    assert agreement_low < agreement, "Divergent models should have lower agreement"

    print(f"✓ Model agreement (divergent): {agreement_low:.4f}")

    return True


def test_aggregate_predictions():
    """Test unified prediction aggregation with BMA."""
    print("\n" + "="*70)
    print("TEST 7: Unified Prediction Aggregation (BMA)")
    print("="*70)

    council = (0.52, 0.28, 0.20)
    bayesian = (0.50, 0.30, 0.20)
    statistical = (0.51, 0.29, 0.20)
    market = (0.48, 0.32, 0.20)

    # Test with BMA weights from accuracy
    expert_accuracies = {
        "council": 0.65,
        "bayesian": 0.60,
        "statistical": 0.58,
        "market": 0.55,
    }

    pred = aggregate_predictions(
        council_probs=council,
        bayesian_probs=bayesian,
        statistical_probs=statistical,
        market_probs=market,
        use_bayesian_averaging=True,
        expert_accuracies=expert_accuracies,
        lambda_h=1.3,
        lambda_a=1.1,
    )

    # Verify output
    assert 0.0 <= pred.p_home <= 1.0
    assert 0.0 <= pred.p_draw <= 1.0
    assert 0.0 <= pred.p_away <= 1.0
    assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 0.01

    assert pred.model_agreement > 0.8, f"Should have high agreement, got {pred.model_agreement}"
    assert 0.0 <= pred.upset_risk <= 1.0
    assert -1.0 <= pred.value_edge_home <= 1.0

    print(f"✓ Aggregated prediction: P(H)={pred.p_home:.4f}, P(D)={pred.p_draw:.4f}, P(A)={pred.p_away:.4f}")
    print(f"✓ Model agreement: {pred.model_agreement:.4f}")
    print(f"✓ Upset risk: {pred.upset_risk:.4f}")
    print(f"✓ Value edge (home): {pred.value_edge_home:.4f}")
    print(f"✓ Confidence: {pred.confidence:.4f}")
    print(f"✓ Prediction interval: {pred.prediction_interval}")

    return True


def main():
    """Run all tests."""
    print("\n" + "█"*70)
    print("█ ADVANCED MATHEMATICAL IMPLEMENTATIONS TEST SUITE")
    print("█"*70)

    tests = [
        ("TrueSkill Engine", test_trueskill_engine),
        ("TrueSkill Expert", test_trueskill_expert),
        ("Bayesian Engine Enhancements", test_bayesian_engine_enhancements),
        ("Value Edge Computation", test_value_edge_computation),
        ("Upset Risk Scoring", test_upset_risk_scoring),
        ("Jensen-Shannon Divergence", test_jensen_shannon_divergence),
        ("Unified Prediction Aggregation", test_aggregate_predictions),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "█"*70)
    print(f"█ RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("█"*70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
