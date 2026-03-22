#!/usr/bin/env python3
"""Standalone test for new implementations (doesn't import broken EloExpert).

Tests:
    1. TrueSkill rating system
    2. Bayesian engine enhancements
"""
import sys
sys.path.insert(0, '/sessions/friendly-nice-dirac/mnt/football-predictor-main/src')

import math
import numpy as np
import pandas as pd
from footy.models.trueskill import TrueSkillEngine, Skill, TeamSkills
from footy.models.bayesian_engine import BayesianStateSpaceEngine


def test_trueskill():
    """Test TrueSkill system."""
    print("\n" + "="*70)
    print("TEST 1: TrueSkill Rating System")
    print("="*70)

    engine = TrueSkillEngine()

    # Simulate matches
    matches = [
        {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1},
        {"home_team": "B", "away_team": "C", "home_goals": 0, "away_goals": 3},
        {"home_team": "A", "away_team": "C", "home_goals": 1, "away_goals": 1},
    ]

    engine.batch_update(matches)

    # Predict
    p_h, p_d, p_a = engine.predict_probs("A", "B")
    assert 0.0 <= p_h <= 1.0 and 0.0 <= p_d <= 1.0 and 0.0 <= p_a <= 1.0
    assert abs(p_h + p_d + p_a - 1.0) < 0.01

    rankings = engine.get_rankings(top_n=10)
    assert len(rankings) == 3

    print(f"✓ Predictions: P(H)={p_h:.4f}, P(D)={p_d:.4f}, P(A)={p_a:.4f}")
    print(f"✓ Rankings: {[r['team'] for r in rankings]}")

    # Test state export/import
    state = engine.export_state()
    engine2 = TrueSkillEngine()
    engine2.import_state(state)
    p_h2, p_d2, p_a2 = engine2.predict_probs("A", "B")
    assert abs(p_h - p_h2) < 0.001

    print(f"✓ State export/import working")
    return True


def test_skill_class():
    """Test Skill class."""
    print("\n" + "="*70)
    print("TEST 2: Skill Class")
    print("="*70)

    skill = Skill(mu=1500.0, sigma=350.0)
    assert skill.mu == 1500.0
    assert skill.sigma == 350.0
    assert skill.variance == 350.0 ** 2

    # Test clamping
    skill2 = Skill(mu=10000.0, sigma=5000.0)
    assert skill2.mu <= 5000.0
    assert skill2.sigma <= 1000.0

    print(f"✓ Skill creation and clamping working")
    print(f"✓ Variance property: {skill.variance:.0f}")
    return True


def test_team_skills():
    """Test TeamSkills class."""
    print("\n" + "="*70)
    print("TEST 3: TeamSkills Class")
    print("="*70)

    ts = TeamSkills()
    assert ts.overall_strength() == 0.0  # μ_attack - μ_defense = 0
    assert ts.uncertainty() > 0

    ts.attack.mu = 1600.0
    ts.defense.mu = 1400.0
    assert ts.overall_strength() == 200.0

    print(f"✓ Overall strength: {ts.overall_strength():.1f}")
    print(f"✓ Uncertainty: {ts.uncertainty():.1f}")
    return True


def test_bayesian_enhancements():
    """Test Bayesian engine improvements."""
    print("\n" + "="*70)
    print("TEST 4: Enhanced Bayesian Engine")
    print("="*70)

    engine = BayesianStateSpaceEngine(
        surprise_history_max=200,
        lambda_bounds=(0.1, 5.0),
        per_league_zero_inflation=True,
    )

    # Verify parameters
    assert engine.rue_salvesen_gamma <= 0.15
    assert engine.lambda_bounds == (0.1, 5.0)
    assert engine.surprise_history_max <= 500

    # Update with matches
    matches = [
        {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1, "league": "PL"},
        {"home_team": "C", "away_team": "D", "home_goals": 0, "away_goals": 0, "league": "PL"},
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

    # Test prediction with bounds
    pred = engine.predict("A", "B")
    assert engine.lambda_bounds[0] <= pred.lambda_home <= engine.lambda_bounds[1]
    assert engine.lambda_bounds[0] <= pred.lambda_away <= engine.lambda_bounds[1]
    assert 0.0 <= pred.zero_inflation <= 0.15

    print(f"✓ Surprise history bounded: {len(engine._surprise_history)} ≤ 200")
    print(f"✓ Lambda home: {pred.lambda_home:.3f} (within [{engine.lambda_bounds[0]}, {engine.lambda_bounds[1]}])")
    print(f"✓ Zero-inflation: {pred.zero_inflation:.4f}")
    return True


def test_inactivity_penalty():
    """Test inactivity penalty in TrueSkill."""
    print("\n" + "="*70)
    print("TEST 5: TrueSkill Inactivity Penalty")
    print("="*70)

    engine = TrueSkillEngine(tau=10.0)

    # Update team A
    engine.update("A", "B", 2, 1, match_idx=0)
    ts_a1 = engine._get_team("A")
    sigma_after_match = ts_a1.attack.sigma

    # Apply inactivity penalty (20 matches later)
    engine.apply_inactivity_penalty("A", 20)
    ts_a2 = engine._get_team("A")
    sigma_after_inactivity = ts_a2.attack.sigma

    # Uncertainty should increase
    assert sigma_after_inactivity > sigma_after_match, \
        f"Inactivity should increase σ: {sigma_after_match:.1f} → {sigma_after_inactivity:.1f}"

    print(f"✓ Inactivity penalty working")
    print(f"✓ σ after match: {sigma_after_match:.1f}")
    print(f"✓ σ after 20-match gap: {sigma_after_inactivity:.1f}")
    return True


def test_home_advantage_learning():
    """Test home advantage learning in TrueSkill."""
    print("\n" + "="*70)
    print("TEST 6: TrueSkill Home Advantage Learning")
    print("="*70)

    engine = TrueSkillEngine()

    # Team A wins at home several times
    for i in range(3):
        engine.update("A", f"Team{i}", 2, 1, match_idx=i)

    ts_a = engine._get_team("A")
    home_adv = ts_a.home_advantage

    assert home_adv > 0, f"Home advantage should be positive after wins at home, got {home_adv}"

    print(f"✓ Home advantage learned: {home_adv:.1f}")
    return True


def test_parameter_validation():
    """Test parameter validation and clamping."""
    print("\n" + "="*70)
    print("TEST 7: Parameter Validation & Clamping")
    print("="*70)

    # Bayesian with extreme parameters
    engine = BayesianStateSpaceEngine(
        rue_salvesen_gamma=1.5,  # Should clamp to 0.15
        surprise_history_max=1000,  # Should clamp to 500
        lambda_bounds=(0.0, 20.0),  # Should clamp to [0.05, 15.0]
    )

    assert engine.rue_salvesen_gamma <= 0.15
    assert engine.surprise_history_max <= 500
    assert engine.lambda_bounds[0] >= 0.05
    assert engine.lambda_bounds[1] <= 15.0

    print(f"✓ rue_salvesen_gamma clamped: {engine.rue_salvesen_gamma:.4f} ≤ 0.15")
    print(f"✓ surprise_history_max clamped: {engine.surprise_history_max} ≤ 500")
    print(f"✓ lambda_bounds clamped: {engine.lambda_bounds}")
    return True


def main():
    """Run all tests."""
    print("\n" + "█"*70)
    print("█ STANDALONE TEST SUITE — NEW IMPLEMENTATIONS")
    print("█"*70)

    tests = [
        ("TrueSkill System", test_trueskill),
        ("Skill Class", test_skill_class),
        ("TeamSkills Class", test_team_skills),
        ("Bayesian Enhancements", test_bayesian_enhancements),
        ("Inactivity Penalty", test_inactivity_penalty),
        ("Home Advantage Learning", test_home_advantage_learning),
        ("Parameter Validation", test_parameter_validation),
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
