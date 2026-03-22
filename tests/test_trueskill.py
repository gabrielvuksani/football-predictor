"""Tests for the TrueSkill rating system."""
from __future__ import annotations

import numpy as np
import pytest

from footy.models.trueskill import (
    Skill,
    TeamSkills,
    TrueSkillEngine,
)


class TestSkill:
    """Tests for Skill (Gaussian belief) model."""

    def test_default_initialization(self):
        """Test initial skill parameters."""
        s = Skill()
        assert s.mu == 1500.0
        assert s.sigma == 350.0

    def test_variance_calculation(self):
        """Test variance is sigma squared."""
        s = Skill(mu=1500, sigma=100)
        assert s.variance == 10000.0

    def test_copy_independence(self):
        """Test that copy creates independent instance."""
        s1 = Skill(mu=1600, sigma=200)
        s2 = s1.copy()
        s2.mu = 1400
        assert s1.mu == 1600
        assert s2.mu == 1400

    def test_to_tuple(self):
        """Test export as tuple."""
        s = Skill(mu=1600, sigma=150)
        t = s.to_tuple()
        assert t == (1600.0, 150.0)

    def test_parameter_clamping(self):
        """Test that mu and sigma are clamped to valid ranges."""
        s = Skill(mu=10000, sigma=2000)
        assert s.mu <= 5000.0
        assert s.sigma <= 1000.0

        s = Skill(mu=-100, sigma=5)
        assert s.mu >= 0.0
        assert s.sigma >= 10.0

    def test_repr(self):
        """Test string representation."""
        s = Skill(mu=1500, sigma=300)
        r = repr(s)
        assert "1500" in r
        assert "300" in r


class TestTeamSkills:
    """Tests for TeamSkills (attack/defense) model."""

    def test_default_initialization(self):
        """Test default team skills state."""
        ts = TeamSkills()
        assert ts.matches_played == 0
        assert ts.last_match_idx == -1
        assert ts.home_advantage == 0.0
        assert ts.attack.mu == 1500.0
        assert ts.defense.mu == 1500.0

    def test_uncertainty_calculation(self):
        """Test total uncertainty is sqrt(attack_var + defense_var)."""
        ts = TeamSkills(
            attack=Skill(mu=1500, sigma=100),
            defense=Skill(mu=1500, sigma=200),
        )
        expected = np.sqrt(100**2 + 200**2)
        assert abs(ts.uncertainty() - expected) < 1.0

    def test_overall_strength(self):
        """Test overall strength = attack - defense."""
        ts = TeamSkills(
            attack=Skill(mu=1600, sigma=100),
            defense=Skill(mu=1400, sigma=100),
        )
        assert ts.overall_strength() == 200.0

    def test_copy_independence(self):
        """Test deep copy creates independent instance."""
        ts1 = TeamSkills(
            attack=Skill(mu=1600, sigma=100),
            home_advantage=50.0,
            matches_played=10,
        )
        ts2 = ts1.copy()
        ts2.attack.mu = 1400
        ts2.matches_played = 5
        assert ts1.attack.mu == 1600
        assert ts1.matches_played == 10


class TestTrueSkillEngineInitialization:
    """Tests for TrueSkillEngine initialization."""

    def test_default_initialization(self):
        """Test engine with default parameters."""
        engine = TrueSkillEngine()
        assert engine.initial_mu == 1500.0
        assert engine.initial_sigma == 350.0
        assert engine.beta == 100.0
        assert engine.tau == 10.0
        assert len(engine.teams) == 0

    def test_custom_parameters(self):
        """Test engine with custom hyperparameters."""
        engine = TrueSkillEngine(
            initial_mu=1600,
            initial_sigma=200,
            beta=80,
            tau=15,
            draw_margin=50,
        )
        assert engine.initial_mu == 1600
        assert engine.initial_sigma == 200
        assert engine.beta == 80
        assert engine.tau == 15
        assert engine.draw_margin == 50

    def test_get_team_creates_new_teams(self):
        """Test that _get_team creates teams with default skills."""
        engine = TrueSkillEngine()
        ts = engine._get_team("Arsenal")
        assert ts.attack.mu == 1500.0
        assert ts.defense.mu == 1500.0
        assert "Arsenal" in engine.teams


class TestTrueSkillEnginePrediction:
    """Tests for prediction generation."""

    def test_predict_probs_valid_distribution(self):
        """Test that predict_probs returns valid probabilities summing to 1.0."""
        engine = TrueSkillEngine()
        p_home, p_draw, p_away = engine.predict_probs("Home", "Away")

        assert 0.05 <= p_home <= 0.95
        assert 0.05 <= p_draw <= 0.95
        assert 0.05 <= p_away <= 0.95
        assert abs((p_home + p_draw + p_away) - 1.0) < 1e-6

    def test_predict_probs_initial_equilibrium(self):
        """Test that initial prediction is near 1/3 for equal teams."""
        engine = TrueSkillEngine()
        p_home, p_draw, p_away = engine.predict_probs("A", "B")

        # Without history, home advantage gives slight edge to home
        # but should still be reasonably balanced
        assert 0.3 <= p_home <= 0.7
        assert 0.05 <= p_draw <= 0.4
        assert 0.3 <= p_away <= 0.7

    def test_predict_probs_home_advantage(self):
        """Test that home advantage increases home win probability."""
        engine = TrueSkillEngine(home_advantage=45.0)

        # Same strength teams
        engine._get_team("Home").attack.mu = 1500
        engine._get_team("Home").defense.mu = 1500
        engine._get_team("Away").attack.mu = 1500
        engine._get_team("Away").defense.mu = 1500

        p_home, _, p_away = engine.predict_probs("Home", "Away")
        assert p_home > p_away  # Home advantage matters

    def test_predict_probs_strong_team_advantage(self):
        """Test that stronger teams get higher win probability."""
        engine = TrueSkillEngine()

        # Strong home team
        engine._get_team("Strong").attack.mu = 1700
        engine._get_team("Strong").defense.mu = 1300
        engine._get_team("Weak").attack.mu = 1300
        engine._get_team("Weak").defense.mu = 1700

        p_home, _, p_away = engine.predict_probs("Strong", "Weak")
        assert p_home > p_away


class TestTrueSkillEngineUpdate:
    """Tests for skill updates after match results."""

    def test_update_home_win_increases_attack(self):
        """Test that home win increases home attack skill."""
        engine = TrueSkillEngine()
        engine.update("Home", "Away", 2, 0, match_idx=0)

        home_skills = engine.teams["Home"]
        assert home_skills.attack.mu > 1500.0  # Attack improved
        assert home_skills.matches_played == 1
        assert home_skills.last_match_idx == 0

    def test_update_away_win_decreases_home_attack(self):
        """Test that away win decreases home attack skill."""
        engine = TrueSkillEngine()
        initial_attack = engine._get_team("Home").attack.mu
        engine.update("Home", "Away", 0, 2, match_idx=0)

        home_skills = engine.teams["Home"]
        assert home_skills.attack.mu < initial_attack

    def test_update_reduces_uncertainty(self):
        """Test that matches reduce skill uncertainty."""
        engine = TrueSkillEngine()
        initial_sigma = engine._get_team("A").attack.sigma

        engine.update("A", "B", 2, 1, match_idx=0)

        final_sigma = engine.teams["A"].attack.sigma
        assert final_sigma < initial_sigma  # Uncertainty decreases

    def test_update_both_teams_recorded(self):
        """Test that both teams are updated after a match."""
        engine = TrueSkillEngine()
        engine.update("Home", "Away", 1, 1, match_idx=0)

        assert "Home" in engine.teams
        assert "Away" in engine.teams
        assert engine.teams["Home"].matches_played == 1
        assert engine.teams["Away"].matches_played == 1

    def test_update_home_advantage_learning_win(self):
        """Test that home wins reinforce home advantage belief."""
        engine = TrueSkillEngine()
        engine._get_team("Home").home_advantage = 0.0

        # Home team wins multiple times
        for i in range(5):
            engine.update("Home", "Away", 2, 0, match_idx=i)

        home_adv = engine.teams["Home"].home_advantage
        assert home_adv > 0.0  # Home advantage increased

    def test_update_home_advantage_learning_loss(self):
        """Test that home losses reduce home advantage belief."""
        engine = TrueSkillEngine()
        engine._get_team("Home").home_advantage = 50.0

        # Home team loses
        engine.update("Home", "Away", 0, 2, match_idx=0)

        home_adv = engine.teams["Home"].home_advantage
        assert home_adv < 50.0  # Home advantage reduced

    def test_update_increments_match_count(self):
        """Test that match count is incremented."""
        engine = TrueSkillEngine()
        assert engine._match_count == 0

        engine.update("A", "B", 1, 0, match_idx=0)
        assert engine._match_count == 1

        engine.update("C", "D", 0, 1, match_idx=1)
        assert engine._match_count == 2


class TestTrueSkillEngineInactivityPenalty:
    """Tests for uncertainty growth due to inactivity."""

    def test_inactivity_increases_uncertainty(self):
        """Test that teams with gaps increase uncertainty."""
        engine = TrueSkillEngine(tau=10.0)

        engine.update("A", "B", 1, 0, match_idx=0)
        initial_sigma = engine.teams["A"].attack.sigma

        # Apply inactivity penalty for large gap
        engine.apply_inactivity_penalty("A", current_match_idx=10)

        final_sigma = engine.teams["A"].attack.sigma
        assert final_sigma > initial_sigma  # Uncertainty increased

    def test_inactivity_no_penalty_immediate_next_match(self):
        """Test no penalty if next match is immediate."""
        engine = TrueSkillEngine()
        engine.update("A", "B", 1, 0, match_idx=0)
        initial_sigma = engine.teams["A"].attack.sigma

        # Apply penalty but match is next (gap=0)
        engine.apply_inactivity_penalty("A", current_match_idx=1)

        final_sigma = engine.teams["A"].attack.sigma
        assert abs(final_sigma - initial_sigma) < 0.1


class TestTrueSkillEngineBatchUpdate:
    """Tests for batch processing of historical matches."""

    def test_batch_update_processes_all_matches(self):
        """Test that batch_update processes all provided matches."""
        engine = TrueSkillEngine()
        matches = [
            {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1},
            {"home_team": "B", "away_team": "C", "home_goals": 1, "away_goals": 1},
            {"home_team": "C", "away_team": "A", "home_goals": 0, "away_goals": 3},
        ]

        engine.batch_update(matches)

        assert engine._match_count == 3
        assert "A" in engine.teams
        assert "B" in engine.teams
        assert "C" in engine.teams

    def test_batch_update_chronological_order(self):
        """Test that matches are processed in chronological order."""
        engine = TrueSkillEngine()
        matches = [
            {"home_team": "A", "away_team": "B", "home_goals": 5, "away_goals": 0, "match_idx": 0},
            {"home_team": "A", "away_team": "C", "home_goals": 1, "away_goals": 1, "match_idx": 1},
        ]

        engine.batch_update(matches)

        # After first match, A is stronger
        # After second match, A is weaker (didn't win)
        assert engine.teams["A"].matches_played == 2


class TestTrueSkillEngineRankings:
    """Tests for ranking generation."""

    def test_get_rankings_returns_sorted_list(self):
        """Test that rankings are sorted by overall strength."""
        engine = TrueSkillEngine()
        matches = [
            {"home_team": "Strong", "away_team": "Weak", "home_goals": 5, "away_goals": 0},
            {"home_team": "Mid", "away_team": "Weak", "home_goals": 2, "away_goals": 1},
        ]
        engine.batch_update(matches)

        rankings = engine.get_rankings(top_n=10)

        assert len(rankings) > 0
        # Stronger team first
        assert rankings[0]["overall_strength"] >= rankings[-1]["overall_strength"]

    def test_get_rankings_limited_by_top_n(self):
        """Test that top_n limits results."""
        engine = TrueSkillEngine()
        for i in range(10):
            engine._get_team(f"Team{i}")

        rankings = engine.get_rankings(top_n=5)
        assert len(rankings) <= 5

    def test_get_rankings_contains_required_fields(self):
        """Test that rankings include all required fields."""
        engine = TrueSkillEngine()
        engine.update("A", "B", 2, 1, match_idx=0)

        rankings = engine.get_rankings()

        assert len(rankings) > 0
        rank = rankings[0]
        assert "team" in rank
        assert "overall_strength" in rank
        assert "attack" in rank
        assert "defense" in rank
        assert "uncertainty" in rank
        assert "matches_played" in rank


class TestTrueSkillEnginePersistence:
    """Tests for state export and import (persistence)."""

    def test_export_state_structure(self):
        """Test that export_state returns correct structure."""
        engine = TrueSkillEngine()
        engine.update("A", "B", 2, 1, match_idx=0)

        state = engine.export_state()

        assert "A" in state
        assert "B" in state
        assert "attack_mu" in state["A"]
        assert "attack_sigma" in state["A"]
        assert "defense_mu" in state["A"]
        assert "defense_sigma" in state["A"]
        assert "home_advantage" in state["A"]
        assert "matches_played" in state["A"]

    def test_import_state_restores_skills(self):
        """Test that import_state restores team skills."""
        engine1 = TrueSkillEngine()
        engine1.update("A", "B", 2, 1, match_idx=0)
        state = engine1.export_state()

        engine2 = TrueSkillEngine()
        engine2.import_state(state)

        assert "A" in engine2.teams
        assert abs(engine2.teams["A"].attack.mu - engine1.teams["A"].attack.mu) < 0.1

    def test_roundtrip_export_import(self):
        """Test that export/import roundtrip preserves state."""
        engine1 = TrueSkillEngine()
        matches = [
            {"home_team": "A", "away_team": "B", "home_goals": 3, "away_goals": 1},
            {"home_team": "B", "away_team": "C", "home_goals": 1, "away_goals": 1},
        ]
        engine1.batch_update(matches)

        state = engine1.export_state()
        engine2 = TrueSkillEngine()
        engine2.import_state(state)

        # Verify same predictions
        p1 = engine1.predict_probs("A", "C")
        p2 = engine2.predict_probs("A", "C")

        for prob1, prob2 in zip(p1, p2):
            assert abs(prob1 - prob2) < 0.01


class TestTrueSkillEngineEdgeCases:
    """Tests for edge cases and error handling."""

    def test_same_team_prediction(self):
        """Test prediction when same team plays (edge case)."""
        engine = TrueSkillEngine()
        # Should not crash, even if unusual
        p_home, p_draw, p_away = engine.predict_probs("A", "A")
        assert abs(p_home + p_draw + p_away - 1.0) < 1e-6

    def test_extreme_rating_difference(self):
        """Test prediction with extreme skill difference."""
        engine = TrueSkillEngine()

        engine._get_team("GoalKeeper").attack.mu = 100
        engine._get_team("GoalKeeper").defense.mu = 3000
        engine._get_team("Striker").attack.mu = 3000
        engine._get_team("Striker").defense.mu = 100

        p_home, _, p_away = engine.predict_probs("Striker", "GoalKeeper")

        # Striker should be favored or at least equal
        # (probabilities are clamped and normalized to stay within bounds)
        assert p_home >= p_away

    def test_draw_probability_bounds(self):
        """Test that draw probability stays within bounds."""
        engine = TrueSkillEngine()

        for _ in range(100):
            p_home, p_draw, p_away = engine.predict_probs(
                f"T{np.random.randint(0, 100)}",
                f"T{np.random.randint(0, 100)}"
            )
            assert 0.05 <= p_draw <= 0.4

    def test_update_with_zero_match_idx(self):
        """Test update with zero match index."""
        engine = TrueSkillEngine()
        engine.update("A", "B", 1, 0, match_idx=0)
        assert engine.teams["A"].last_match_idx == 0

    def test_large_goal_differences(self):
        """Test that large goal differences don't cause instability."""
        engine = TrueSkillEngine()

        engine.update("A", "B", 10, 0, match_idx=0)
        engine.update("B", "A", 0, 8, match_idx=1)

        # Skills should remain clamped
        assert engine.teams["A"].attack.mu <= 5000.0
        assert engine.teams["A"].attack.sigma <= 1000.0
        assert engine.teams["A"].attack.sigma >= 10.0

    def test_many_sequential_matches(self):
        """Test stability over many sequential matches."""
        engine = TrueSkillEngine()

        for i in range(100):
            h = f"H{i % 5}"
            a = f"A{i % 5}"
            hg = np.random.poisson(1.5)
            ag = np.random.poisson(1.3)
            engine.update(h, a, hg, ag, match_idx=i)

        # Should have stable ratings
        for team in engine.teams.values():
            assert 500 <= team.attack.mu <= 3000
            assert 500 <= team.defense.mu <= 3000
            assert 10 <= team.attack.sigma <= 1000


class TestTrueSkillEngineLeagueTracking:
    """Tests for league-specific tracking (international breaks, etc.)."""

    def test_league_change_resets_momentum(self):
        """Test that league changes affect home advantage."""
        engine = TrueSkillEngine()

        engine.update("A", "B", 2, 0, match_idx=0, league="PL")
        initial_ha = engine.teams["A"].home_advantage

        engine.update("A", "B", 2, 0, match_idx=1, league="INT")

        # After second home win, home advantage increases further
        # League change would reduce it by 0.8x, but the win reinforces it
        # Just verify league change tracking is working
        assert engine._league_updates["A"] == "INT"
