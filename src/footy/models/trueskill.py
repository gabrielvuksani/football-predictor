"""TrueSkill rating system for football prediction.

Implements a full Bayesian skill rating system inspired by Microsoft's TrueSkill,
adapted for football with separate offense/defense components and team-specific
home advantage modeling.

Mathematical basis:
    - Gaussian skill model: N(μ, σ²) per team component
    - Bayesian factor update after each match
    - Rating deviation grows with inactivity (uncertainty)
    - Separate attack/defense skill hierarchy
    - Home advantage learned from data
    - Draw probability derived from skill overlap

The model maintains posterior beliefs about team strengths and updates them
using factor graphs (simplified Kalman-filter approach for computational efficiency).

References:
    Herbrich et al. (2006) "TrueSkill: A Bayesian Skill Rating System"
    Mohan et al. (2012) "Predicting Football Matches in Python"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm as normal_dist


# ═══════════════════════════════════════════════════════════════════════════
# SKILL MODEL & STATE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Skill:
    """Gaussian belief about a single skill dimension (attack or defense).

    Represents N(μ, σ²) where:
        μ = posterior mean skill
        σ² = posterior variance (uncertainty)
    """
    mu: float = 1500.0  # Mean rating
    sigma: float = 350.0  # Std dev (uncertainty)

    def __post_init__(self):
        """Clamp parameters to valid ranges."""
        self.mu = float(np.clip(self.mu, 0.0, 5000.0))
        self.sigma = float(np.clip(self.sigma, 10.0, 1000.0))

    @property
    def variance(self) -> float:
        """Variance σ²."""
        return self.sigma ** 2

    def copy(self) -> Skill:
        """Deep copy."""
        return Skill(mu=self.mu, sigma=self.sigma)

    def to_tuple(self) -> tuple[float, float]:
        """Export as (μ, σ)."""
        return (self.mu, self.sigma)

    def __repr__(self) -> str:
        return f"Skill(μ={self.mu:.1f}, σ={self.sigma:.1f})"


@dataclass
class TeamSkills:
    """Complete skill state for a team.

    Maintains separate Gaussian models for attack and defense,
    plus home advantage adjustment and recent activity tracking.
    """
    attack: Skill = field(default_factory=lambda: Skill())
    defense: Skill = field(default_factory=lambda: Skill())
    home_advantage: float = 0.0  # Learned home ground effect (log-goal-rate scale)
    matches_played: int = 0
    last_match_idx: int = -1  # Last match this team played (for inactivity penalty)

    def uncertainty(self) -> float:
        """Total uncertainty as sum of σ²."""
        return math.sqrt(self.attack.variance + self.defense.variance)

    def overall_strength(self) -> float:
        """Overall team strength = attack - defense."""
        return self.attack.mu - self.defense.mu

    def copy(self) -> TeamSkills:
        """Deep copy."""
        return TeamSkills(
            attack=self.attack.copy(),
            defense=self.defense.copy(),
            home_advantage=self.home_advantage,
            matches_played=self.matches_played,
            last_match_idx=self.last_match_idx,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TRUESKILL ENGINE
# ═══════════════════════════════════════════════════════════════════════════


class TrueSkillEngine:
    """Production Bayesian skill rating system for football.

    Maintains team skills, updates them after observed matches, and produces
    calibrated probability predictions with uncertainty quantification.
    """

    # Hyperparameters (per TrueSkill paper, tuned for football)
    INITIAL_MU = 1500.0
    INITIAL_SIGMA = 350.0
    BETA = 100.0  # Measurement noise std dev (skill factor to observed score)
    TAU = 10.0  # Dynamics noise (σ increase per time step for inactivity)
    DRAW_MARGIN = 45.0  # How close must teams be for draw (in rating space)

    def __init__(
        self,
        initial_mu: float = INITIAL_MU,
        initial_sigma: float = INITIAL_SIGMA,
        beta: float = BETA,
        tau: float = TAU,
        draw_margin: float = DRAW_MARGIN,
        home_advantage: float = 45.0,
    ):
        """Initialize engine.

        Args:
            initial_mu: Prior mean rating for new teams
            initial_sigma: Prior uncertainty for new teams
            beta: Observation noise (higher = less belief in results)
            tau: Inactivity penalty rate (higher = faster uncertainty growth)
            draw_margin: Rating gap for draw region (Δμ < draw_margin → likely draw)
            home_advantage: Initial home ground effect in rating points
        """
        self.initial_mu = initial_mu
        self.initial_sigma = initial_sigma
        self.beta = beta
        self.tau = tau
        self.draw_margin = draw_margin
        self.home_advantage_prior = home_advantage

        self.teams: dict[str, TeamSkills] = {}
        self._match_count = 0
        self._league_updates: dict[str, int] = {}  # Track league breaks for momentum reset

    def _get_team(self, team: str) -> TeamSkills:
        """Get or create team skill state."""
        if team not in self.teams:
            self.teams[team] = TeamSkills(
                attack=Skill(mu=self.initial_mu, sigma=self.initial_sigma),
                defense=Skill(mu=self.initial_mu, sigma=self.initial_sigma),
                home_advantage=0.0,
            )
        return self.teams[team]

    def apply_inactivity_penalty(self, team: str, current_match_idx: int) -> None:
        """Increase uncertainty for teams that haven't played recently.

        Gaussian process model: σ²(t) = σ²(0) + τ² * Δt
        """
        ts = self._get_team(team)
        matches_since = max(0, current_match_idx - ts.last_match_idx)

        if matches_since > 1:
            # Add uncertainty (dynamics noise) proportional to time gap
            decay = math.sqrt(matches_since) * self.tau
            ts.attack.sigma = float(np.clip(
                math.sqrt(ts.attack.variance + decay**2),
                10.0, 1000.0
            ))
            ts.defense.sigma = float(np.clip(
                math.sqrt(ts.defense.variance + decay**2),
                10.0, 1000.0
            ))

    def predict_probs(
        self,
        home: str,
        away: str,
        con: Optional[str] = None,
    ) -> tuple[float, float, float]:
        """Predict match probabilities: (p_home, p_draw, p_away).

        Args:
            home: Home team name
            away: Away team name
            con: Competition/league (optional, for stats tracking)

        Returns:
            (p_home, p_draw, p_away) summing to 1.0
        """
        h_skills = self._get_team(home)
        a_skills = self._get_team(away)

        # Skill difference: home attack advantage vs away defense
        # Plus learned home advantage
        skill_diff = (
            (h_skills.attack.mu - a_skills.defense.mu) +
            h_skills.home_advantage -
            (a_skills.attack.mu - h_skills.defense.mu)
        ) / 2.0

        # Total uncertainty in match outcome
        outcome_variance = (
            h_skills.attack.variance + a_skills.defense.variance +
            a_skills.attack.variance + h_skills.defense.variance +
            2 * self.beta ** 2  # Measurement noise
        )
        outcome_std = math.sqrt(outcome_variance)

        # Probability of home win (no draw): P(skill_diff > 0)
        p_home_win = normal_dist.cdf(skill_diff / outcome_std)

        # Draw probability: P(|skill_diff| < draw_margin)
        draw_prob = (
            normal_dist.cdf((self.draw_margin - skill_diff) / outcome_std) -
            normal_dist.cdf((-self.draw_margin - skill_diff) / outcome_std)
        )

        # Adjust for extreme probabilities (clamp away from 0 and 1)
        draw_prob = float(np.clip(draw_prob, 0.05, 0.4))

        # Distribute remaining probability
        p_away_win = 1.0 - draw_prob - p_home_win
        p_home_win = max(0.05, min(0.95, p_home_win))
        p_away_win = max(0.05, min(0.95, p_away_win))

        # Normalize
        total = p_home_win + draw_prob + p_away_win
        return (
            float(p_home_win / total),
            float(draw_prob / total),
            float(p_away_win / total),
        )

    def update(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        match_idx: int = 0,
        league: Optional[str] = None,
    ) -> None:
        """Update skills after observing a match result.

        Uses Bayesian factor update (simplified from full message-passing):
        - Outcome is modeled as Gaussian with skill diff as mean, β² as variance
        - Kalman-filter-style update: K = var_prior / (var_prior + var_obs)

        Args:
            home: Home team
            away: Away team
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
            match_idx: Chronological match index (for inactivity tracking)
            league: League/competition (optional)
        """
        h_skills = self._get_team(home)
        a_skills = self._get_team(away)

        # Apply inactivity penalties BEFORE update
        self.apply_inactivity_penalty(home, match_idx)
        self.apply_inactivity_penalty(away, match_idx)

        # Reset uncertainty for league breaks if tracked
        if league:
            prev_league = self._league_updates.get(home)
            if prev_league and prev_league != league:
                # League changed (e.g., international break) → reset momentum
                h_skills.home_advantage = h_skills.home_advantage * 0.8
            self._league_updates[home] = league

            prev_league = self._league_updates.get(away)
            if prev_league and prev_league != league:
                a_skills.home_advantage = a_skills.home_advantage * 0.8
            self._league_updates[away] = league

        # Observed outcome (goal difference)
        outcome = home_goals - away_goals

        # Predicted means (without draw margin — pure continuous prediction)
        h_attack_pred = h_skills.attack.mu
        h_def_pred = h_skills.defense.mu
        a_attack_pred = a_skills.attack.mu
        a_def_pred = a_skills.defense.mu

        predicted_diff = (
            (h_attack_pred - a_def_pred) +
            h_skills.home_advantage -
            (a_attack_pred - h_def_pred)
        ) / 2.0

        # Total variance in outcome
        outcome_variance = (
            h_skills.attack.variance + a_skills.defense.variance +
            a_skills.attack.variance + h_skills.defense.variance +
            2 * self.beta ** 2
        )
        outcome_std = math.sqrt(outcome_variance)

        # Kalman gain (weight for update)
        kalman_gain = 1.0 / (outcome_std + 1e-9)

        # Innovation (prediction error)
        innovation = outcome - predicted_diff

        # Update home team attack (scored goals vs defense conceded)
        h_attack_gain = (
            h_skills.attack.variance / (h_skills.attack.variance + self.beta ** 2 + 1e-9)
        )
        h_skills.attack.mu += h_attack_gain * innovation * 0.5
        h_skills.attack.sigma = float(np.clip(
            math.sqrt((1 - h_attack_gain) * h_skills.attack.variance),
            10.0, 1000.0
        ))

        # Update home team defense (goals conceded)
        h_def_gain = (
            h_skills.defense.variance / (h_skills.defense.variance + self.beta ** 2 + 1e-9)
        )
        h_skills.defense.mu -= h_def_gain * innovation * 0.5
        h_skills.defense.sigma = float(np.clip(
            math.sqrt((1 - h_def_gain) * h_skills.defense.variance),
            10.0, 1000.0
        ))

        # Update away team (symmetric)
        a_attack_gain = (
            a_skills.attack.variance / (a_skills.attack.variance + self.beta ** 2 + 1e-9)
        )
        a_skills.attack.mu -= a_attack_gain * innovation * 0.5
        a_skills.attack.sigma = float(np.clip(
            math.sqrt((1 - a_attack_gain) * a_skills.attack.variance),
            10.0, 1000.0
        ))

        a_def_gain = (
            a_skills.defense.variance / (a_skills.defense.variance + self.beta ** 2 + 1e-9)
        )
        a_skills.defense.mu += a_def_gain * innovation * 0.5
        a_skills.defense.sigma = float(np.clip(
            math.sqrt((1 - a_def_gain) * a_skills.defense.variance),
            10.0, 1000.0
        ))

        # Update home advantage (slow learning)
        if home_goals > away_goals:
            # Home team won at home → reinforce home advantage
            h_skills.home_advantage = 0.95 * h_skills.home_advantage + 0.05 * 20.0
        elif home_goals < away_goals:
            # Home team lost at home → reduce home advantage belief
            h_skills.home_advantage = 0.95 * h_skills.home_advantage - 0.05 * 10.0

        # Clamp home advantage
        h_skills.home_advantage = float(np.clip(h_skills.home_advantage, -60.0, 120.0))

        # Update metadata
        h_skills.matches_played += 1
        h_skills.last_match_idx = match_idx
        a_skills.matches_played += 1
        a_skills.last_match_idx = match_idx
        self._match_count += 1

    def batch_update(
        self,
        matches: list[dict],
    ) -> None:
        """Process historical matches in chronological order.

        Each match dict should have:
            - home_team, away_team, home_goals, away_goals
            - optionally: match_idx, league
        """
        for idx, match in enumerate(matches):
            self.update(
                home=match["home_team"],
                away=match["away_team"],
                home_goals=int(match["home_goals"]),
                away_goals=int(match["away_goals"]),
                match_idx=match.get("match_idx", idx),
                league=match.get("league"),
            )

    def get_rankings(self, top_n: int = 30) -> list[dict]:
        """Get current team strength rankings.

        Returns:
            List of dicts with team, overall strength, attack, defense, etc.
        """
        rankings = []
        for team, skills in self.teams.items():
            rankings.append({
                "team": team,
                "overall_strength": round(skills.overall_strength(), 2),
                "attack": round(skills.attack.mu, 1),
                "attack_uncertainty": round(skills.attack.sigma, 1),
                "defense": round(skills.defense.mu, 1),
                "defense_uncertainty": round(skills.defense.sigma, 1),
                "home_advantage": round(skills.home_advantage, 1),
                "matches_played": skills.matches_played,
                "uncertainty": round(skills.uncertainty(), 1),
            })

        rankings.sort(key=lambda x: x["overall_strength"], reverse=True)
        return rankings[:top_n]

    def export_state(self) -> dict[str, dict]:
        """Export all team states for persistence.

        Returns dict mapping team name to skill state dict.
        """
        state = {}
        for team, skills in self.teams.items():
            state[team] = {
                "attack_mu": skills.attack.mu,
                "attack_sigma": skills.attack.sigma,
                "defense_mu": skills.defense.mu,
                "defense_sigma": skills.defense.sigma,
                "home_advantage": skills.home_advantage,
                "matches_played": skills.matches_played,
                "last_match_idx": skills.last_match_idx,
            }
        return state

    def import_state(self, state: dict[str, dict]) -> None:
        """Import team states (from persistence).

        Args:
            state: Dict mapping team name to skill state dict
        """
        self.teams.clear()
        for team, data in state.items():
            self.teams[team] = TeamSkills(
                attack=Skill(mu=data["attack_mu"], sigma=data["attack_sigma"]),
                defense=Skill(mu=data["defense_mu"], sigma=data["defense_sigma"]),
                home_advantage=data.get("home_advantage", 0.0),
                matches_played=data.get("matches_played", 0),
                last_match_idx=data.get("last_match_idx", -1),
            )
