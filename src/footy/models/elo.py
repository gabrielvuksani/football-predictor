from __future__ import annotations
from dataclasses import dataclass
from footy.models.elo_core import (
    elo_expected as elo_expected_base,
    elo_draw_prob,
    dynamic_k as dynamic_k_base,
)

DEFAULT_RATING = 1500.0
HOME_ADV = 65.0
K_BASE = 20.0


@dataclass
class EloState:
    """In-memory Elo state for a session/batch."""
    team_match_counts: dict[str, int]
    counts_loaded: bool = False

    def load_match_counts(self, con) -> None:
        """Load team match counts from elo_applied records for K-factor persistence."""
        if self.counts_loaded:
            return
        try:
            rows = con.execute(
                "SELECT home_team, away_team FROM elo_applied"
            ).fetchall()
            for home, away in rows:
                self.team_match_counts[home] = self.team_match_counts.get(home, 0) + 1
                self.team_match_counts[away] = self.team_match_counts.get(away, 0) + 1
        except Exception:
            pass  # elo_applied may not exist yet
        self.counts_loaded = True

    def dynamic_k(self, team: str, goal_diff: int) -> float:
        """Dynamic K via shared elo_core with team-specific convergence."""
        n = self.team_match_counts.get(team, 0)
        return dynamic_k_base(n, goal_diff, k_base=K_BASE, convergence_matches=60)

    def ensure_team(self, con, team: str):
        con.execute("INSERT OR IGNORE INTO elo_state(team, rating) VALUES (?, ?)", [team, DEFAULT_RATING])

    def get_rating(self, con, team: str) -> float:
        self.ensure_team(con, team)
        return float(con.execute("SELECT rating FROM elo_state WHERE team=?", [team]).fetchone()[0])

    def set_rating(self, con, team: str, rating: float):
        con.execute("UPDATE elo_state SET rating=? WHERE team=?", [rating, team])

    def update_from_match(self, con, home: str, away: str, hg: int, ag: int):
        """Update Elo ratings after a match result."""
        self.load_match_counts(con)  # Ensure counts are loaded from DB
        r_home = self.get_rating(con, home) + HOME_ADV
        r_away = self.get_rating(con, away)
        exp_home = elo_expected_base(r_home, r_away)

        if hg > ag:
            s_home = 1.0
        elif hg == ag:
            s_home = 0.5
        else:
            s_home = 0.0

        goal_diff = hg - ag
        k_home = self.dynamic_k(home, goal_diff)
        k_away = self.dynamic_k(away, -goal_diff)
        k = (k_home + k_away) / 2.0  # average K for the match

        delta = k * (s_home - exp_home)
        self.set_rating(con, home, self.get_rating(con, home) + delta)
        self.set_rating(con, away, self.get_rating(con, away) - delta)

        # Track match counts
        self.team_match_counts[home] = self.team_match_counts.get(home, 0) + 1
        self.team_match_counts[away] = self.team_match_counts.get(away, 0) + 1

    def predict_probs(self, con, home: str, away: str) -> tuple[float, float, float]:
        """Predict match probabilities using Elo ratings."""
        r_home = self.get_rating(con, home) + HOME_ADV
        r_away = self.get_rating(con, away)
        e_home = elo_expected_base(r_home, r_away)

        # Dynamic draw probability based on rating closeness
        p_draw = elo_draw_prob(r_home, r_away)

        p_home_adj = e_home * (1.0 - p_draw)
        p_away_adj = (1.0 - e_home) * (1.0 - p_draw)
        s = p_home_adj + p_draw + p_away_adj
        return (p_home_adj / s, p_draw / s, p_away_adj / s)


# Global instance for backwards compatibility
_global_state = EloState(team_match_counts={})

# Backward compatibility: expose module-level globals that tests and other code may access
_team_match_counts = _global_state.team_match_counts
_counts_loaded = False  # Kept for compatibility but not actively used


def _expected(r_home: float, r_away: float) -> float:
    """Backward compatibility wrapper for elo_expected."""
    return elo_expected_base(r_home, r_away)


def _dynamic_k(team: str, goal_diff: int) -> float:
    """Backward compatibility wrapper for dynamic K calculation using global state."""
    return _global_state.dynamic_k(team, goal_diff)


def _load_match_counts(con) -> None:
    """Backward compatibility wrapper for loading match counts."""
    _global_state.load_match_counts(con)


def ensure_team(con, team: str):
    """Ensure team exists in elo_state; initialize if needed."""
    _global_state.ensure_team(con, team)

def get_rating(con, team: str) -> float:
    """Get current Elo rating for a team."""
    return _global_state.get_rating(con, team)

def set_rating(con, team: str, rating: float):
    """Set Elo rating for a team."""
    _global_state.set_rating(con, team, rating)

def update_from_match(con, home: str, away: str, hg: int, ag: int):
    """Update Elo ratings after a match result."""
    _global_state.update_from_match(con, home, away, hg, ag)

def predict_probs(con, home: str, away: str) -> tuple[float, float, float]:
    """Predict match probabilities using Elo ratings."""
    return _global_state.predict_probs(con, home, away)


def get_state() -> EloState:
    """Get the global Elo state instance (for advanced usage or testing)."""
    return _global_state


def reset_state() -> None:
    """Reset the global state (useful for testing)."""
    global _global_state
    _global_state = EloState(team_match_counts={})
