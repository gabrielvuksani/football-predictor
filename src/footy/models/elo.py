from __future__ import annotations
import math
from footy.db import connect

DEFAULT_RATING = 1500.0
HOME_ADV = 65.0
K_BASE = 20.0

# Dynamic K-factor: higher for decisive results, lower for close ones
# Also higher for new teams (fewer matches) to converge faster
_team_match_counts: dict[str, int] = {}

def _expected(r_home: float, r_away: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_home - r_away) / 400.0))

def _dynamic_k(team: str, goal_diff: int) -> float:
    """
    Dynamic K-factor that:
    1. Is higher for new teams (fast convergence)
    2. Scales with goal difference (decisive wins = more informative)
    3. Settles to base K after enough matches
    """
    n = _team_match_counts.get(team, 0)

    # Convergence factor: starts at 2.0x, decays to 1.0x over ~60 matches
    convergence = 1.0 + max(0.0, 1.0 - n / 60.0)

    # Goal diff multiplier: 1.0 for draws/1-goal, up to 1.5 for blowouts
    gd = abs(goal_diff)
    if gd <= 1:
        gd_mult = 1.0
    elif gd == 2:
        gd_mult = 1.15
    elif gd == 3:
        gd_mult = 1.30
    else:
        gd_mult = 1.30 + 0.05 * min(gd - 3, 4)  # cap at 1.50

    return K_BASE * convergence * gd_mult

def ensure_team(con, team: str):
    con.execute("INSERT OR IGNORE INTO elo_state(team, rating) VALUES (?, ?)", [team, DEFAULT_RATING])

def get_rating(con, team: str) -> float:
    ensure_team(con, team)
    return float(con.execute("SELECT rating FROM elo_state WHERE team=?", [team]).fetchone()[0])

def set_rating(con, team: str, rating: float):
    con.execute("UPDATE elo_state SET rating=? WHERE team=?", [rating, team])

def update_from_match(con, home: str, away: str, hg: int, ag: int):
    r_home = get_rating(con, home) + HOME_ADV
    r_away = get_rating(con, away)
    exp_home = _expected(r_home, r_away)

    if hg > ag:
        s_home = 1.0
    elif hg == ag:
        s_home = 0.5
    else:
        s_home = 0.0

    goal_diff = hg - ag
    k_home = _dynamic_k(home, goal_diff)
    k_away = _dynamic_k(away, -goal_diff)
    k = (k_home + k_away) / 2.0  # average K for the match

    delta = k * (s_home - exp_home)
    set_rating(con, home, get_rating(con, home) + delta)
    set_rating(con, away, get_rating(con, away) - delta)

    # Track match counts
    _team_match_counts[home] = _team_match_counts.get(home, 0) + 1
    _team_match_counts[away] = _team_match_counts.get(away, 0) + 1

def predict_probs(con, home: str, away: str) -> tuple[float, float, float]:
    r_home = get_rating(con, home) + HOME_ADV
    r_away = get_rating(con, away)
    e_home = _expected(r_home, r_away)

    # Dynamic draw probability based on rating closeness
    # When teams are close in rating, draws are more likely
    rating_diff = abs(r_home - r_away)
    # Base draw rate 0.26, increases up to 0.32 when diff is small
    p_draw = 0.26 + 0.06 * math.exp(-rating_diff / 200.0)
    # Decrease slightly when one team is much stronger
    p_draw = max(0.18, min(0.34, p_draw))

    p_home_adj = e_home * (1.0 - p_draw)
    p_away_adj = (1.0 - e_home) * (1.0 - p_draw)
    s = p_home_adj + p_draw + p_away_adj
    return (p_home_adj / s, p_draw / s, p_away_adj / s)
