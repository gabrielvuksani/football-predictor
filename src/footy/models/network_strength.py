"""Network-based team strength ratings: PageRank, Massey, Colley.

Models the league as a directed weighted graph where nodes are teams
and edges are match results. Derives strength ratings from network structure.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def pagerank_ratings(teams: list[str], results: list[dict[str, Any]],
                     damping: float = 0.85, max_iter: int = 100,
                     tol: float = 1e-6) -> dict[str, float]:
    """Compute PageRank-style team strength ratings.

    Teams are "linked to" by teams they have beaten (weighted by margin).

    Parameters
    ----------
    teams : list of team names
    results : list of dicts with keys: home_team, away_team, home_goals, away_goals
    damping : PageRank damping factor
    max_iter : maximum iterations
    tol : convergence tolerance

    Returns
    -------
    Dict mapping team name → PageRank score (higher = stronger).
    """
    n = len(teams)
    if n == 0:
        return {}

    team_idx = {t: i for i, t in enumerate(teams)}

    # Build adjacency matrix (weighted by goal strength)
    adj = np.zeros((n, n))
    for r in results:
        ht, at = r.get("home_team", ""), r.get("away_team", "")
        hg = r.get("home_goals")
        ag = r.get("away_goals")
        if ht not in team_idx or at not in team_idx or hg is None or ag is None:
            continue
        hi, ai = team_idx[ht], team_idx[at]
        margin = abs(hg - ag) + 1  # minimum weight 1 even for draws
        if hg > ag:
            adj[hi, ai] += margin  # home won: home "receives" link from away
        elif ag > hg:
            adj[ai, hi] += margin  # away won
        else:
            adj[hi, ai] += 0.5
            adj[ai, hi] += 0.5

    # Column-normalize (stochastic matrix)
    col_sums = adj.sum(axis=0)
    col_sums[col_sums == 0] = 1.0  # avoid division by zero
    M = adj / col_sums

    # Power iteration
    pr = np.ones(n) / n
    for _ in range(max_iter):
        new_pr = (1 - damping) / n + damping * M @ pr
        if np.abs(new_pr - pr).max() < tol:
            break
        pr = new_pr

    # Normalize to sum to 1
    total = pr.sum()
    if total > 0:
        pr /= total

    return {teams[i]: float(pr[i]) for i in range(n)}


def massey_ratings(teams: list[str],
                   results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute Massey ratings via least-squares on score margins.

    Solves the system: for each match, rating[home] - rating[away] ≈ home_goals - away_goals.

    Parameters
    ----------
    teams : list of team names
    results : list of dicts with home_team, away_team, home_goals, away_goals

    Returns
    -------
    Dict mapping team name → Massey rating.
    """
    n = len(teams)
    if n < 2:
        return {t: 0.0 for t in teams}

    team_idx = {t: i for i, t in enumerate(teams)}

    # Build Massey system
    M = np.zeros((n, n))
    p = np.zeros(n)

    for r in results:
        ht, at = r.get("home_team", ""), r.get("away_team", "")
        hg = r.get("home_goals")
        ag = r.get("away_goals")
        if ht not in team_idx or at not in team_idx or hg is None or ag is None:
            continue
        hi, ai = team_idx[ht], team_idx[at]

        M[hi, hi] += 1
        M[ai, ai] += 1
        M[hi, ai] -= 1
        M[ai, hi] -= 1

        p[hi] += hg - ag
        p[ai] += ag - hg

    # Replace last equation with sum-to-zero constraint
    M[-1, :] = 1.0
    p[-1] = 0.0

    try:
        ratings = np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        # Singular matrix, fallback to least squares
        ratings, _, _, _ = np.linalg.lstsq(M, p, rcond=None)

    return {teams[i]: float(ratings[i]) for i in range(n)}


def colley_ratings(teams: list[str],
                   results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute Colley matrix ratings (bias-free from win/loss records).

    The Colley method factors out strength of schedule and produces
    ratings purely from a team's record, adjusted for opponent quality.

    Parameters
    ----------
    teams : list of team names
    results : list of dicts with home_team, away_team, home_goals, away_goals

    Returns
    -------
    Dict mapping team name → Colley rating (0.5 = average).
    """
    n = len(teams)
    if n < 2:
        return {t: 0.5 for t in teams}

    team_idx = {t: i for i, t in enumerate(teams)}

    # Colley matrix: C[i,i] = 2 + total_games_i, C[i,j] = -n_games_ij
    C = np.zeros((n, n))
    b = np.ones(n)  # b[i] = 1 + (wins_i - losses_i) / 2, initialized to 1

    wins = np.zeros(n)
    losses = np.zeros(n)

    for r in results:
        ht, at = r.get("home_team", ""), r.get("away_team", "")
        hg = r.get("home_goals")
        ag = r.get("away_goals")
        if ht not in team_idx or at not in team_idx or hg is None or ag is None:
            continue
        hi, ai = team_idx[ht], team_idx[at]

        C[hi, hi] += 1
        C[ai, ai] += 1
        C[hi, ai] -= 1
        C[ai, hi] -= 1

        if hg > ag:
            wins[hi] += 1
            losses[ai] += 1
        elif ag > hg:
            wins[ai] += 1
            losses[hi] += 1
        else:
            wins[hi] += 0.5
            losses[hi] += 0.5
            wins[ai] += 0.5
            losses[ai] += 0.5

    # Add 2 to diagonal (Colley regularization)
    for i in range(n):
        C[i, i] += 2

    b += (wins - losses) / 2.0

    try:
        ratings = np.linalg.solve(C, b)
    except np.linalg.LinAlgError:
        ratings, _, _, _ = np.linalg.lstsq(C, b, rcond=None)

    return {teams[i]: float(ratings[i]) for i in range(n)}


def compute_all_ratings(teams: list[str],
                        results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute all network-based ratings for a set of teams.

    Returns
    -------
    Dict with keys 'pagerank', 'massey', 'colley', each mapping team → rating.
    """
    return {
        "pagerank": pagerank_ratings(teams, results),
        "massey": massey_ratings(teams, results),
        "colley": colley_ratings(teams, results),
    }
