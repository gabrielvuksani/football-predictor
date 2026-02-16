"""Advanced mathematical functions for football prediction.

Implements research-backed methods from the mathbasedtheorems.txt:
- Empirical Bayes shrinkage (Beta-Binomial)
- Dixon-Coles low-score correction (τ adjustment)
- Skellam distribution for goal-difference modelling
- Monte Carlo scoreline simulation
- Logit-space probability manipulation
- EWMA with adaptive time-decay
- Nonlinear strength transforms

References:
    Dixon & Coles (1997) "Modelling Association Football Scores and
        Inefficiencies in the Football Betting Market"
    Karlis & Ntzoufras (2003) "Analysis of sports data by using
        bivariate Poisson models"
    Gelman et al. "Bayesian Data Analysis" — empirical Bayes shrinkage
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import poisson as poisson_dist


# ═══════════════════════════════════════════════════════════════════
# 1. EMPIRICAL BAYES — BETA-BINOMIAL SHRINKAGE
# ═══════════════════════════════════════════════════════════════════
def beta_binomial_shrink(
    successes: float,
    trials: float,
    prior_alpha: float = 2.0,
    prior_beta: float = 2.0,
) -> float:
    """Shrink an observed rate towards a prior using Beta-Binomial conjugacy.

    For any rate p = successes/trials (win%, CS%, BTTS%, O2.5%):
        p_hat = (successes + α) / (trials + α + β)

    With small samples, p_hat → prior mean α/(α+β).
    With large samples, p_hat → observed rate.

    Args:
        successes: Number of success events (e.g., wins, clean sheets)
        trials: Total number of events (e.g., matches played)
        prior_alpha: Beta prior α (pseudo-successes)
        prior_beta: Beta prior β (pseudo-failures)

    Returns:
        Shrunk rate estimate
    """
    return (successes + prior_alpha) / (trials + prior_alpha + prior_beta)


def league_specific_prior(
    league: str,
    stat_type: str = "home_win",
) -> tuple[float, float]:
    """Return league-specific Beta prior (α, β) for a given statistic.

    Based on long-run league averages from historical data.
    """
    _PRIORS = {
        # (home_win_rate, draw_rate, away_win_rate) historical averages
        "PL":  {"home_win": (4.6, 5.4), "draw": (2.5, 7.5), "cs": (3.6, 6.4), "btts": (5.0, 5.0), "o25": (5.0, 5.0)},
        "PD":  {"home_win": (4.7, 5.3), "draw": (2.4, 7.6), "cs": (3.4, 6.6), "btts": (5.2, 4.8), "o25": (4.8, 5.2)},
        "SA":  {"home_win": (4.4, 5.6), "draw": (2.7, 7.3), "cs": (3.8, 6.2), "btts": (4.8, 5.2), "o25": (4.6, 5.4)},
        "BL1": {"home_win": (4.5, 5.5), "draw": (2.3, 7.7), "cs": (3.2, 6.8), "btts": (5.4, 4.6), "o25": (5.2, 4.8)},
        "FL1": {"home_win": (4.3, 5.7), "draw": (2.6, 7.4), "cs": (3.6, 6.4), "btts": (4.6, 5.4), "o25": (4.4, 5.6)},
    }
    default = {"home_win": (4.5, 5.5), "draw": (2.5, 7.5), "cs": (3.5, 6.5), "btts": (5.0, 5.0), "o25": (4.8, 5.2)}
    priors = _PRIORS.get(league, default)
    return priors.get(stat_type, (2.0, 2.0))


# ═══════════════════════════════════════════════════════════════════
# 2. DIXON-COLES τ CORRECTION
# ═══════════════════════════════════════════════════════════════════
def dixon_coles_tau(
    hg: int, ag: int,
    lambda_h: float, lambda_a: float,
    rho: float = -0.13,
) -> float:
    """Dixon-Coles adjustment factor τ(x, y, λ, μ, ρ) for low-scoring games.

    The standard independent Poisson model underestimates draws (especially 0-0)
    and overestimates 1-0/0-1 results. The τ factor corrects this.

    τ adjustments:
        (0,0): 1 - λ·μ·ρ    — increases 0-0 probability when ρ < 0
        (1,0): 1 + λ·ρ      — decreases 1-0 probability when ρ < 0
        (0,1): 1 + μ·ρ      — decreases 0-1 probability when ρ < 0
        (1,1): 1 - ρ         — increases 1-1 draws when ρ < 0
        else:  1             — no adjustment for higher scores

    Args:
        hg: Home goals scored
        ag: Away goals scored
        lambda_h: Home team expected goals (λ)
        lambda_a: Away team expected goals (μ)
        rho: Correlation parameter (typically -0.13 to -0.05)

    Returns:
        Multiplicative adjustment factor (τ ≥ 0)
    """
    if hg == 0 and ag == 0:
        return max(0.0, 1.0 - lambda_h * lambda_a * rho)
    elif hg == 1 and ag == 0:
        return max(0.0, 1.0 + lambda_h * rho)
    elif hg == 0 and ag == 1:
        return max(0.0, 1.0 + lambda_a * rho)
    elif hg == 1 and ag == 1:
        return max(0.0, 1.0 - rho)
    return 1.0


def build_dc_score_matrix(
    lambda_h: float,
    lambda_a: float,
    rho: float = -0.13,
    max_goals: int = 8,
) -> np.ndarray:
    """Build a Dixon-Coles adjusted score probability matrix.

    Returns (max_goals+1, max_goals+1) matrix where M[i,j] = P(home=i, away=j).
    """
    goals_range = np.arange(max_goals + 1)
    ph_dist = poisson_dist.pmf(goals_range, lambda_h)
    pa_dist = poisson_dist.pmf(goals_range, lambda_a)
    score_mx = np.outer(ph_dist, pa_dist)

    # Apply τ correction to low scores
    for i in range(min(2, max_goals + 1)):
        for j in range(min(2, max_goals + 1)):
            tau = dixon_coles_tau(i, j, lambda_h, lambda_a, rho)
            score_mx[i, j] *= tau

    # Renormalize
    total = score_mx.sum()
    if total > 0:
        score_mx /= total

    return score_mx


# ═══════════════════════════════════════════════════════════════════
# 3. SKELLAM DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════
def skellam_probs(
    lambda_h: float,
    lambda_a: float,
    max_diff: int = 6,
) -> dict[str, float]:
    """Compute goal-difference probabilities using Skellam distribution.

    The Skellam distribution models the difference of two independent
    Poisson random variables: D = X - Y ~ Skellam(λ₁, λ₂).

    Returns dict with home_win, draw, away_win, and moment features.
    """
    from scipy.stats import skellam as skellam_dist

    k_range = np.arange(-max_diff, max_diff + 1)
    pmf = skellam_dist.pmf(k_range, lambda_h, lambda_a)

    p_home = float(np.sum(pmf[k_range > 0]))
    p_draw = float(pmf[k_range == 0][0]) if 0 in k_range else 0.0
    p_away = float(np.sum(pmf[k_range < 0]))

    # Moments
    mean_diff = lambda_h - lambda_a
    var_diff = lambda_h + lambda_a
    std_diff = math.sqrt(max(var_diff, 1e-8))

    # Higher moments
    skew = (lambda_h - lambda_a) / max(var_diff ** 1.5, 1e-8) if var_diff > 0 else 0.0
    kurtosis = 1.0 / max(var_diff, 1e-8)  # excess kurtosis for Skellam

    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "mean_diff": mean_diff, "var_diff": var_diff, "std_diff": std_diff,
        "skewness": skew, "kurtosis": kurtosis,
    }


# ═══════════════════════════════════════════════════════════════════
# 4. MONTE CARLO SCORELINE SIMULATION
# ═══════════════════════════════════════════════════════════════════
def monte_carlo_simulate(
    lambda_h: float,
    lambda_a: float,
    rho: float = -0.13,
    n_sims: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Monte Carlo scoreline simulation with Dixon-Coles correlation.

    Draws n_sims scorelines from a bivariate Poisson model with
    Dixon-Coles low-score correction, then derives empirical probabilities.

    Args:
        lambda_h: Home expected goals
        lambda_a: Away expected goals
        rho: Dixon-Coles correlation parameter
        n_sims: Number of simulations
        rng: Random number generator (for reproducibility)

    Returns:
        Dict with p_home, p_draw, p_away, p_btts, p_o15, p_o25, p_o35,
        p_cs, mean_total, var_total, most_likely_score
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Generate correlated Poisson draws using rejection sampling
    # with Dixon-Coles adjustment
    h_goals = rng.poisson(lambda_h, n_sims)
    a_goals = rng.poisson(lambda_a, n_sims)

    # Apply DC correction via acceptance-rejection
    # For each simulated scoreline, compute acceptance probability
    # proportional to τ(h, a) / max_τ
    if abs(rho) > 1e-6:
        accept_probs = np.array([
            dixon_coles_tau(int(h), int(a), lambda_h, lambda_a, rho)
            for h, a in zip(h_goals, a_goals)
        ])
        accept_probs = np.clip(accept_probs, 0, 2)  # safety clamp
        # Normalize to max 1
        max_tau = accept_probs.max()
        if max_tau > 0:
            accept_probs /= max_tau
        # Accept/reject
        u = rng.uniform(0, 1, n_sims)
        mask = u < accept_probs
        h_goals = h_goals[mask]
        a_goals = a_goals[mask]

    n_accepted = len(h_goals)
    if n_accepted < 100:
        # Fallback to analytical if too few accepted
        return {
            "mc_p_home": 0.0, "mc_p_draw": 0.0, "mc_p_away": 0.0,
            "mc_btts": 0.0, "mc_o15": 0.0, "mc_o25": 0.0, "mc_o35": 0.0,
            "mc_cs": 0.0, "mc_mean_total": lambda_h + lambda_a,
            "mc_var_total": lambda_h + lambda_a,
            "mc_n_sims": 0,
        }

    total = h_goals + a_goals
    diff = h_goals.astype(int) - a_goals.astype(int)

    p_home = float(np.mean(diff > 0))
    p_draw = float(np.mean(diff == 0))
    p_away = float(np.mean(diff < 0))
    p_btts = float(np.mean((h_goals > 0) & (a_goals > 0)))
    p_o15 = float(np.mean(total > 1.5))
    p_o25 = float(np.mean(total > 2.5))
    p_o35 = float(np.mean(total > 3.5))
    p_cs = float(np.mean((h_goals == 0) | (a_goals == 0)))
    mean_total = float(np.mean(total))
    var_total = float(np.var(total))

    # Most likely score
    from collections import Counter
    score_counts = Counter(zip(h_goals.tolist(), a_goals.tolist()))
    ml_score = score_counts.most_common(1)[0][0] if score_counts else (1, 1)

    return {
        "mc_p_home": p_home, "mc_p_draw": p_draw, "mc_p_away": p_away,
        "mc_btts": p_btts, "mc_o15": p_o15, "mc_o25": p_o25, "mc_o35": p_o35,
        "mc_cs": p_cs, "mc_mean_total": mean_total, "mc_var_total": var_total,
        "mc_n_sims": n_accepted,
        "mc_ml_hg": ml_score[0], "mc_ml_ag": ml_score[1],
    }


# ═══════════════════════════════════════════════════════════════════
# 5. LOGIT-SPACE PROBABILITY MANIPULATION
# ═══════════════════════════════════════════════════════════════════
def logit(p: float, eps: float = 1e-8) -> float:
    """Logit transform: ln(p / (1-p))."""
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def inv_logit(x: float) -> float:
    """Inverse logit (sigmoid): 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def logit_space_delta(p_model: float, p_market: float, eps: float = 1e-8) -> float:
    """Compute model-vs-market delta in logit space.

    Δ = logit(p_model) - logit(p_market)

    Positive Δ means model thinks outcome is more likely than market,
    indicating potential value.
    """
    return logit(p_model, eps) - logit(p_market, eps)


def remove_overround(odds: Sequence[float]) -> tuple[float, ...]:
    """Convert decimal odds to fair probabilities by removing overround.

    Uses multiplicative method: p_i = (1/odds_i) / Σ(1/odds_j)

    Args:
        odds: Sequence of decimal odds (e.g., [2.5, 3.3, 2.8])

    Returns:
        Tuple of fair probabilities summing to 1.0
    """
    implied = [1.0 / max(o, 1.01) for o in odds]
    total = sum(implied)
    if total <= 0:
        return tuple(1.0 / len(odds) for _ in odds)
    return tuple(p / total for p in implied)


def odds_entropy(probs: Sequence[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution: H(p) = -Σ p_i ln p_i.

    Higher entropy → more uncertain / competitive match.
    Lower entropy → one-sided (clear favourite).
    """
    return -sum(max(p, eps) * math.log(max(p, eps)) for p in probs)


def odds_dispersion(odds_list: list[Sequence[float]]) -> float:
    """Compute dispersion across multiple bookmakers' odds.

    Higher dispersion → bookmakers disagree → more uncertainty.
    Uses standard deviation of implied probabilities across books.
    """
    if len(odds_list) < 2:
        return 0.0
    probs_list = [remove_overround(odds) for odds in odds_list]
    h_probs = [p[0] for p in probs_list]
    return float(np.std(h_probs))


# ═══════════════════════════════════════════════════════════════════
# 6. EWMA WITH ADAPTIVE TIME-DECAY
# ═══════════════════════════════════════════════════════════════════
def adaptive_ewma(
    values: list[float],
    timestamps: list[float] | None = None,
    tau_days: float = 30.0,
    default_alpha: float = 0.1,
) -> float:
    """EWMA with time-adaptive smoothing factor.

    If timestamps provided:
        α_t = 1 - exp(-Δt / τ)
        s_t = α_t · x_t + (1 - α_t) · s_{t-1}

    Otherwise uses fixed α = default_alpha.
    """
    if not values:
        return 0.0

    if timestamps and len(timestamps) == len(values):
        s = values[0]
        for i in range(1, len(values)):
            dt = max(timestamps[i] - timestamps[i - 1], 0.1)
            alpha = 1.0 - math.exp(-dt / max(tau_days, 0.1))
            s = alpha * values[i] + (1 - alpha) * s
        return s
    else:
        alpha = default_alpha
        s = values[0]
        for v in values[1:]:
            s = alpha * v + (1 - alpha) * s
        return s


# ═══════════════════════════════════════════════════════════════════
# 7. NONLINEAR STRENGTH TRANSFORMS
# ═══════════════════════════════════════════════════════════════════
def tanh_transform(diff: float, k: float = 400.0) -> float:
    """Squash a strength difference through tanh for bounded feature.

    tanh(diff/k) maps large differences to ±1 while preserving
    sensitivity around 0.
    """
    return math.tanh(diff / max(k, 1.0))


def log_transform(diff: float) -> float:
    """Signed log transform: sign(x) · log(1 + |x|)."""
    return math.copysign(math.log1p(abs(diff)), diff)


def rank_normalize(value: float, values: Sequence[float]) -> float:
    """Percentile rank normalization: maps value to [0, 1] within distribution."""
    if not values or len(values) < 2:
        return 0.5
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    rank = sum(1 for v in sorted_vals if v <= value)
    return rank / n


# ═══════════════════════════════════════════════════════════════════
# 8. KL DIVERGENCE
# ═══════════════════════════════════════════════════════════════════
def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """KL divergence D_KL(P || Q) = Σ p_i · ln(p_i / q_i).

    Measures how much expert P diverges from ensemble Q.
    Higher values → expert disagrees more strongly with consensus.
    """
    return sum(
        max(pi, eps) * math.log(max(pi, eps) / max(qi, eps))
        for pi, qi in zip(p, q)
    )


# ═══════════════════════════════════════════════════════════════════
# 9. SCHEDULE DIFFICULTY
# ═══════════════════════════════════════════════════════════════════
def schedule_difficulty(
    opponent_strengths: list[float],
    window: int = 5,
) -> float:
    """Rolling average opponent strength over last N matches.

    Higher values → team has faced stronger opposition.
    Useful for contextualizing form (e.g., poor form vs strong schedule).
    """
    if not opponent_strengths:
        return 1500.0  # neutral
    recent = opponent_strengths[-window:]
    return float(np.mean(recent))


# ═══════════════════════════════════════════════════════════════════
# 10. LINEUP / SQUAD STABILITY
# ═══════════════════════════════════════════════════════════════════
def jaccard_overlap(set_a: set, set_b: set) -> float:
    """Jaccard similarity J = |A ∩ B| / |A ∪ B|.

    Used for lineup stability: how similar are consecutive starting XIs.
    """
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / max(union, 1)


def availability_index(
    unavailable_players: list[dict],
    total_minutes: float = 3420.0,  # 38 * 90 minutes in a season
) -> float:
    """Weighted availability index.

    Σ (minutes_share × importance_weight) for unavailable players.
    Higher values → more impactful absences.

    Each player dict should have: {'minutes_pct': float, 'importance': float}
    """
    if not unavailable_players:
        return 0.0
    return sum(
        p.get("minutes_pct", 0.0) * p.get("importance", 1.0)
        for p in unavailable_players
    )
