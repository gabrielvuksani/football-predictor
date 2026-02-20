"""Advanced mathematical functions for football prediction.

Implements research-backed methods:
- Empirical Bayes shrinkage (Beta-Binomial)
- Dixon-Coles low-score correction (τ adjustment)
- Skellam distribution for goal-difference modelling
- Monte Carlo scoreline simulation
- Logit-space probability manipulation
- EWMA with adaptive time-decay
- Nonlinear strength transforms
- Bivariate Poisson joint distribution (Karlis & Ntzoufras 2003)
- Frank Copula for full-scoreline dependency modelling
- Ranked Probability Score (RPS) for ordered outcomes
- Conway-Maxwell-Poisson for over/under-dispersed counts
- Bradley-Terry pairwise comparison model
- Platt scaling (temperature calibration)

References:
    Dixon & Coles (1997) "Modelling Association Football Scores and
        Inefficiencies in the Football Betting Market"
    Karlis & Ntzoufras (2003) "Analysis of sports data by using
        bivariate Poisson models"
    Gelman et al. "Bayesian Data Analysis" — empirical Bayes shrinkage
    Joe (1997) "Multivariate Models and Dependence Concepts" — copulas
    Epstein (1969) "A Scoring System for Probability Forecasts of
        Ranked Categories" — Ranked Probability Score
    Conway & Maxwell (1962) "A queuing model with state dependent
        service rates" — COM-Poisson distribution
    Bradley & Terry (1952) "Rank Analysis of Incomplete Block Designs"
    Platt (1999) "Probabilistic Outputs for Support Vector Machines"
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import poisson as poisson_dist
from scipy.special import comb as scipy_comb, gammaln


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

def estimate_rho_mle(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambdas_h: np.ndarray,
    lambdas_a: np.ndarray,
    bounds: tuple[float, float] = (-0.4, 0.0),
) -> float:
    """Estimate Dixon-Coles ρ via maximum likelihood for a dataset.

    Given observed scorelines (hg, ag) and their corresponding Poisson
    expected goals (λ_h, λ_a), find the ρ that maximises the log-likelihood
    of the τ-adjusted bivariate Poisson model.

    Uses scipy.optimize.minimize_scalar (bounded Brent) which is robust
    for this single-parameter problem.

    Args:
        home_goals: Array of observed home goals (N,)
        away_goals: Array of observed away goals (N,)
        lambdas_h: Array of expected home goals (N,)
        lambdas_a: Array of expected away goals (N,)
        bounds: (lower, upper) bounds for ρ search

    Returns:
        MLE estimate of ρ (typically between -0.25 and -0.03)
    """
    from scipy.optimize import minimize_scalar

    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)
    lh = np.asarray(lambdas_h, dtype=float)
    la = np.asarray(lambdas_a, dtype=float)

    # Only apply τ where hg ≤ 1 and ag ≤ 1 (low scores)
    low_mask = (hg <= 1) & (ag <= 1)
    hg_low = hg[low_mask]
    ag_low = ag[low_mask]
    lh_low = lh[low_mask]
    la_low = la[low_mask]

    if len(hg_low) < 30:
        # Not enough low-score data; fall back to -0.13
        return -0.13

    def neg_log_lik(rho: float) -> float:
        """Negative log-likelihood of τ factors (we only need the low-score part)."""
        taus = np.array([
            dixon_coles_tau(int(h), int(a), float(lh_i), float(la_i), rho)
            for h, a, lh_i, la_i in zip(hg_low, ag_low, lh_low, la_low)
        ])
        # Clamp to avoid log(0)
        taus = np.clip(taus, 1e-12, None)
        return -np.sum(np.log(taus))

    result = minimize_scalar(neg_log_lik, bounds=bounds, method="bounded")
    rho_hat = float(result.x)

    # Safety clamp
    rho_hat = max(bounds[0], min(bounds[1], rho_hat))
    return rho_hat


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
    """Composite schedule difficulty index over last N matches.

    Combines multiple signals:
    - Weighted mean (recent opponents count more): exponential decay weights
    - Variance (a mixed schedule of strong+weak ≠ uniformly average)
    - Max opponent (one very strong opponent raises difficulty)

    Output: composite score centered around 1500 (neutral Elo).
    Higher = harder schedule. Scale: ~1200 (easy) to ~1800 (brutal).
    """
    if not opponent_strengths:
        return 1500.0  # neutral
    recent = opponent_strengths[-window:]
    n = len(recent)
    if n == 0:
        return 1500.0

    # Exponential decay weights: most recent match gets highest weight
    weights = np.array([0.7 ** (n - 1 - i) for i in range(n)])
    weights /= weights.sum()

    weighted_mean = float(np.dot(weights, recent))
    variance = float(np.var(recent)) if n > 1 else 0.0
    max_opp = float(max(recent))

    # Composite: 70% weighted mean + 15% max-boosted + 15% variance penalty
    # Variance penalty: high variance → slightly harder (unpredictability)
    variance_factor = min(variance / 40000.0, 0.15)  # normalized, capped
    composite = 0.70 * weighted_mean + 0.15 * max_opp + 0.15 * (weighted_mean + variance_factor * 1500)

    return float(composite)


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


# ═══════════════════════════════════════════════════════════════════
# 11. BIVARIATE POISSON (Karlis & Ntzoufras 2003)
# ═══════════════════════════════════════════════════════════════════
def bivariate_poisson_pmf(
    x: int, y: int,
    lambda_1: float, lambda_2: float, lambda_3: float,
) -> float:
    """Bivariate Poisson PMF: P(X=x, Y=y).

    The bivariate Poisson models (X, Y) where:
        X = X₁ + X₃,  Y = X₂ + X₃
        X₁ ~ Poisson(λ₁),  X₂ ~ Poisson(λ₂),  X₃ ~ Poisson(λ₃)

    So Cov(X, Y) = λ₃ ≥ 0, with marginals:
        X ~ Poisson(λ₁ + λ₃),  Y ~ Poisson(λ₂ + λ₃)

    This models positive correlation (teams scoring in open games),
    more principled than Dixon-Coles τ which only adjusts 4 scorelines.

    PMF(x, y | λ₁, λ₂, λ₃) = e^{-(λ₁+λ₂+λ₃)} ·
        (λ₁^x / x!) · (λ₂^y / y!) ·
        Σ_{k=0}^{min(x,y)} C(x,k) · C(y,k) · k! · (λ₃/(λ₁·λ₂))^k

    Args:
        x: Home goals scored (non-negative integer)
        y: Away goals scored (non-negative integer)
        lambda_1: Home-specific intensity (marginal = λ₁ + λ₃)
        lambda_2: Away-specific intensity (marginal = λ₂ + λ₃)
        lambda_3: Shared (covariance) intensity ≥ 0

    Returns:
        P(X=x, Y=y) under bivariate Poisson model
    """
    if lambda_1 <= 0 or lambda_2 <= 0 or lambda_3 < 0:
        # Fall back to independent Poisson
        return float(
            poisson_dist.pmf(x, max(lambda_1 + lambda_3, 1e-6))
            * poisson_dist.pmf(y, max(lambda_2 + lambda_3, 1e-6))
        )

    # Log-space computation for numerical stability
    log_base = (
        -(lambda_1 + lambda_2 + lambda_3)
        + x * math.log(lambda_1) - gammaln(x + 1)
        + y * math.log(lambda_2) - gammaln(y + 1)
    )

    k_max = min(x, y)
    if lambda_3 < 1e-12 or k_max == 0:
        return float(math.exp(log_base))

    # Sum over k in log-space for stability
    log_ratio = math.log(lambda_3) - math.log(lambda_1) - math.log(lambda_2)
    log_terms = []
    for k in range(k_max + 1):
        log_term = (
            gammaln(x + 1) - gammaln(x - k + 1)  # ln C(x,k) · k!
            + gammaln(y + 1) - gammaln(y - k + 1)
            - gammaln(k + 1)
            + k * log_ratio
        )
        log_terms.append(log_term)

    # log-sum-exp for stable summation
    max_log = max(log_terms)
    log_sum = max_log + math.log(sum(math.exp(lt - max_log) for lt in log_terms))

    return float(math.exp(log_base + log_sum))


def build_bivariate_poisson_matrix(
    lambda_h: float,
    lambda_a: float,
    lambda_3: float = 0.1,
    max_goals: int = 8,
) -> np.ndarray:
    """Build score matrix from bivariate Poisson model.

    Marginal expected goals: E[H] = λ_h, E[A] = λ_a
    So: λ₁ = λ_h - λ₃,  λ₂ = λ_a - λ₃  (must be > 0)

    If λ₃ > min(λ_h, λ_a), it's automatically clamped.

    Args:
        lambda_h: Home expected goals (marginal)
        lambda_a: Away expected goals (marginal)
        lambda_3: Covariance parameter (≥ 0)
        max_goals: Maximum goals per team

    Returns:
        (max_goals+1) × (max_goals+1) probability matrix
    """
    # Clamp λ₃ so marginal-specific intensities stay positive
    lambda_3 = max(0.0, min(lambda_3, min(lambda_h, lambda_a) - 0.01))
    if lambda_3 < 0:
        lambda_3 = 0.0

    l1 = lambda_h - lambda_3
    l2 = lambda_a - lambda_3

    n = max_goals + 1
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = bivariate_poisson_pmf(i, j, l1, l2, lambda_3)

    # Renormalize
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


# ═══════════════════════════════════════════════════════════════════
# 12. FRANK COPULA SCORE MATRIX
# ═══════════════════════════════════════════════════════════════════
def _frank_copula(u: float, v: float, theta: float) -> float:
    """Frank copula C(u, v; θ).

    C(u,v;θ) = -1/θ · ln(1 + (e^{-θu} - 1)(e^{-θv} - 1) / (e^{-θ} - 1))

    Properties:
        θ → 0:  independence (Π copula)
        θ > 0:  positive dependence
        θ < 0:  negative dependence

    Unlike Dixon-Coles τ (which only adjusts 4 low-scoring cells),
    the Frank copula models dependency across ALL scorelines.

    Args:
        u: Marginal CDF value for home goals, in (0, 1)
        v: Marginal CDF value for away goals, in (0, 1)
        theta: Dependency parameter

    Returns:
        Joint CDF C(u, v)
    """
    if abs(theta) < 1e-8:
        return u * v  # independence
    e_t = math.exp(-theta)
    e_tu = math.exp(-theta * u)
    e_tv = math.exp(-theta * v)
    numer = (e_tu - 1.0) * (e_tv - 1.0)
    denom = e_t - 1.0
    if abs(denom) < 1e-15:
        return u * v
    return -1.0 / theta * math.log(max(1e-15, 1.0 + numer / denom))


def build_copula_score_matrix(
    lambda_h: float,
    lambda_a: float,
    theta: float = -2.0,
    max_goals: int = 8,
) -> np.ndarray:
    """Build score matrix using Frank copula for full dependency structure.

    Rather than adjusting only 4 cells (Dixon-Coles), the copula models
    dependency across ALL scorelines by coupling the marginal Poisson CDFs.

    The joint PMF is obtained via inclusion-exclusion on the copula:
        P(H=i, A=j) = C(F_H(i), F_A(j)) - C(F_H(i-1), F_A(j))
                     - C(F_H(i), F_A(j-1)) + C(F_H(i-1), F_A(j-1))

    where F_H, F_A are the Poisson CDFs.

    Typical θ values for football:
        θ ≈ -2 to -1: negative dependency (draws → 0-0 more likely)
        θ ≈ 0: independence
        θ ≈ 1 to 3: positive dependency (open games)

    Args:
        lambda_h: Home expected goals
        lambda_a: Away expected goals
        theta: Frank copula parameter (< 0 = negative dep, like DC)
        max_goals: Maximum goals per team

    Returns:
        (max_goals+1) × (max_goals+1) probability matrix
    """
    n = max_goals + 1
    # Poisson CDFs — include -1 sentinel for inclusion-exclusion
    cdf_h = np.zeros(n + 1)
    cdf_a = np.zeros(n + 1)
    cdf_h[0] = 0.0  # F(-1) = 0
    cdf_a[0] = 0.0
    for k in range(n):
        cdf_h[k + 1] = float(poisson_dist.cdf(k, lambda_h))
        cdf_a[k + 1] = float(poisson_dist.cdf(k, lambda_a))

    # Joint PMF via inclusion-exclusion
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c11 = _frank_copula(cdf_h[i + 1], cdf_a[j + 1], theta)
            c10 = _frank_copula(cdf_h[i],     cdf_a[j + 1], theta)
            c01 = _frank_copula(cdf_h[i + 1], cdf_a[j],     theta)
            c00 = _frank_copula(cdf_h[i],     cdf_a[j],     theta)
            mat[i, j] = max(0.0, c11 - c10 - c01 + c00)

    # Renormalize
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


# ═══════════════════════════════════════════════════════════════════
# 13. RANKED PROBABILITY SCORE (RPS)
# ═══════════════════════════════════════════════════════════════════
def ranked_probability_score(
    probs: Sequence[float],
    outcome_idx: int,
) -> float:
    """Ranked Probability Score for ordered categorical outcomes.

    The RPS is the proper scoring rule for ordered categories.
    For football: categories are [Home, Draw, Away] — ordered by goal diff.

    RPS = 1/(K-1) · Σ_{k=1}^{K-1} (CDF_pred(k) - CDF_actual(k))²

    where CDF_pred(k) = Σ_{j=1}^{k} p_j  and CDF_actual is step function.

    Properties:
        - RPS ∈ [0, 1], lower is better
        - Unlike logloss, RPS penalizes predictions that are "close"
          less than predictions that are "far" (e.g., predicting Away
          when Home wins is worse than predicting Draw when Home wins)
        - This is critical for football where the ordering matters

    Reference: Epstein (1969), Constantinou & Fenton (2012)

    Args:
        probs: Predicted probabilities [p_home, p_draw, p_away]
        outcome_idx: Actual outcome index (0=Home, 1=Draw, 2=Away)

    Returns:
        RPS score (lower = better prediction)
    """
    k = len(probs)
    if k < 2:
        return 0.0

    cdf_pred = 0.0
    rps = 0.0
    for j in range(k - 1):
        cdf_pred += probs[j]
        cdf_actual = 1.0 if j >= outcome_idx else 0.0
        rps += (cdf_pred - cdf_actual) ** 2

    return rps / (k - 1)


def rps_from_result(
    probs: Sequence[float],
    home_goals: int,
    away_goals: int,
) -> float:
    """Convenience wrapper: compute RPS from match result.

    Maps result to outcome_idx: home win → 0, draw → 1, away win → 2.
    """
    if home_goals > away_goals:
        idx = 0
    elif home_goals == away_goals:
        idx = 1
    else:
        idx = 2
    return ranked_probability_score(probs, idx)


# ═══════════════════════════════════════════════════════════════════
# 14. CONWAY-MAXWELL-POISSON (COM-POISSON)
# ═══════════════════════════════════════════════════════════════════
def _com_poisson_normalizing(
    lam: float, nu: float, max_k: int = 30,
) -> float:
    """Compute the COM-Poisson normalizing constant Z(λ, ν).

    Z(λ, ν) = Σ_{k=0}^{∞} λ^k / (k!)^ν

    The COM-Poisson (Conway-Maxwell-Poisson) distribution generalizes
    Poisson with a dispersion parameter ν:
        ν = 1 → standard Poisson
        ν < 1 → over-dispersed (more variance than Poisson)
        ν > 1 → under-dispersed (less variance than Poisson)

    Football goals are often slightly over-dispersed (variance > mean),
    which standard Poisson cannot capture. COM-Poisson handles this.
    """
    log_z_terms = []
    for k in range(max_k + 1):
        log_term = k * math.log(max(lam, 1e-15)) - nu * gammaln(k + 1)
        log_z_terms.append(log_term)
    max_log = max(log_z_terms)
    return max_log + math.log(
        sum(math.exp(lt - max_log) for lt in log_z_terms)
    )


def com_poisson_pmf(
    k: int, lam: float, nu: float = 1.0, max_terms: int = 30,
) -> float:
    """Conway-Maxwell-Poisson PMF.

    P(X=k) = λ^k / ((k!)^ν · Z(λ,ν))

    When goals show over-dispersion (σ² > μ, common in football),
    standard Poisson underestimates extreme scorelines.
    COM-Poisson with ν < 1 corrects this.

    Typical football values:
        ν ≈ 0.85-0.95 for leagues with high variance (e.g., BL1)
        ν ≈ 1.0 for "Poisson-like" leagues
        ν ≈ 1.05-1.15 for defensive leagues

    Args:
        k: Number of goals (non-negative integer)
        lam: Rate parameter (λ > 0)
        nu: Dispersion parameter (ν > 0, 1=Poisson)
        max_terms: Terms for normalizing constant

    Returns:
        P(X=k) under COM-Poisson
    """
    log_z = _com_poisson_normalizing(lam, nu, max_terms)
    log_pmf = k * math.log(max(lam, 1e-15)) - nu * gammaln(k + 1) - log_z
    return float(math.exp(log_pmf))


def build_com_poisson_matrix(
    lambda_h: float,
    lambda_a: float,
    nu_h: float = 0.92,
    nu_a: float = 0.92,
    max_goals: int = 8,
) -> np.ndarray:
    """Build score matrix using independent COM-Poisson marginals.

    Extends standard Poisson by allowing each team's goal distribution
    to be over-dispersed (ν < 1) or under-dispersed (ν > 1).

    This is particularly useful for:
    - High-variance leagues (Bundesliga) where 5-0 and 0-0 are common
    - Cup matches with atypical variance patterns

    Args:
        lambda_h: Home rate parameter
        lambda_a: Away rate parameter
        nu_h: Home dispersion (< 1 = over-dispersed)
        nu_a: Away dispersion (< 1 = over-dispersed)
        max_goals: Maximum goals per team

    Returns:
        (max_goals+1) × (max_goals+1) probability matrix
    """
    n = max_goals + 1
    pmf_h = np.array([com_poisson_pmf(k, lambda_h, nu_h) for k in range(n)])
    pmf_a = np.array([com_poisson_pmf(k, lambda_a, nu_a) for k in range(n)])

    mat = np.outer(pmf_h, pmf_a)

    # Renormalize
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


# ═══════════════════════════════════════════════════════════════════
# 15. BRADLEY-TERRY WITH HOME ADVANTAGE
# ═══════════════════════════════════════════════════════════════════
def bradley_terry_probs(
    strength_h: float,
    strength_a: float,
    home_adv: float = 0.3,
    draw_factor: float = 0.25,
) -> tuple[float, float, float]:
    """Bradley-Terry model with home advantage and draw extension.

    The Bradley-Terry (1952) model gives:
        P(i beats j) = π_i / (π_i + π_j)

    Extended with home advantage (Agresti 2002):
        P(Home wins) = θ·π_h / (θ·π_h + π_a)

    Extended with draws (Davidson 1970):
        P(Draw) = ν·√(π_h·π_a) / (θ·π_h + π_a + ν·√(π_h·π_a))

    where:
        π_i = exp(strength_i) is team i's strength
        θ = exp(home_adv) is the home advantage factor
        ν = draw_factor controls draw probability

    This is algebraically different from Elo and can capture
    different aspects of team comparison.

    Args:
        strength_h: Home team log-strength
        strength_a: Away team log-strength
        home_adv: Log home advantage (θ = e^{home_adv})
        draw_factor: Draw intensity (ν ≥ 0)

    Returns:
        (p_home, p_draw, p_away) summing to 1.0
    """
    pi_h = math.exp(strength_h)
    pi_a = math.exp(strength_a)
    theta = math.exp(home_adv)

    numerator_h = theta * pi_h
    numerator_a = pi_a
    numerator_d = draw_factor * math.sqrt(pi_h * pi_a)

    total = numerator_h + numerator_a + numerator_d
    if total <= 0:
        return (1.0 / 3, 1.0 / 3, 1.0 / 3)

    p_h = numerator_h / total
    p_d = numerator_d / total
    p_a = numerator_a / total

    return (p_h, p_d, p_a)


# ═══════════════════════════════════════════════════════════════════
# 16. PLATT SCALING (TEMPERATURE CALIBRATION)
# ═══════════════════════════════════════════════════════════════════
def platt_scale(
    probs: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Platt/temperature scaling for probability calibration.

    Applies temperature scaling in logit-space:
        p_calibrated = softmax(logit(p) / T)

    Temperature effects:
        T = 1.0: no change (identity)
        T > 1.0: softer probabilities (less confident)
        T < 1.0: sharper probabilities (more confident)

    This is a post-hoc calibration method that preserves the ranking
    of predicted classes while adjusting confidence levels.

    Guo et al. (2017) "On Calibration of Modern Neural Networks" showed
    temperature scaling is highly effective for multi-class calibration.

    Args:
        probs: Array of shape (N, K) or (K,) — probabilities
        temperature: Temperature parameter T > 0

    Returns:
        Calibrated probabilities, same shape as input
    """
    eps = 1e-8
    single = probs.ndim == 1
    if single:
        probs = probs[np.newaxis, :]

    # Log-probability space (correct for multi-class temperature scaling)
    p_clipped = np.clip(probs, eps, 1.0)
    log_probs = np.log(p_clipped)

    # Scale by temperature
    scaled_log_probs = log_probs / max(temperature, eps)

    # Softmax
    shifted = scaled_log_probs - scaled_log_probs.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    result = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return result[0] if single else result


def find_optimal_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    t_range: tuple[float, float] = (0.5, 3.0),
) -> float:
    """Find optimal temperature T that minimizes NLL on validation set.

    Uses scipy bounded optimization to find the T that gives
    best-calibrated probabilities.

    Args:
        probs: Predicted probabilities (N, K)
        labels: True class indices (N,) — 0, 1, or 2
        t_range: Search bounds for temperature

    Returns:
        Optimal temperature value
    """
    from scipy.optimize import minimize_scalar

    def nll(t: float) -> float:
        scaled = platt_scale(probs, temperature=t)
        eps = 1e-12
        ll = 0.0
        for i in range(len(labels)):
            ll -= math.log(max(scaled[i, int(labels[i])], eps))
        return ll

    result = minimize_scalar(nll, bounds=t_range, method="bounded")
    return float(result.x)


# ═══════════════════════════════════════════════════════════════════
# 17. OUTCOME-PROBABILITY EXTRACTORS FROM SCORE MATRICES
# ═══════════════════════════════════════════════════════════════════
def extract_match_probs(mat: np.ndarray) -> dict[str, float]:
    """Extract comprehensive match probabilities from ANY score matrix.

    Works with: Dixon-Coles, bivariate Poisson, Frank copula, COM-Poisson.

    Returns dict with:
        1X2 probabilities, BTTS, Over/Under lines, clean sheet,
        expected goals, most likely scoreline, and dispersion metrics.
    """
    n = mat.shape[0]
    goals_range = np.arange(n)

    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())

    # BTTS (both teams to score)
    p_btts = float(mat[1:, 1:].sum())

    # Over/Under lines
    total_goals = np.add.outer(goals_range, goals_range)
    p_o15 = float(mat[total_goals >= 2].sum())
    p_o25 = float(mat[total_goals >= 3].sum())
    p_o35 = float(mat[total_goals >= 4].sum())
    p_o45 = float(mat[total_goals >= 5].sum())

    # Clean sheet
    p_cs_home = float(mat[:, 0].sum())  # away scores 0
    p_cs_away = float(mat[0, :].sum())  # home scores 0

    # Expected goals from the matrix
    eg_h = float(np.sum(goals_range[:, None] * mat))
    eg_a = float(np.sum(goals_range[None, :] * mat))

    # Variance of goals (for dispersion detection)
    var_h = float(np.sum(goals_range[:, None] ** 2 * mat)) - eg_h ** 2
    var_a = float(np.sum(goals_range[None, :] ** 2 * mat)) - eg_a ** 2

    # Dispersion index (variance / mean) — > 1 = over-dispersed
    disp_h = var_h / max(eg_h, 1e-6)
    disp_a = var_a / max(eg_a, 1e-6)

    # Most likely scoreline
    flat_idx = int(np.argmax(mat))
    ml_h, ml_a = divmod(flat_idx, n)

    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "p_btts": p_btts,
        "p_o15": p_o15, "p_o25": p_o25, "p_o35": p_o35, "p_o45": p_o45,
        "p_cs_home": p_cs_home, "p_cs_away": p_cs_away,
        "eg_home": eg_h, "eg_away": eg_a,
        "var_home": var_h, "var_away": var_a,
        "disp_home": disp_h, "disp_away": disp_a,
        "ml_score_h": ml_h, "ml_score_a": ml_a,
    }


# ═══════════════════════════════════════════════════════════════════
# 18. ENSEMBLE AGREEMENT METRICS
# ═══════════════════════════════════════════════════════════════════
def jensen_shannon_divergence(
    p: Sequence[float], q: Sequence[float], eps: float = 1e-12,
) -> float:
    """Jensen-Shannon divergence — symmetric, bounded version of KL.

    JSD(P || Q) = ½ KL(P || M) + ½ KL(Q || M),  where M = ½(P + Q)

    Properties:
        JSD ∈ [0, ln(2)] ≈ [0, 0.693]
        JSD is symmetric: JSD(P||Q) = JSD(Q||P)
        √JSD is a proper metric (satisfies triangle inequality)

    Better than KL divergence for measuring expert disagreement because:
    1. Symmetric (expert A vs B = expert B vs A)
    2. Always finite (no division-by-zero risk)
    3. Bounded (easier to interpret as a feature)

    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Smoothing constant

    Returns:
        JSD value in [0, ln(2)]
    """
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    kl_pm = sum(
        max(pi, eps) * math.log(max(pi, eps) / max(mi, eps))
        for pi, mi in zip(p, m)
    )
    kl_qm = sum(
        max(qi, eps) * math.log(max(qi, eps) / max(mi, eps))
        for qi, mi in zip(q, m)
    )
    return 0.5 * kl_pm + 0.5 * kl_qm


def multi_expert_jsd(
    expert_probs: list[Sequence[float]], eps: float = 1e-12,
) -> float:
    """Generalized JSD across N expert distributions.

    JSD(P₁, ..., Pₙ) = H(M) - 1/n Σ H(Pᵢ)

    where M = 1/n Σ Pᵢ  and H is Shannon entropy.

    This measures total disagreement across all experts simultaneously,
    rather than pairwise. Higher values → more confusion / uncertainty.

    Args:
        expert_probs: List of N probability distributions
        eps: Smoothing constant

    Returns:
        Multi-expert JSD value
    """
    if not expert_probs:
        return 0.0
    n = len(expert_probs)
    k = len(expert_probs[0])

    # Mixture distribution M
    m = [sum(expert_probs[i][j] for i in range(n)) / n for j in range(k)]

    # H(M) — entropy of mixture
    h_m = -sum(max(mi, eps) * math.log(max(mi, eps)) for mi in m)

    # Average entropy of individual experts
    avg_h = 0.0
    for p in expert_probs:
        avg_h -= sum(max(pi, eps) * math.log(max(pi, eps)) for pi in p) / n

    return max(0.0, h_m - avg_h)
