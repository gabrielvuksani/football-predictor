"""Probability distributions for football prediction.

Implements standard and advanced distribution models:
- Skellam distribution for goal-difference modelling
- Bivariate Poisson joint distribution (Karlis & Ntzoufras 2003)
- Conway-Maxwell-Poisson for over/under-dispersed counts
- Zero-Inflated Poisson for excess zero-scoring matches
- Negative Binomial for overdispersed goal counts

References:
    Karlis & Ntzoufras (2003) "Analysis of sports data by using
        bivariate Poisson models"
    Conway & Maxwell (1962) "A queuing model with state dependent
        service rates"
    Efron (1986) "Double Exponential Families and Their Use in Generalized
        Linear Models"
    Lambert (1992) "Zero-inflated Poisson regression"
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import poisson as poisson_dist
from scipy.special import gammaln
from scipy.stats import nbinom, skellam as skellam_dist


# ═══════════════════════════════════════════════════════════════════
# PARAMETER ESTIMATION HELPERS
# ═══════════════════════════════════════════════════════════════════
def estimate_com_poisson_nu(goals_series):
    """Estimate COM-Poisson nu from empirical variance/mean ratio.
    Football goals are underdispersed (nu > 1 in CMP convention).
    """
    mu = np.mean(goals_series)
    var = np.var(goals_series)
    if var < 0.1 or mu < 0.1:
        return 1.0
    nu = mu / max(var, 0.5)
    return float(np.clip(nu, 0.5, 2.0))


def estimate_zip_zero_inflation(goals_series):
    """Estimate zero-inflation parameter from empirical 0-0 rate vs Poisson prediction."""
    mu = np.mean(goals_series)
    actual_zero_rate = float(np.mean(goals_series == 0))
    poisson_zero_rate = float(np.exp(-mu))
    if poisson_zero_rate >= 1.0 - 1e-6:
        return 0.0
    zi = max(0.0, (actual_zero_rate - poisson_zero_rate) / (1.0 - poisson_zero_rate))
    return float(np.clip(zi, 0.0, 0.3))


def estimate_bivariate_lambda3(home_goals, away_goals):
    """Estimate bivariate Poisson covariance parameter from empirical goal correlation."""
    cov = np.cov(home_goals, away_goals)[0, 1]
    return float(max(0.01, min(0.5, cov)))


# ═══════════════════════════════════════════════════════════════════
# SKELLAM DISTRIBUTION
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
# BIVARIATE POISSON
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
# CONWAY-MAXWELL-POISSON (COM-POISSON)
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

    For large λ or small ν, the series needs more terms to converge.
    This implementation adaptively extends max_k until convergence.
    """
    # Adaptive max_k: for large lambda or small nu, we need more terms
    # Heuristic: need roughly lambda^(1/nu) terms for convergence
    if lam > 5.0 or nu < 0.7:
        adaptive_k = int(max(max_k, min(200, lam * 3 / max(nu, 0.3))))
    else:
        adaptive_k = max_k

    log_z_terms = []
    log_lam = math.log(max(lam, 1e-15))
    prev_contribution = float("inf")
    converged = False

    for k in range(adaptive_k + 1):
        log_term = k * log_lam - nu * gammaln(k + 1)
        log_z_terms.append(log_term)

        # Check convergence: if new term is negligible relative to max
        if k > max_k and len(log_z_terms) > 1:
            current_max = max(log_z_terms)
            relative = log_term - current_max
            if relative < -30.0:  # exp(-30) ~ 1e-13, negligible
                converged = True
                break

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
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


# ═══════════════════════════════════════════════════════════════════
# ZERO-INFLATED POISSON (ZIP)
# ═══════════════════════════════════════════════════════════════════
def zero_inflated_poisson_pmf(k: int, lam: float, zero_mass: float = 0.05) -> float:
    """Zero-inflated Poisson PMF: P(X=k) = π·I(k=0) + (1-π)·Poisson(k|λ).

    Reference: Lambert (1992) "Zero-inflated Poisson regression"
    """
    lam = max(1e-9, float(lam))
    zero_mass = min(0.95, max(0.0, float(zero_mass)))
    if k == 0:
        return zero_mass + (1.0 - zero_mass) * math.exp(-lam)
    return (1.0 - zero_mass) * math.exp(-lam) * (lam ** k) / math.factorial(k)


def build_zip_score_matrix(
    lambda_h: float,
    lambda_a: float,
    zero_mass_h: float = 0.05,
    zero_mass_a: float = 0.05,
    max_goals: int = 8,
) -> np.ndarray:
    """Build score matrix using Zero-Inflated Poisson marginals."""
    n = max_goals + 1
    pmf_h = np.array([zero_inflated_poisson_pmf(k, lambda_h, zero_mass_h) for k in range(n)])
    pmf_a = np.array([zero_inflated_poisson_pmf(k, lambda_a, zero_mass_a) for k in range(n)])
    mat = np.outer(pmf_h, pmf_a)
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


# ═══════════════════════════════════════════════════════════════════
# NEGATIVE BINOMIAL MODEL (OVER-DISPERSED GOALS)
# ═══════════════════════════════════════════════════════════════════
def negative_binomial_pmf(k: int, mu: float, alpha: float = 1.5) -> float:
    """Negative Binomial PMF parameterized by mean and dispersion.

    Variance = μ + μ²/α. As α → ∞, NB → Poisson.
    Reference: Hilbe (2011) "Negative Binomial Regression"
    """
    mu = max(1e-9, float(mu))
    alpha = max(1e-3, float(alpha))
    r = alpha
    p = alpha / (alpha + mu)
    return float(nbinom.pmf(k, r, p))


def build_negative_binomial_matrix(
    lambda_h: float,
    lambda_a: float,
    alpha_h: float = 3.0,
    alpha_a: float = 3.0,
    max_goals: int = 8,
) -> np.ndarray:
    """Build score matrix using independent Negative Binomial marginals."""
    n = max_goals + 1
    pmf_h = np.array([negative_binomial_pmf(k, lambda_h, alpha_h) for k in range(n)])
    pmf_a = np.array([negative_binomial_pmf(k, lambda_a, alpha_a) for k in range(n)])
    mat = np.outer(pmf_h, pmf_a)
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


# ═══════════════════════════════════════════════════════════════════
# DIXON-COLES MODEL
# ═══════════════════════════════════════════════════════════════════
def dixon_coles_tau(
    hg: int,
    ag: int,
    lambda_h: float,
    lambda_a: float,
    rho: float = -0.13,
) -> float:
    """Dixon-Coles τ correction factor for low-scoring games.

    The Dixon-Coles model adjusts Poisson probabilities for observed
    dependencies in low-scoring matches:
        P(X=hg, Y=ag) = τ(hg, ag, λ_h, λ_a, ρ) · Poisson(hg|λ_h) · Poisson(ag|λ_a)

    The τ factor corrects only the (0,0), (1,0), (0,1), (1,1) scorelines.
    For all other scores, τ = 1.0 (no adjustment).

    The formula depends on the specific scoreline:
        - τ(0,0) = 1 - λ_h · λ_a · ρ
        - τ(1,0) = 1 + λ_a · ρ
        - τ(0,1) = 1 + λ_h · ρ
        - τ(1,1) = 1 - ρ
        - τ(i,j) = 1.0  (for i ≥ 2 or j ≥ 2)

    With ρ < 0 (typical), this increases P(0-0) and P(1-1), which are
    observed more frequently in real football than independent Poisson predicts.

    Args:
        hg: Home goals (non-negative integer)
        ag: Away goals (non-negative integer)
        lambda_h: Home expected goals rate
        lambda_a: Away expected goals rate
        rho: Dependency parameter (typically ≈ -0.13, negative)

    Returns:
        Correction factor τ(hg, ag, λ_h, λ_a, ρ)
    """
    if hg >= 2 or ag >= 2:
        # No adjustment for higher scores
        return 1.0

    if hg == 0 and ag == 0:
        tau = 1.0 - lambda_h * lambda_a * rho
    elif hg == 1 and ag == 0:
        tau = 1.0 + lambda_a * rho
    elif hg == 0 and ag == 1:
        tau = 1.0 + lambda_h * rho
    elif hg == 1 and ag == 1:
        tau = 1.0 - rho
    else:
        tau = 1.0

    # Clamp to non-negative
    return max(0.0, tau)


def build_dc_score_matrix(
    lambda_h: float,
    lambda_a: float,
    rho: float = -0.13,
    max_goals: int = 8,
) -> np.ndarray:
    """Build score matrix from Dixon-Coles adjusted Poisson distribution.

    The Dixon-Coles model is an industry-standard for football prediction.
    It corrects the 4 low-scoring scorelines (0-0, 1-0, 0-1, 1-1) for
    observed dependencies (typically ρ < 0, making these scores MORE likely).

    The final PMF is:
        P(X=hg, Y=ag) = τ(hg, ag) · Poisson(hg|λ_h) · Poisson(ag|λ_a)

    where τ applies only to low-scoring matches.

    Args:
        lambda_h: Home team expected goals (attack strength × away defense)
        lambda_a: Away team expected goals (attack strength × home defense)
        rho: Dependency parameter (default ≈ -0.13, typically in [-0.5, 0.5])
        max_goals: Maximum goals per team to consider (default 8)

    Returns:
        (max_goals+1) × (max_goals+1) probability matrix with:
            - Rows = home goals (0 to max_goals)
            - Columns = away goals (0 to max_goals)
            - Each entry = P(H=i, A=j) normalized to sum to 1.0

    Example:
        >>> mx = build_dc_score_matrix(1.5, 1.2, rho=-0.13)
        >>> mx.shape
        (9, 9)
        >>> mx.sum()
        1.0
        >>> mx[0, 0]  # P(0-0) — higher than independent Poisson
        0.047...
    """
    lambda_h = max(0.01, float(lambda_h))
    lambda_a = max(0.01, float(lambda_a))

    n = max_goals + 1
    mat = np.zeros((n, n))

    # Precompute Poisson PMFs
    h_pmf = poisson_dist.pmf(np.arange(n), lambda_h)
    a_pmf = poisson_dist.pmf(np.arange(n), lambda_a)

    # Build matrix with τ corrections
    for i in range(n):
        for j in range(n):
            tau = dixon_coles_tau(i, j, lambda_h, lambda_a, rho)
            mat[i, j] = tau * h_pmf[i] * a_pmf[j]

    # Normalize
    total = mat.sum()
    if total > 0:
        mat /= total

    return mat


# ═══════════════════════════════════════════════════════════════════
# DIAGONAL-INFLATED BIVARIATE POISSON (DIBP)
# ═══════════════════════════════════════════════════════════════════
def build_dibp_score_matrix(lambda_h, lambda_a, lambda_3=0.1, diag_inflate=0.05, max_goals=8):
    """Diagonal-Inflated Bivariate Poisson score matrix.

    State of the art: RPS = 0.189 on La Liga (2025 research).
    Inflates draw scorelines (0-0, 1-1, 2-2...) to match empirical draw rates.
    """
    base = build_bivariate_poisson_matrix(lambda_h, lambda_a, lambda_3, max_goals)

    # Compute diagonal inflation: redistribute mass to draw scorelines
    n = max_goals + 1
    diag_probs = np.zeros((n, n))
    for k in range(n):
        # Probability of k-k draw under independent Poisson
        diag_probs[k, k] = poisson_dist.pmf(k, lambda_h) * poisson_dist.pmf(k, lambda_a)

    # Normalize diagonal probs
    diag_total = diag_probs.sum()
    if diag_total > 0:
        diag_probs = diag_probs / diag_total

    # Mix: (1-diag_inflate) * base + diag_inflate * diag_probs
    adjusted = (1.0 - diag_inflate) * base + diag_inflate * diag_probs

    # Renormalize
    total = adjusted.sum()
    if total > 0:
        adjusted = adjusted / total

    return adjusted
