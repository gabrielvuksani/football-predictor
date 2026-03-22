"""Copula-based joint distributions for football prediction.

Implements Frank, Clayton, Gumbel, and Gaussian copulas for modelling
full-match dependencies between home and away team scores.

References:
    Joe (1997) "Multivariate Models and Dependence Concepts"
    Nelsen (2006) "An Introduction to Copulas"
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import poisson as poisson_dist


# ═══════════════════════════════════════════════════════════════════
# COPULA FUNCTIONS
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


def _clayton_copula(u: float, v: float, theta: float) -> float:
    """Clayton copula C(u, v; θ).

    C(u,v;θ) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}

    Properties:
        θ → 0:  independence
        θ > 0:  positive lower-tail dependence (correlated defensive strength)
        θ → ∞:  perfect positive dependence

    Clayton copula captures lower-tail dependence: when both teams score
    low (defensive matches), scores are more correlated. Good for modelling
    defensive leagues.
    """
    if abs(theta) < 1e-8:
        return u * v
    if u < 1e-15 or v < 1e-15:
        return 0.0
    val = u ** (-theta) + v ** (-theta) - 1.0
    if val <= 0:
        return 0.0
    return max(0.0, min(1.0, val ** (-1.0 / theta)))


def _gumbel_copula(u: float, v: float, theta: float) -> float:
    """Gumbel copula C(u, v; θ).

    C(u,v;θ) = exp(-((−ln u)^θ + (−ln v)^θ)^{1/θ})

    Properties:
        θ = 1:  independence
        θ > 1:  positive upper-tail dependence (correlated attacking play)
        θ → ∞:  perfect positive dependence

    Gumbel copula captures upper-tail dependence: when both teams score
    high (open attacking matches), scores are more correlated. Good for
    modelling attacking leagues with many goals.
    """
    if theta < 1.0:
        theta = 1.0
    if abs(theta - 1.0) < 1e-8:
        return u * v
    if u < 1e-15 or v < 1e-15:
        return 0.0
    try:
        neg_ln_u = -math.log(max(u, 1e-15))
        neg_ln_v = -math.log(max(v, 1e-15))
        A = neg_ln_u ** theta + neg_ln_v ** theta
        return math.exp(-(A ** (1.0 / theta)))
    except (OverflowError, ValueError):
        return u * v


def _gaussian_copula(u: float, v: float, rho: float) -> float:
    """Gaussian copula C(u, v; ρ).

    Uses the bivariate normal CDF with correlation ρ.
    Symmetric dependence — no tail preference.

    Args:
        u, v: Marginal CDF values in (0, 1)
        rho: Correlation parameter in (-1, 1)
    """
    from scipy.stats import norm as norm_dist
    from scipy.stats import multivariate_normal

    if abs(rho) < 1e-8:
        return u * v
    u_c = max(1e-10, min(1 - 1e-10, u))
    v_c = max(1e-10, min(1 - 1e-10, v))
    x = norm_dist.ppf(u_c)
    y = norm_dist.ppf(v_c)
    # Bivariate normal CDF approximation
    cov = [[1.0, rho], [rho, 1.0]]
    rv = multivariate_normal(mean=[0, 0], cov=cov)
    return float(rv.cdf([x, y]))


# ═══════════════════════════════════════════════════════════════════
# COPULA SCORE MATRIX
# ═══════════════════════════════════════════════════════════════════
def build_copula_score_matrix(
    lambda_h: float,
    lambda_a: float,
    theta: float = -2.0,
    max_goals: int = 8,
    copula_family: str = "frank",
) -> np.ndarray:
    """Build score matrix using copula for full dependency structure.

    Rather than adjusting only 4 cells (Dixon-Coles), the copula models
    dependency across ALL scorelines by coupling the marginal Poisson CDFs.

    The joint PMF is obtained via inclusion-exclusion on the copula:
        P(H=i, A=j) = C(F_H(i), F_A(j)) - C(F_H(i-1), F_A(j))
                     - C(F_H(i), F_A(j-1)) + C(F_H(i-1), F_A(j-1))

    where F_H, F_A are the Poisson CDFs.

    Copula families:
        frank:    symmetric dependence, good all-purpose (θ ≈ -2 to 3)
        clayton:  lower-tail dependence, for defensive leagues (θ > 0)
        gumbel:   upper-tail dependence, for attacking leagues (θ >= 1)
        gaussian: symmetric bivariate normal dependence (ρ ∈ (-1, 1))

    Args:
        lambda_h: Home expected goals
        lambda_a: Away expected goals
        theta: Copula dependency parameter
        max_goals: Maximum goals per team
        copula_family: One of "frank", "clayton", "gumbel", "gaussian"

    Returns:
        (max_goals+1) × (max_goals+1) probability matrix
    """
    copula_fn = {
        "frank": _frank_copula,
        "clayton": _clayton_copula,
        "gumbel": _gumbel_copula,
        "gaussian": _gaussian_copula,
    }.get(copula_family, _frank_copula)
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
            c11 = copula_fn(cdf_h[i + 1], cdf_a[j + 1], theta)
            c10 = copula_fn(cdf_h[i],     cdf_a[j + 1], theta)
            c01 = copula_fn(cdf_h[i + 1], cdf_a[j],     theta)
            c00 = copula_fn(cdf_h[i],     cdf_a[j],     theta)
            mat[i, j] = max(0.0, c11 - c10 - c01 + c00)

    # Renormalize
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat
