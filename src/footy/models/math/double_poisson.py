"""Efron's Double Poisson distribution for football goal prediction.

Implements the Double Poisson distribution (Efron 1986) for modelling
goal counts with flexible dispersion independent of the mean.

References:
    Efron (1986) "Double Exponential Families and Their Use in
        Generalized Linear Models"
    Frome et al. (1992) "The Double Poisson Distribution and Applied
        Regression"
"""
from __future__ import annotations

import math

import numpy as np


def double_poisson_pmf(k: int, mu: float, phi: float = 1.0) -> float:
    """Double Poisson PMF: P(X=k).

    A generalization of Poisson with independent dispersion parameter φ:
        φ < 1: over-dispersed (variance > mean)
        φ = 1: standard Poisson
        φ > 1: under-dispersed (variance < mean)

    The Double Poisson distribution can better capture the empirical
    variance-to-mean ratio of football goal counts than standard Poisson.

    P(X=k) ≈ Constant · √φ · exp(-φμ + k·log(e·μ) - (k - μ)² / (2μ))

    Reference: Efron (1986)

    Args:
        k: Number of goals (non-negative integer)
        mu: Mean parameter (expected goals)
        phi: Dispersion parameter (1 = Poisson)

    Returns:
        P(X=k) under Double Poisson model
    """
    mu = max(1e-9, float(mu))
    phi = max(0.1, float(phi))

    if k == 0:
        return math.exp(-phi * mu)

    # Log-space computation for numerical stability
    # Core Poisson-like term
    log_pmf = (
        0.5 * math.log(phi)  # normalization
        - phi * mu  # -μ term (scaled by φ)
        + k * math.log(math.e * mu)  # k·log(e·μ)
        - k * math.log(max(k, 1))  # -log(k!)  approximation
        + k  # factorial adjustment
        - 0.5 * math.log(2.0 * math.pi * max(k, 1))  # Stirling normalization
    )

    return max(0.0, math.exp(log_pmf))


def build_double_poisson_matrix(
    lambda_h: float,
    lambda_a: float,
    phi_h: float = 0.92,
    phi_a: float = 0.92,
    max_goals: int = 8,
) -> np.ndarray:
    """Build score matrix using Double Poisson marginals.

    Double Poisson allows teams to have different variance-to-mean
    ratios independently of their mean goal counts.

    Args:
        lambda_h: Home expected goals (mean)
        lambda_a: Away expected goals (mean)
        phi_h: Home dispersion (< 1 = over-dispersed, typical ≈ 0.92)
        phi_a: Away dispersion (< 1 = over-dispersed)
        max_goals: Maximum goals per team

    Returns:
        (max_goals+1) × (max_goals+1) probability matrix
    """
    n = max_goals + 1
    pmf_h = np.array([double_poisson_pmf(k, lambda_h, phi_h) for k in range(n)])
    pmf_a = np.array([double_poisson_pmf(k, lambda_a, phi_a) for k in range(n)])

    # Normalize marginals
    pmf_h = pmf_h / max(pmf_h.sum(), 1e-12)
    pmf_a = pmf_a / max(pmf_a.sum(), 1e-12)

    # Build joint distribution (independent marginals)
    mat = np.outer(pmf_h, pmf_a)

    # Renormalize full matrix
    total = mat.sum()
    if total > 0:
        mat /= total

    return mat
