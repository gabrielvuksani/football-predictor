"""
Data-driven parameter estimation for expert models.

Replaces hardcoded expert parameters with estimates learned from
historical goal sequences. Provides Baum-Welch HMM fitting,
Score-Driven (GAS) MLE, copula dependence estimation, and
COM-Poisson dispersion estimation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import kendalltau


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function computed in log-space to avoid overflow.

    Parameters
    ----------
    k : int
        Non-negative integer count.
    lam : float
        Poisson rate parameter (must be > 0).

    Returns
    -------
    float
        P(X = k) for X ~ Poisson(lam).
    """
    if lam <= 0:
        return 0.0 if k > 0 else 1.0
    log_pmf = k * np.log(lam) - lam - gammaln(k + 1)
    return float(np.exp(log_pmf))


def _default_hmm_params(n_states: int = 3) -> dict:
    """Return sensible default HMM parameters when data is insufficient.

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 3, representing low/medium/high
        scoring phases).

    Returns
    -------
    dict
        Keys: 'emission_rates' (ndarray of shape (n_states,)),
              'transition_matrix' (ndarray of shape (n_states, n_states)).
    """
    # Low / medium / high scoring regimes
    rates = np.linspace(0.5, 2.5, n_states)
    # Sticky diagonal with uniform off-diagonal
    transition = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1))
    np.fill_diagonal(transition, 0.95)
    # Normalise rows
    transition = transition / transition.sum(axis=1, keepdims=True)
    return {
        "emission_rates": rates,
        "transition_matrix": transition,
    }


# ---------------------------------------------------------------------------
# 1. HMM — Baum-Welch EM
# ---------------------------------------------------------------------------

def learn_hmm_params(
    goals: list[int],
    n_states: int = 3,
    n_iter: int = 100,
) -> dict:
    """Fit a Poisson-emission HMM via Baum-Welch EM.

    Each hidden state emits goals according to a Poisson distribution
    with its own rate parameter.  The algorithm iterates E-step
    (forward-backward) and M-step (re-estimation) until convergence or
    *n_iter* iterations.

    Parameters
    ----------
    goals : list[int]
        Observed goal counts per match.
    n_states : int
        Number of latent states (default 3).
    n_iter : int
        Maximum EM iterations (default 100).

    Returns
    -------
    dict
        'emission_rates': ndarray of shape (n_states,) sorted ascending.
        'transition_matrix': ndarray of shape (n_states, n_states) with
        rows reordered to match the sorted emission rates.
    """
    if len(goals) < 20:
        return _default_hmm_params(n_states)

    T = len(goals)
    obs = np.asarray(goals, dtype=int)

    # Initialise emission rates spread across the data range
    emission_rates = np.linspace(
        max(0.3, np.mean(obs) - 1.0),
        max(1.0, np.mean(obs) + 1.0),
        n_states,
    )

    # Uniform initial state distribution
    pi = np.ones(n_states) / n_states

    # Sticky transition matrix
    trans = np.full((n_states, n_states), 0.1 / max(n_states - 1, 1))
    np.fill_diagonal(trans, 0.9)
    trans = trans / trans.sum(axis=1, keepdims=True)

    for _iteration in range(n_iter):
        # --- Emission matrix B[t, j] = P(obs[t] | state j) ---
        B = np.zeros((T, n_states))
        for j in range(n_states):
            for t in range(T):
                B[t, j] = _poisson_pmf(obs[t], emission_rates[j])
        # Floor to avoid zeros
        B = np.maximum(B, 1e-300)

        # --- Forward pass (scaled) ---
        alpha = np.zeros((T, n_states))
        scales = np.zeros(T)
        alpha[0] = pi * B[0]
        scales[0] = alpha[0].sum()
        if scales[0] == 0:
            return _default_hmm_params(n_states)
        alpha[0] /= scales[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ trans) * B[t]
            scales[t] = alpha[t].sum()
            if scales[t] == 0:
                return _default_hmm_params(n_states)
            alpha[t] /= scales[t]

        # --- Backward pass (scaled) ---
        beta = np.zeros((T, n_states))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (trans @ (B[t + 1] * beta[t + 1]))
            if scales[t + 1] > 0:
                beta[t] /= scales[t + 1]

        # --- Gamma and Xi ---
        gamma = alpha * beta
        gamma_sums = gamma.sum(axis=1, keepdims=True)
        gamma_sums = np.maximum(gamma_sums, 1e-300)
        gamma = gamma / gamma_sums

        xi = np.zeros((T - 1, n_states, n_states))
        for t in range(T - 1):
            numerator = (
                alpha[t][:, None]
                * trans
                * B[t + 1][None, :]
                * beta[t + 1][None, :]
            )
            denom = numerator.sum()
            if denom > 0:
                xi[t] = numerator / denom

        # --- M-step ---
        pi = gamma[0]

        # Transition matrix
        xi_sum = xi.sum(axis=0)
        row_sums = xi_sum.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-300)
        trans = xi_sum / row_sums

        # Emission rates
        gamma_col_sums = gamma.sum(axis=0)
        gamma_col_sums = np.maximum(gamma_col_sums, 1e-300)
        for j in range(n_states):
            emission_rates[j] = (gamma[:, j] * obs).sum() / gamma_col_sums[j]
        # Floor rates
        emission_rates = np.maximum(emission_rates, 0.01)

    # Sort states by emission rate ascending
    order = np.argsort(emission_rates)
    emission_rates = emission_rates[order]
    trans = trans[order][:, order]

    return {
        "emission_rates": emission_rates,
        "transition_matrix": trans,
    }


# ---------------------------------------------------------------------------
# 2. Score-Driven (GAS) model — MLE
# ---------------------------------------------------------------------------

def learn_gas_params(goals: NDArray, home: bool = True) -> dict:
    """Maximum-likelihood estimation for a Score-Driven (GAS) model.

    The model follows Koopman & Lit (2019) with Poisson observations and
    score-driven intensity updates:

        lambda[t] = omega + A * score[t-1] + B * lambda[t-1]

    where score[t] = y[t] - lambda[t] (the Poisson score).

    Parameters
    ----------
    goals : ndarray
        Observed goal counts (1-D integer array).
    home : bool
        Whether this is the home side (affects default omega).

    Returns
    -------
    dict
        'omega': float — intercept.
        'A': float — score loading.
        'B': float — autoregressive coefficient.
    """
    default_omega = 0.15 if home else 0.10
    defaults = {"omega": default_omega, "A": 0.05, "B": 0.95}

    goals = np.asarray(goals, dtype=float)
    if len(goals) < 30:
        return defaults

    def neg_log_lik(params: NDArray) -> float:
        omega, A, B = params
        T = len(goals)
        lam = np.mean(goals)  # initialise at sample mean
        ll = 0.0
        for t in range(T):
            lam = max(lam, 0.01)  # floor to avoid log(0)
            ll += goals[t] * np.log(lam) - lam - gammaln(goals[t] + 1)
            score = goals[t] - lam
            lam = omega + A * score + B * lam
        return -ll

    bounds = [(-2.0, 2.0), (0.001, 0.5), (0.5, 0.999)]
    x0 = np.array([default_omega, 0.05, 0.90])

    try:
        result = minimize(
            neg_log_lik,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if result.success or result.fun < neg_log_lik(x0):
            return {
                "omega": float(result.x[0]),
                "A": float(result.x[1]),
                "B": float(result.x[2]),
            }
    except Exception:
        pass

    return defaults


# ---------------------------------------------------------------------------
# 3. Copula dependence parameter
# ---------------------------------------------------------------------------

def learn_copula_theta(
    home_goals: NDArray,
    away_goals: NDArray,
    family: str = "frank",
) -> float:
    """Estimate copula dependence parameter from Kendall's tau.

    Supports Frank, Clayton, and Gumbel families with the standard
    tau-to-theta relationships.

    Parameters
    ----------
    home_goals : ndarray
        Home team goal counts.
    away_goals : ndarray
        Away team goal counts.
    family : str
        Copula family: 'frank', 'clayton', or 'gumbel'.

    Returns
    -------
    float
        Estimated copula dependence parameter theta, clipped to a safe
        range for the chosen family.
    """
    family = family.lower()
    defaults = {"frank": -1.0, "clayton": 0.5, "gumbel": 1.2}
    default_theta = defaults.get(family, -1.0)

    home_goals = np.asarray(home_goals, dtype=float)
    away_goals = np.asarray(away_goals, dtype=float)

    if len(home_goals) < 30 or len(away_goals) < 30:
        return default_theta

    # Kendall's tau
    tau, _pvalue = kendalltau(home_goals, away_goals)

    if np.isnan(tau):
        return default_theta

    if family == "frank":
        # Frank: theta ≈ -9 * tau  (linearised Debye inversion)
        theta = -9.0 * tau
        theta = float(np.clip(theta, -9.0, 9.0))
    elif family == "clayton":
        # Clayton: theta = 2*tau / (1 - tau), requires tau < 1
        if tau >= 1.0:
            tau = 0.99
        theta = 2.0 * tau / (1.0 - tau)
        theta = float(np.clip(theta, 0.01, 20.0))
    elif family == "gumbel":
        # Gumbel: theta = 1 / (1 - tau), requires tau < 1
        if tau >= 1.0:
            tau = 0.99
        theta = 1.0 / (1.0 - tau)
        theta = float(np.clip(theta, 1.0, 20.0))
    else:
        raise ValueError(f"Unknown copula family: {family!r}")

    return theta


# ---------------------------------------------------------------------------
# 4. COM-Poisson dispersion
# ---------------------------------------------------------------------------

def learn_com_poisson_nu(goals: NDArray) -> float:
    """Estimate the COM-Poisson dispersion parameter from the data.

    The COM-Poisson distribution generalises the Poisson by adding a
    dispersion parameter nu.  When nu = 1 it reduces to standard Poisson;
    nu > 1 is under-dispersed; nu < 1 is over-dispersed.

    The estimator uses the variance-to-mean ratio: nu ≈ 1 / (Var/Mean).

    Parameters
    ----------
    goals : ndarray
        Observed goal counts.

    Returns
    -------
    float
        Estimated nu, clipped to [0.3, 3.0].
    """
    goals = np.asarray(goals, dtype=float)

    if len(goals) < 30:
        return 0.93  # Slightly over-dispersed default for football

    mean = np.mean(goals)
    var = np.var(goals, ddof=1)

    if mean <= 0 or var <= 0:
        return 0.93

    ratio = var / mean
    nu = 1.0 / ratio

    return float(np.clip(nu, 0.3, 3.0))
