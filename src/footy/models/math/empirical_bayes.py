"""Empirical Bayes methods for football prediction.

Implements Beta-Binomial shrinkage and league-specific priors for:
- Win rate estimation
- Clean sheet rates
- BTTS (Both Teams To Score)
- Over/Under betting lines
- Dixon-Coles rho parameter MLE estimation

References:
    Gelman et al. "Bayesian Data Analysis" — empirical Bayes shrinkage
    Efron & Morris (1973) "Stein's Estimation Rule and Its Competitors"
    Dixon & Coles (1997) "Modelling association football scores and unsorted tables"
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson as poisson_dist


# ═══════════════════════════════════════════════════════════════════
# EMPIRICAL BAYES — BETA-BINOMIAL SHRINKAGE
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

    Args:
        league: League code (e.g., 'PL', 'PD', 'SA', 'BL1', 'FL1')
        stat_type: Type of statistic ('home_win', 'draw', 'cs', 'btts', 'o25')

    Returns:
        Tuple (alpha, beta) for Beta distribution prior
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
# DIXON-COLES RHO PARAMETER MLE ESTIMATION
# ═══════════════════════════════════════════════════════════════════
def estimate_rho_mle(
    hg_or_df,
    ag=None,
    lam_h=None,
    lam_a=None,
) -> float:
    """Estimate the Dixon-Coles rho parameter via maximum likelihood estimation.

    The rho parameter controls the dependency structure in low-scoring matches:
        - ρ ≈ -0.13 is typical for European football (default)
        - ρ < 0: matches have MORE 0-0, 1-0, 0-1, 1-1 than independent Poisson
        - ρ > 0: matches have FEWER of these scorelines (rare)

    MLE optimization finds ρ that maximizes the likelihood of observed data.

    Accepts two calling conventions:
        1. estimate_rho_mle(df) — DataFrame with 'home_goals', 'away_goals', 'lambda_h', 'lambda_a'
        2. estimate_rho_mle(hg, ag, lam_h, lam_a) — separate arrays

    Returns:
        Optimal rho parameter (typically in [-0.5, 0.5])

    Note:
        If optimization fails or data is insufficient, returns -0.13 (default).
    """
    # Handle both calling conventions
    if ag is not None:
        # Called as estimate_rho_mle(hg, ag, lam_h, lam_a)
        hg_arr = np.asarray(hg_or_df, dtype=float)
        ag_arr = np.asarray(ag, dtype=float)
        lh = np.asarray(lam_h, dtype=float) if lam_h is not None else np.full_like(hg_arr, 1.3)
        la = np.asarray(lam_a, dtype=float) if lam_a is not None else np.full_like(ag_arr, 1.1)
        if len(hg_arr) < 10:
            return -0.13
        hg = hg_arr
        ag = ag_arr
    else:
        # Called as estimate_rho_mle(df)
        matches_df = hg_or_df
        if hasattr(matches_df, 'empty') and (matches_df.empty or len(matches_df) < 10):
            return -0.13
        required_cols = ["home_goals", "away_goals", "lambda_h", "lambda_a"]
        if hasattr(matches_df, 'columns') and not all(col in matches_df.columns for col in required_cols):
            return -0.13
        hg = matches_df["home_goals"].values.astype(float)
        ag = matches_df["away_goals"].values.astype(float)
        lh = matches_df["lambda_h"].values.astype(float)
        la = matches_df["lambda_a"].values.astype(float)

    def negative_log_likelihood(rho_val: float) -> float:
        """Compute negative log-likelihood of observed data under rho."""
        rho = float(rho_val[0])
        nll = 0.0

        for i in range(len(hg)):
            h, a, lh_i, la_i = hg[i], ag[i], lh[i], la[i]

            # Compute τ correction
            if h >= 2 or a >= 2:
                tau = 1.0
            elif h == 0 and a == 0:
                tau = max(0.0, 1.0 - lh_i * la_i * rho)
            elif h == 1 and a == 0:
                tau = max(0.0, 1.0 + la_i * rho)
            elif h == 0 and a == 1:
                tau = max(0.0, 1.0 + lh_i * rho)
            elif h == 1 and a == 1:
                tau = max(0.0, 1.0 - rho)
            else:
                tau = 1.0

            # Log-likelihood contribution
            # P(H=h, A=a) = τ · Poisson(h|λ_h) · Poisson(a|λ_a)
            pois_h = poisson_dist.pmf(int(h), lh_i)
            pois_a = poisson_dist.pmf(int(a), la_i)

            p = tau * pois_h * pois_a

            # Avoid log(0)
            if p > 1e-15:
                nll -= math.log(p)
            else:
                nll += 100.0  # Large penalty for near-zero probability

        return nll

    # Optimize over bounded domain — rho typically in [-0.4, 0.4] for football
    result = minimize(
        negative_log_likelihood,
        x0=[-0.13],
        method="L-BFGS-B",
        bounds=[(-0.4, 0.4)],
    )

    if result.success:
        return float(result.x[0])
    else:
        return -0.13  # Default if optimization fails
