from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def fit_poisson(matches: pd.DataFrame, halflife_days: float = 180.0) -> dict:
    """
    Fits team attack/defense + home advantage using weighted Poisson likelihood.
    matches columns: home_team, away_team, home_goals, away_goals, utc_date

    Improvements over basic version:
    - L-BFGS-B with parameter bounds to prevent extreme values
    - Lower L2 regularization (0.02) for better fit
    - Centered parameters after fitting for stability
    - Robust handling of edge cases
    """
    df = matches.dropna(subset=["home_goals","away_goals","utc_date"]).copy()
    if df.empty:
        return {"teams": [], "attack": [], "defense": [], "home_adv": 0.0, "mu": 0.0}

    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)
    tmax = df["utc_date"].max()
    age_days = (tmax - df["utc_date"]).dt.total_seconds() / 86400.0
    w = np.exp(-np.log(2) * age_days / halflife_days).to_numpy()

    teams = pd.Index(sorted(set(df["home_team"]) | set(df["away_team"])))
    idx = {t:i for i,t in enumerate(teams)}
    n = len(teams)

    h = df["home_team"].map(idx).to_numpy()
    a = df["away_team"].map(idx).to_numpy()
    hg = df["home_goals"].to_numpy(dtype=float)
    ag = df["away_goals"].to_numpy(dtype=float)

    # parameters: [mu, home_adv, attack(0..n-2), defense(0..n-2)]
    # last team fixed at 0 for identifiability
    n_params = 2 + 2*(n-1)
    x0 = np.zeros(n_params, dtype=float)

    # Bounds: prevent extreme parameter values
    bounds = [(None, None)] * 2  # mu, home_adv unbounded (ish)
    bounds[0] = (-1.0, 2.0)   # mu (log-scale intercept)
    bounds[1] = (-0.5, 1.0)   # home advantage
    bounds += [(-2.5, 2.5)] * (2 * (n - 1))  # attack and defense

    def unpack(x):
        mu = x[0]
        ha = x[1]
        att = np.zeros(n)
        deff = np.zeros(n)
        att[:-1] = x[2:2+(n-1)]
        deff[:-1] = x[2+(n-1):]
        return mu, ha, att, deff

    def nll(x):
        mu, ha, att, deff = unpack(x)
        lam_h = np.exp(mu + ha + att[h] - deff[a])
        lam_a = np.exp(mu + att[a] - deff[h])
        # Clamp lambdas for numerical stability
        lam_h = np.clip(lam_h, 1e-6, 20.0)
        lam_a = np.clip(lam_a, 1e-6, 20.0)
        # Poisson loglik: k*log(lam) - lam - log(k!)
        ll = (hg*np.log(lam_h) - lam_h) + (ag*np.log(lam_a) - lam_a)
        return -np.sum(w * ll)

    def reg(x, l2=0.02):
        return l2 * np.sum(x[2:]**2)

    def obj(x):
        return nll(x) + reg(x)

    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 300, "ftol": 1e-9})
    mu, ha, att, deff = unpack(res.x)

    # Center parameters for stability (doesn't affect predictions)
    att_mean = att.mean()
    deff_mean = deff.mean()
    att = att - att_mean
    deff = deff - deff_mean
    mu = mu + att_mean - deff_mean  # absorb centering into mu

    return {"teams": teams.tolist(), "attack": att.tolist(), "defense": deff.tolist(), "home_adv": float(ha), "mu": float(mu)}

def expected_goals(state: dict, home_team: str, away_team: str) -> tuple[float,float]:
    teams = state["teams"]
    if not teams:
        return (1.3, 1.1)
    idx = {t:i for i,t in enumerate(teams)}
    if home_team not in idx or away_team not in idx:
        return (1.3, 1.1)
    mu = state["mu"]; ha = state["home_adv"]
    att = np.array(state["attack"]); deff = np.array(state["defense"])
    i = idx[home_team]; j = idx[away_team]
    lam_h = float(np.clip(np.exp(mu + ha + att[i] - deff[j]), 0.1, 8.0))
    lam_a = float(np.clip(np.exp(mu + att[j] - deff[i]), 0.1, 8.0))
    return lam_h, lam_a

def scoreline_probs(lam_h: float, lam_a: float, max_goals: int = 8) -> np.ndarray:
    # matrix [hg, ag]
    hg = np.arange(max_goals+1)
    ag = np.arange(max_goals+1)
    ph = np.exp(-lam_h) * np.power(lam_h, hg) / np.array([math.factorial(int(k)) for k in hg])
    pa = np.exp(-lam_a) * np.power(lam_a, ag) / np.array([math.factorial(int(k)) for k in ag])
    return np.outer(ph, pa)

def outcome_probs(lam_h: float, lam_a: float) -> tuple[float,float,float]:
    M = scoreline_probs(lam_h, lam_a, max_goals=8)
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())
    s = p_home + p_draw + p_away
    return (p_home/s, p_draw/s, p_away/s)
