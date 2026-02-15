from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import poisson

# Dixon–Coles low-score adjustment tau
def _tau(hg: int, ag: int, lam: float, mu: float, rho: float) -> float:
    if hg == 0 and ag == 0:
        return 1.0 - (lam * mu * rho)
    if hg == 0 and ag == 1:
        return 1.0 + (lam * rho)
    if hg == 1 and ag == 0:
        return 1.0 + (mu * rho)
    if hg == 1 and ag == 1:
        return 1.0 - rho
    return 1.0

def _half_life_xi(half_life_days: float) -> float:
    return float(np.log(2.0) / max(1e-9, half_life_days))

@dataclass
class DCModel:
    teams: list[str]
    team_idx: dict[str, int]
    atk: np.ndarray           # shape (n,)
    dfn: np.ndarray           # shape (n,)
    home_adv: float
    rho: float
    xi: float
    max_goals: int = 10

def fit_dc(
    df: pd.DataFrame,
    half_life_days: float = 365.0,
    max_goals: int = 10,
    l2: float = 0.002,
) -> DCModel | None:
    """
    df columns required:
      utc_date (datetime64[ns, UTC]), home_team, away_team, home_goals, away_goals
    """
    if df is None or df.empty or len(df) < 250:
        return None

    df = df.copy()
    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)
    tmax = df["utc_date"].max()
    days_ago = (tmax - df["utc_date"]).dt.total_seconds().to_numpy() / 86400.0
    xi = _half_life_xi(half_life_days)
    w = np.exp(-xi * days_ago).astype(float)

    teams = sorted(pd.unique(df[["home_team", "away_team"]].values.ravel("K")))
    n = len(teams)
    if n < 6:
        return None
    idx = {t: i for i, t in enumerate(teams)}

    h = df["home_team"].map(idx).to_numpy(int)
    a = df["away_team"].map(idx).to_numpy(int)
    hg = df["home_goals"].to_numpy(int)
    ag = df["away_goals"].to_numpy(int)

    # Fix identifiability by pinning last team atk/dfn = 0
    n_free = n - 1
    # params: atk[0:n_free], dfn[0:n_free], home_adv, rho
    x0 = np.zeros(2 * n_free + 2, dtype=float)
    x0[-2] = 0.10   # home_adv
    x0[-1] = -0.05  # rho

    bounds = [(-3.0, 3.0)] * (2 * n_free) + [(-1.5, 1.5), (-0.35, 0.35)]

    def unpack(x):
        atk = np.zeros(n, dtype=float)
        dfn = np.zeros(n, dtype=float)
        atk[:n_free] = x[:n_free]
        dfn[:n_free] = x[n_free:2*n_free]
        home_adv = float(x[-2])
        rho = float(x[-1])
        return atk, dfn, home_adv, rho

    def nll(x):
        atk, dfn, home_adv, rho = unpack(x)

        lam = np.exp(home_adv + atk[h] + dfn[a])
        mu  = np.exp(atk[a] + dfn[h])

        # Clamp for numerical stability
        lam = np.clip(lam, 1e-6, 20.0)
        mu  = np.clip(mu, 1e-6, 20.0)

        # log pmf + tau adjustment
        ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)

        # Vectorized tau computation for low-score states
        tau = np.ones_like(ll)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        tau = np.maximum(tau, 1e-9)
        ll = ll + np.log(tau)

        # time weights
        ll = ll * w

        # mild L2 to prevent explosions
        reg = l2 * (np.sum(atk**2) + np.sum(dfn**2) + home_adv**2 + rho**2)
        return float(-np.sum(ll) + reg)

    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    if not res.success:
        # still return best found if finite
        if not np.isfinite(res.fun):
            return None

    atk, dfn, home_adv, rho = unpack(res.x)

    # center parameters (optional stability)
    atk = atk - atk.mean()
    dfn = dfn - dfn.mean()

    return DCModel(
        teams=teams,
        team_idx=idx,
        atk=atk,
        dfn=dfn,
        home_adv=home_adv,
        rho=rho,
        xi=xi,
        max_goals=max_goals,
    )

def predict_1x2(model: DCModel, home_team: str, away_team: str) -> tuple[float,float,float,float,float,float]:
    """
    Returns: (p_home, p_draw, p_away, eg_home, eg_away, p_over25)
    """
    hi = model.team_idx.get(home_team)
    ai = model.team_idx.get(away_team)
    if hi is None or ai is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    lam = float(np.exp(model.home_adv + model.atk[hi] + model.dfn[ai]))
    mu  = float(np.exp(model.atk[ai] + model.dfn[hi]))

    m = model.max_goals
    ph = poisson.pmf(np.arange(m+1), lam)
    pa = poisson.pmf(np.arange(m+1), mu)

    # score matrix
    P = np.outer(ph, pa)

    # apply tau to low scores
    for hg in range(0, 2):
        for ag in range(0, 2):
            P[hg, ag] *= max(1e-9, _tau(hg, ag, lam, mu, model.rho))

    # renormalize
    s = P.sum()
    if s <= 0:
        return (0.0, 0.0, 0.0, lam, mu, 0.0)
    P = P / s

    p_home = float(np.tril(P, -1).sum())
    p_draw = float(np.trace(P))
    p_away = float(np.triu(P,  1).sum())

    # over 2.5
    p_over25 = float(P[(np.add.outer(np.arange(m+1), np.arange(m+1)) >= 3)].sum())

    return (p_home, p_draw, p_away, lam, mu, p_over25)
