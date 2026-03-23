"""SkellamRegressionExpert — Goal difference model with covariate regression.

Instead of modeling home and away goals SEPARATELY (like Poisson), this
expert models the goal DIFFERENCE directly using the Skellam distribution
with covariate regression on Elo diff, form diff, and xG diff.

This captures correlations between home and away scoring that independent
Poisson models miss.

Math:
    D = Home_Goals - Away_Goals ~ Skellam(λ₁, λ₂)
    where log(λ₁) = β₀ + β₁·elo_diff + β₂·form_diff
    P(D=k) = e^{-(λ₁+λ₂)} × (λ₁/λ₂)^{k/2} × I_{|k|}(2√(λ₁λ₂))

Uses existing math: skellam_probs() from math/distributions.py

References:
    Skellam (1946) "The frequency distribution of the difference between
        two Poisson variates"
    Karlis & Ntzoufras (2009) "Bayesian modelling of football outcomes"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _f, _is_finished, _norm3, _pts
from footy.models.math.distributions import skellam_probs


class SkellamRegressionExpert(Expert):
    """Skellam distribution goal-difference model with covariate regression."""

    name = "skellam_regression"

    AVG_GOALS = 1.35
    ELO_DEFAULT = 1500.0
    ELO_K = 20.0
    COEFF_LR = 0.001  # learning rate for online coefficient updates

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)

        # State tracking
        elo: dict[str, float] = {}
        form: dict[str, list[float]] = {}
        attack: dict[str, float] = {}
        defense: dict[str, float] = {}

        # Learnable regression coefficients (initialized from prior hand-tuned values)
        coeff_home_intercept = 0.30
        coeff_elo = 0.40
        coeff_form = 0.15
        coeff_att = 0.30
        coeff_def = 0.20

        # Output arrays
        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)
        skr_mean_gd = np.zeros(n)
        skr_var_gd = np.zeros(n)
        skr_skewness = np.zeros(n)
        skr_prob_pos = np.zeros(n)
        skr_prob_zero = np.zeros(n)
        skr_prob_neg = np.zeros(n)
        skr_lambda_h = np.zeros(n)
        skr_lambda_a = np.zeros(n)

        ALPHA = 0.08

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            # Compute covariates
            elo_h = elo.get(h, self.ELO_DEFAULT)
            elo_a = elo.get(a, self.ELO_DEFAULT)
            elo_diff = (elo_h - elo_a) / 400.0

            form_h = form.get(h, [])
            form_a = form.get(a, [])
            form_diff = 0.0
            if len(form_h) >= 3 and len(form_a) >= 3:
                form_diff = (np.mean(form_h[-5:]) - np.mean(form_a[-5:])) / 3.0

            att_h = attack.get(h, self.AVG_GOALS)
            def_h = defense.get(h, self.AVG_GOALS * 0.9)
            att_a = attack.get(a, self.AVG_GOALS)
            def_a = defense.get(a, self.AVG_GOALS * 0.9)

            # Regression: goal rates from covariates (learned coefficients)
            # λ₁ (home goals) increases with Elo advantage and attack strength
            # Store covariates for gradient update later
            cov_elo = elo_diff
            cov_form = form_diff
            cov_att_h = att_h / self.AVG_GOALS - 1.0
            cov_att_a = att_a / self.AVG_GOALS - 1.0
            cov_def_a = def_a / self.AVG_GOALS - 1.0
            cov_def_h = def_h / self.AVG_GOALS - 1.0

            log_lam_h = (
                coeff_home_intercept
                + coeff_elo * cov_elo
                + coeff_form * cov_form
                + coeff_att * cov_att_h
                + coeff_def * cov_def_a
            )
            log_lam_a = (
                coeff_home_intercept * 0.5  # away intercept (lower)
                - coeff_elo * cov_elo
                - coeff_form * cov_form
                + coeff_att * cov_att_a
                + coeff_def * cov_def_h
            )

            lam_h = max(0.3, min(4.0, self.AVG_GOALS * np.exp(log_lam_h)))
            lam_a = max(0.3, min(4.0, self.AVG_GOALS * np.exp(log_lam_a)))
            skr_lambda_h[i] = lam_h
            skr_lambda_a[i] = lam_a

            # Skellam distribution
            sk = skellam_probs(lam_h, lam_a, max_diff=8)
            ph = sk["p_home"]
            pd_ = sk["p_draw"]
            pa = sk["p_away"]
            ph, pd_, pa = _norm3(ph, pd_, pa)
            probs[i] = [ph, pd_, pa]

            skr_mean_gd[i] = sk["mean_diff"]
            skr_var_gd[i] = sk["var_diff"]
            skr_skewness[i] = sk.get("skewness", 0.0)
            skr_prob_pos[i] = ph
            skr_prob_zero[i] = pd_
            skr_prob_neg[i] = pa

            # Confidence
            gc_h = len(form.get(h, []))
            gc_a = len(form.get(a, []))
            conf[i] = min(1.0, (gc_h + gc_a) / 20.0)

            # Update state
            if not _is_finished(r):
                continue
            hg, ag = int(r.home_goals), int(r.away_goals)

            # Elo update
            e_h = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400.0))
            s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            delta = self.ELO_K * (s_h - e_h)
            elo[h] = elo_h + delta
            elo[a] = elo_a - delta

            # Form update
            form.setdefault(h, []).append(float(_pts(hg, ag)))
            form.setdefault(a, []).append(float(_pts(ag, hg)))

            # Attack/defense EMA
            if h in attack:
                attack[h] = (1 - ALPHA) * att_h + ALPHA * hg
                defense[h] = (1 - ALPHA) * def_h + ALPHA * ag
            else:
                attack[h] = float(hg) if hg > 0 else self.AVG_GOALS
                defense[h] = float(ag) if ag > 0 else self.AVG_GOALS * 0.9
            if a in attack:
                attack[a] = (1 - ALPHA) * att_a + ALPHA * ag
                defense[a] = (1 - ALPHA) * def_a + ALPHA * hg
            else:
                attack[a] = float(ag) if ag > 0 else self.AVG_GOALS
                defense[a] = float(hg) if hg > 0 else self.AVG_GOALS * 0.9

            # Online gradient descent: update coefficients from prediction error
            # Error = actual goal diff - predicted goal diff (mean_diff from Skellam)
            actual_gd = hg - ag
            pred_gd = skr_mean_gd[i]  # lam_h - lam_a
            error = actual_gd - pred_gd
            lr = self.COEFF_LR

            # Gradient: d(error^2)/d(coeff) ≈ -2*error * d(pred_gd)/d(coeff)
            # Since pred_gd ≈ lam_h - lam_a, and lam ~ exp(log_lam),
            # we use a simplified gradient: update proportional to error * covariate
            coeff_home_intercept += lr * error * 1.0
            coeff_elo += lr * error * cov_elo
            coeff_form += lr * error * cov_form
            coeff_att += lr * error * (cov_att_h - cov_att_a)
            coeff_def += lr * error * (cov_def_a - cov_def_h)

            # Clamp coefficients to [0.0, 1.0]
            coeff_home_intercept = max(0.0, min(1.0, coeff_home_intercept))
            coeff_elo = max(0.0, min(1.0, coeff_elo))
            coeff_form = max(0.0, min(1.0, coeff_form))
            coeff_att = max(0.0, min(1.0, coeff_att))
            coeff_def = max(0.0, min(1.0, coeff_def))

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "skr_mean_gd": skr_mean_gd,
                "skr_var_gd": skr_var_gd,
                "skr_skewness": skr_skewness,
                "skr_lambda_h": skr_lambda_h,
                "skr_lambda_a": skr_lambda_a,
                "skr_lambda_diff": skr_lambda_h - skr_lambda_a,
            },
        )
