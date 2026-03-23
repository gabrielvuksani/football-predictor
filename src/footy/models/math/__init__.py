"""Advanced mathematical functions for football prediction.

This package provides production-quality statistical and mathematical tools
for football match outcome prediction, organized into focused modules:

- distributions: Poisson, Skellam, Bivariate Poisson, COM-Poisson, Zero-Inflated
- empirical_bayes: Beta-Binomial shrinkage and league-specific priors
- copulas: Frank, Clayton, Gumbel, and Gaussian copulas for joint distributions
- scoring: Proper scoring rules (RPS, Brier, log loss) and calibration metrics
- simulation: Monte Carlo simulation and Bradley-Terry models
- kalman: Kalman filtering for dynamic team strength
- negative_binomial: Negative Binomial distributions for overdispersed goals
- weibull: Weibull count models
- attention: Recency-attention weighting schemes
- double_poisson: Efron's double Poisson distribution
- bma: Bayesian Model Averaging

For backward compatibility, all functions are re-exported at the package level.
Import from submodules for cleaner namespacing:
    from footy.models.math.distributions import com_poisson_pmf
    from footy.models.math.scoring import ranked_probability_score

Or import directly for legacy compatibility:
    from footy.models.math import com_poisson_pmf, ranked_probability_score
"""

# Distributions
from footy.models.math.distributions import (
    bivariate_poisson_pmf,
    build_bivariate_poisson_matrix,
    build_com_poisson_matrix,
    build_dc_score_matrix,
    build_dibp_score_matrix,
    build_negative_binomial_matrix,
    build_zip_score_matrix,
    com_poisson_pmf,
    dixon_coles_tau,
    estimate_bivariate_lambda3,
    estimate_com_poisson_nu,
    estimate_zip_zero_inflation,
    negative_binomial_pmf,
    skellam_probs,
    zero_inflated_poisson_pmf,
)

# Empirical Bayes
from footy.models.math.empirical_bayes import (
    beta_binomial_shrink,
    estimate_rho_mle,
    league_specific_prior,
)

# Copulas
from footy.models.math.copulas import (
    build_copula_score_matrix,
)

# Scoring
from footy.models.math.scoring import (
    apply_temperature_scaling,
    brier_score,
    expected_calibration_error,
    find_optimal_temperature,
    inv_logit,
    jensen_shannon_divergence,
    kl_divergence,
    log_loss,
    logit,
    logit_space_delta,
    multi_expert_jsd,
    odds_dispersion,
    odds_entropy,
    ranked_probability_score,
    remove_overround,
    rps_from_result,
)

# Simulation
from footy.models.math.simulation import (
    bradley_terry_probs,
    extract_match_probs,
    monte_carlo_simulate,
    platt_scale,
)

# Transforms
from footy.models.math.transforms import (
    adaptive_ewma,
    log_transform,
    schedule_difficulty,
    tanh_transform,
)

__all__ = [
    # Distributions
    "bivariate_poisson_pmf",
    "build_bivariate_poisson_matrix",
    "build_com_poisson_matrix",
    "build_dc_score_matrix",
    "build_negative_binomial_matrix",
    "build_zip_score_matrix",
    "com_poisson_pmf",
    "dixon_coles_tau",
    "negative_binomial_pmf",
    "skellam_probs",
    "zero_inflated_poisson_pmf",
    "build_dibp_score_matrix",
    "estimate_com_poisson_nu",
    "estimate_zip_zero_inflation",
    "estimate_bivariate_lambda3",
    # Empirical Bayes
    "beta_binomial_shrink",
    "estimate_rho_mle",
    "league_specific_prior",
    # Copulas
    "build_copula_score_matrix",
    # Scoring & Calibration
    "apply_temperature_scaling",
    "brier_score",
    "expected_calibration_error",
    "find_optimal_temperature",
    "inv_logit",
    "jensen_shannon_divergence",
    "kl_divergence",
    "log_loss",
    "logit",
    "logit_space_delta",
    "multi_expert_jsd",
    "odds_dispersion",
    "odds_entropy",
    "ranked_probability_score",
    "remove_overround",
    "rps_from_result",
    # Simulation
    "bradley_terry_probs",
    "extract_match_probs",
    "monte_carlo_simulate",
    "platt_scale",
    # Transforms
    "adaptive_ewma",
    "log_transform",
    "schedule_difficulty",
    "tanh_transform",
]
