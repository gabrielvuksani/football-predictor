"""Configuration management for football prediction models.

Provides centralized, immutable configuration for all model parameters
currently hardcoded across the codebase. Supports environment variable
overrides for deployment flexibility.

Usage:
    from footy.model_config import ModelConfig
    cfg = ModelConfig()
    print(cfg.elo.k_base)  # 32.0
    print(cfg.poisson.lambda_bounds)  # (0.01, 8.0)

Environment variables:
    Override any parameter via env vars following the pattern:
    FOOTY_<SECTION>_<PARAMETER>=value

    Example:
    $ export FOOTY_ELO_K_BASE=24.0
    $ export FOOTY_POISSON_LAMBDA_BOUNDS="0.05,10.0"
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _get_env_float(name: str, default: float) -> float:
    """Get float value from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _get_env_tuple_float(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    """Get tuple of floats from environment variable (comma-separated)."""
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return tuple(float(x.strip()) for x in v.split(","))
    except (ValueError, AttributeError):
        return default


def _get_env_int(name: str, default: int) -> int:
    """Get int value from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


# ═══════════════════════════════════════════════════════════════════
# Configuration Dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EloConfig:
    """ELO rating system parameters."""
    k_base: float = 32.0  # K factor (rating volatility)
    home_advantage: float = 25.0  # Home team rating bonus
    initial_rating: float = 1500.0  # Default starting rating for new teams
    min_rating: float = 500.0  # Minimum possible rating
    max_rating: float = 3000.0  # Maximum possible rating
    k_factor_reduction: float = 0.0  # K reduction per additional rating points

    def __post_init__(self) -> None:
        """Override from environment if available."""
        # Since frozen=True, use object.__setattr__ for post-init modification
        object.__setattr__(self, 'k_base', _get_env_float('FOOTY_ELO_K_BASE', self.k_base))
        object.__setattr__(self, 'home_advantage', _get_env_float('FOOTY_ELO_HOME_ADVANTAGE', self.home_advantage))
        object.__setattr__(self, 'initial_rating', _get_env_float('FOOTY_ELO_INITIAL_RATING', self.initial_rating))


@dataclass(frozen=True)
class PoissonConfig:
    """Poisson distribution model parameters."""
    lambda_bounds: tuple[float, float] = (0.01, 8.0)  # Valid xG range
    regularization: float = 0.0001  # L2 regularization for lambda estimates
    halflife_days: float = 180.0  # EWMA halflife for strength decay
    min_matches: int = 5  # Minimum matches before predictions
    max_goals: int = 8  # Maximum goals to model in score matrix

    def __post_init__(self) -> None:
        """Override from environment if available."""
        object.__setattr__(
            self, 'lambda_bounds',
            _get_env_tuple_float('FOOTY_POISSON_LAMBDA_BOUNDS', self.lambda_bounds)
        )
        object.__setattr__(
            self, 'regularization',
            _get_env_float('FOOTY_POISSON_REGULARIZATION', self.regularization)
        )
        object.__setattr__(
            self, 'halflife_days',
            _get_env_float('FOOTY_POISSON_HALFLIFE_DAYS', self.halflife_days)
        )
        object.__setattr__(
            self, 'min_matches',
            _get_env_int('FOOTY_POISSON_MIN_MATCHES', self.min_matches)
        )
        object.__setattr__(
            self, 'max_goals',
            _get_env_int('FOOTY_POISSON_MAX_GOALS', self.max_goals)
        )


@dataclass(frozen=True)
class DixonColesConfig:
    """Dixon-Coles model parameters."""
    rho_bounds: tuple[float, float] = (-0.4, 0.0)  # Correlation parameter bounds
    min_low_score_matches: int = 30  # Min data for rho MLE estimation
    rho_default: float = -0.13  # Default rho when insufficient data
    min_matches: int = 5  # Minimum matches for model fit
    max_goals: int = 8  # Maximum goals in score matrix


@dataclass(frozen=True)
class BayesianConfig:
    """Bayesian hierarchical model parameters."""
    home_advantage_mean: float = 0.3  # Prior for log(home advantage)
    home_advantage_var: float = 0.05  # Prior variance
    momentum_decay: float = 0.95  # Exponential decay for momentum
    rue_salvesen_gamma: float = 0.05  # Process noise for Kalman filtering
    league_attack_prior: float = 0.0  # League-level attack strength prior
    league_defence_prior: float = 0.0  # League-level defence strength prior
    league_attack_var: float = 0.25  # League attack variance
    league_defence_var: float = 0.25  # League defence variance


@dataclass(frozen=True)
class CalibrationConfig:
    """Probability calibration parameters."""
    n_bins: int = 10  # Number of bins for calibration curves
    methods: tuple[str, ...] = ("platt", "isotonic")  # Available calibration methods
    cv_folds: int = 5  # Cross-validation folds
    temperature_range: tuple[float, float] = (0.5, 3.0)  # Temperature search bounds
    ece_threshold: float = 0.12  # Expected Calibration Error threshold


@dataclass(frozen=True)
class SelfLearningConfig:
    """Self-learning / drift adaptation parameters."""
    window_size: int = 50  # Matches for performance rolling window
    drift_threshold: float = 0.05  # Performance change threshold
    weight_decay: float = 0.995  # Exponential decay for old predictions
    accuracy_threshold: float = 0.38  # Minimum accuracy to activate self-learning
    update_frequency: int = 10  # Update model every N matches


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward validation parameters."""
    n_folds: int = 5  # Number of walk-forward folds
    test_size_matches: int = 50  # Matches in test set per fold
    embargo_days: int = 3  # Days before test period to exclude (prevent leakage)
    min_train_matches: int = 100  # Minimum training set size


# ═══════════════════════════════════════════════════════════════════
# Master Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CouncilBlendConfig:
    """Council model blend factors and weighting parameters."""
    # Score matrix blend factors (from score models vs council prediction)
    ensemble_blend_factor: float = 0.35  # 15% from score models, 85% from council
    
    # Adaptive blending per competition (when calibrated weights available)
    adaptive_blend_with_calibration: float = 0.25  # With competition calibration
    adaptive_blend_without_calibration: float = 0.08  # Without calibration
    
    # Meta-model validation (HistGradientBoosting)
    validation_fraction: float = 0.12  # Validation set fraction for HistGBM
    
    # Feature interaction blend factors
    feature_interaction_blend: float = 0.10  # Weight for domain-specific features
    
    # Cross-expert agreement weighting
    high_agreement_boost: float = 0.05  # Extra weight when experts strongly agree
    disagreement_penalty: float = 0.02  # Reduce confidence when experts disagree


@dataclass(frozen=True)
class ModelConfig:
    """Master configuration container for all model parameters.

    Frozen dataclass ensures immutability (configuration cannot be
    accidentally modified at runtime). All values can be overridden
    via environment variables.

    Access pattern:
        cfg = ModelConfig()
        cfg.elo.k_base  # 32.0
        cfg.poisson.lambda_bounds  # (0.01, 8.0)
    """
    elo: EloConfig = EloConfig()
    poisson: PoissonConfig = PoissonConfig()
    dixon_coles: DixonColesConfig = DixonColesConfig()
    bayesian: BayesianConfig = BayesianConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    self_learning: SelfLearningConfig = SelfLearningConfig()
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    council_blend: CouncilBlendConfig = CouncilBlendConfig()

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as nested dictionary."""
        return {
            "elo": {k: v for k, v in self.elo.__dict__.items()},
            "poisson": {k: v for k, v in self.poisson.__dict__.items()},
            "dixon_coles": {k: v for k, v in self.dixon_coles.__dict__.items()},
            "bayesian": {k: v for k, v in self.bayesian.__dict__.items()},
            "calibration": {k: v for k, v in self.calibration.__dict__.items()},
            "self_learning": {k: v for k, v in self.self_learning.__dict__.items()},
            "walk_forward": {k: v for k, v in self.walk_forward.__dict__.items()},
            "council_blend": {k: v for k, v in self.council_blend.__dict__.items()},
        }


@functools.lru_cache(maxsize=1)
def get_model_config() -> ModelConfig:
    """Get singleton ModelConfig instance.

    Cached to avoid repeated instantiation. All environment variable
    overrides are read once on first call.

    Returns:
        ModelConfig instance with all parameters

    Example:
        cfg = get_model_config()
        print(cfg.elo.k_base)
    """
    return ModelConfig()
