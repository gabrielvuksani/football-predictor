"""Expert modules for the v12 analyst prediction system.

Provides 30+ specialised experts that each produce probability distributions
+ domain-specific features. Import ``ALL_EXPERTS`` for the default ordered
list, or import individual expert classes as needed.

v12 changes:
- Re-enabled 6 previously-excluded experts (Injury, Weather, Referee,
  MarketValue, Venue, XPts) — they now produce features even when enrichment
  data is sparse (graceful degradation instead of flat 33/33/33)
- Added 5 new experts: NegBin, KalmanElo, xGRegression, OrderedProbit,
  SkellamRegression, FBrefAdvanced
"""

from footy.models.experts._base import Expert, ExpertResult, _entropy3, _f, _implied, _is_finished, _label, _norm3, _pts, _raw
from footy.models.experts._league_table_tracker import LeagueTableTracker
from footy.models.experts.bayesian_rate import BayesianRateExpert
from footy.models.experts.bayesian_ss import BayesianStateSpaceExpert
from footy.models.experts.context import ContextExpert
from footy.models.experts.elo import EloExpert
from footy.models.experts.form import FormExpert
from footy.models.experts.glicko2 import Glicko2Expert
from footy.models.experts.goal_pattern import GoalPatternExpert
from footy.models.experts.h2h import H2HExpert
from footy.models.experts.injury import InjuryAvailabilityExpert
from footy.models.experts.league_table import LeagueTableExpert
from footy.models.experts.market import MarketExpert
from footy.models.experts.market_value import MarketValueExpert
from footy.models.experts.momentum import MomentumExpert
from footy.models.experts.network import NetworkStrengthExpert
from footy.models.experts.pi_rating import PiRatingExpert
from footy.models.experts.poisson_expert import PoissonExpert
from footy.models.experts.referee import RefereeExpert
from footy.models.experts.seasonal_pattern import SeasonalPatternExpert
from footy.models.experts.venue import VenueExpert
from footy.models.experts.weather import WeatherExpert
from footy.models.experts.xpts import XPtsExpert
from footy.models.experts.zip_expert import ZIPExpert
from footy.models.experts.copula_expert import CopulaExpert
from footy.models.experts.double_poisson_expert import DoublePoissonExpert
from footy.models.experts.weibull_expert import WeibullExpert
from footy.models.experts.motivation_expert import MotivationExpert
from footy.models.experts.squad_rotation_expert import SquadRotationExpert
from footy.models.experts.momentum_indicators import MomentumIndicatorsExpert

# v12 new experts
from footy.models.experts.negbin_expert import NegBinExpert
from footy.models.experts.kalman_elo_expert import KalmanEloExpert
from footy.models.experts.xg_regression_expert import xGRegressionExpert
from footy.models.experts.ordered_probit_expert import OrderedProbitExpert
from footy.models.experts.skellam_regression_expert import SkellamRegressionExpert
from footy.models.experts.fbref_advanced_expert import FBrefAdvancedExpert

try:
    from footy.models.experts.trueskill import TrueSkillExpert
except ImportError:
    TrueSkillExpert = None  # type: ignore[misc,assignment]

# ── Shared services — instantiated ONCE, passed to experts that need them ──
_shared_league_tracker = LeagueTableTracker()

# ── ALL experts — v12 analyst council (30+) ──
ALL_EXPERTS: list[Expert] = [
    # Rating systems (7)
    EloExpert(),               # Dynamic Elo with momentum — strong baseline
    Glicko2Expert(),           # Glicko-2 with uncertainty tracking
    PiRatingExpert(),          # Separate home/away strength
    NetworkStrengthExpert(),   # Graph-based opponent quality
    KalmanEloExpert(),         # NEW: Kalman-filtered team strength
    BayesianRateExpert(),      # Beta-Binomial shrinkage
    BayesianStateSpaceExpert(),# Bayesian state-space dynamic strengths

    # Goal models (8)
    PoissonExpert(),           # Expected goals via Dixon-Coles + MC + Copula
    NegBinExpert(),            # NEW: Negative Binomial (overdispersion)
    ZIPExpert(),               # Zero-Inflated Poisson
    CopulaExpert(),            # Multi-copula dependency
    WeibullExpert(),           # Weibull count model
    DoublePoissonExpert(),     # Efron's Double Poisson with dispersion
    SkellamRegressionExpert(), # NEW: Skellam goal-difference regression
    OrderedProbitExpert(),     # NEW: Ordinal outcome model

    # Market (2)
    MarketExpert(),            # Betting odds — hardest signal to beat
    MarketValueExpert(),       # RE-ENABLED: Squad market values

    # Form & history (5)
    FormExpert(),              # Recent form with opposition adjustment
    H2HExpert(),               # Head-to-head Bayesian-Dirichlet
    MomentumExpert(),          # EMA crossovers, slope regression
    MomentumIndicatorsExpert(),# RSI, MACD, Bollinger Bands
    GoalPatternExpert(),       # First-goal rate, comeback rate

    # Context (6) — share league table tracker to avoid 3x duplicate computation
    ContextExpert(tracker=_shared_league_tracker),
    MotivationExpert(tracker=_shared_league_tracker),
    SquadRotationExpert(),     # Fixture congestion & fatigue
    InjuryAvailabilityExpert(),# RE-ENABLED: Player injuries/suspensions
    WeatherExpert(),           # RE-ENABLED: Weather impact on match
    RefereeExpert(),           # RE-ENABLED: Referee tendencies

    # Advanced stats (3)
    xGRegressionExpert(),      # NEW: xG regression/mean reversion
    FBrefAdvancedExpert(),     # NEW: Shot quality, discipline, dominance
    XPtsExpert(),              # RE-ENABLED: Expected points analysis

    # Structural (3) — shares league table tracker
    LeagueTableExpert(tracker=_shared_league_tracker),
    SeasonalPatternExpert(),   # Season phase, matchday, cyclical calendar
    VenueExpert(),             # RE-ENABLED: Venue/stadium factors
]

# Add TrueSkill expert if available
if TrueSkillExpert is not None:
    ALL_EXPERTS.append(TrueSkillExpert())

__all__ = [
    "Expert",
    "ExpertResult",
    "ALL_EXPERTS",
    "_f", "_raw", "_pts", "_label", "_entropy3", "_norm3", "_implied", "_is_finished",
    "EloExpert", "MarketExpert", "FormExpert", "PoissonExpert",
    "H2HExpert", "ContextExpert", "GoalPatternExpert", "LeagueTableExpert",
    "MomentumExpert", "BayesianRateExpert", "Glicko2Expert", "PiRatingExpert",
    "InjuryAvailabilityExpert", "WeatherExpert", "RefereeExpert",
    "MarketValueExpert", "VenueExpert", "SeasonalPatternExpert", "XPtsExpert",
    "NetworkStrengthExpert", "ZIPExpert", "BayesianStateSpaceExpert",
    "TrueSkillExpert", "CopulaExpert", "DoublePoissonExpert", "WeibullExpert",
    "MotivationExpert", "SquadRotationExpert", "MomentumIndicatorsExpert",
    "NegBinExpert", "KalmanEloExpert", "xGRegressionExpert",
    "OrderedProbitExpert", "SkellamRegressionExpert", "FBrefAdvancedExpert",
]
