"""Expert modules for the v10 council prediction system.

Provides 11 specialised experts that each produce probability distributions
+ features.  Import ``ALL_EXPERTS`` for the default ordered list, or import
individual expert classes as needed.
"""

from footy.models.experts._base import Expert, ExpertResult, _entropy3, _f, _implied, _label, _norm3, _pts, _raw
from footy.models.experts.bayesian_rate import BayesianRateExpert
from footy.models.experts.context import ContextExpert
from footy.models.experts.elo import EloExpert
from footy.models.experts.form import FormExpert
from footy.models.experts.goal_pattern import GoalPatternExpert
from footy.models.experts.h2h import H2HExpert
from footy.models.experts.injury import InjuryAvailabilityExpert
from footy.models.experts.league_table import LeagueTableExpert
from footy.models.experts.market import MarketExpert
from footy.models.experts.momentum import MomentumExpert
from footy.models.experts.poisson_expert import PoissonExpert

ALL_EXPERTS: list[Expert] = [
    EloExpert(),
    MarketExpert(),
    FormExpert(),
    PoissonExpert(),
    H2HExpert(),
    ContextExpert(),
    GoalPatternExpert(),
    LeagueTableExpert(),
    MomentumExpert(),
    BayesianRateExpert(),
    InjuryAvailabilityExpert(),
]

__all__ = [
    "Expert",
    "ExpertResult",
    "ALL_EXPERTS",
    "_f",
    "_raw",
    "_pts",
    "_label",
    "_entropy3",
    "_norm3",
    "_implied",
    "EloExpert",
    "MarketExpert",
    "FormExpert",
    "PoissonExpert",
    "H2HExpert",
    "ContextExpert",
    "GoalPatternExpert",
    "LeagueTableExpert",
    "MomentumExpert",
    "BayesianRateExpert",
    "InjuryAvailabilityExpert",
]
