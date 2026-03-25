"""HMMExpert — Hidden Markov Model for team performance regime detection.

Theory:
    Each team occupies one of 3 latent states at any time:
      - State 0 "dominant":   high attack rate (1.8), low concession rate (0.8)
      - State 1 "balanced":   average attack/defence (1.3 / 1.3)
      - State 2 "vulnerable": low attack rate (0.8), high concession rate (1.8)

    The transition matrix is diagonal-heavy (teams tend to stay in their
    current regime), but goal evidence shifts the posterior via a forward
    algorithm pass after each observed match.

    Key insight: a team in "vulnerable" state that has 3 recent wins is
    *lucky* (regression likely), while a team in "dominant" state with
    the same record is genuinely in form.  This regime information is
    invisible to pure form or Elo models.

Implementation:
    - Lightweight forward algorithm (no scipy/hmmlearn dependency)
    - Poisson emission model per state (goals scored + goals conceded)
    - Per-team posterior state vector updated after each finished match
    - Match probabilities from Monte-Carlo-free analytic Poisson convolution
      weighted by joint state distribution

References:
    - Rabiner (1989) "A Tutorial on Hidden Markov Models"
    - Dixon & Coles (1997) adapted to regime-switching context
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3


# Pre-compute log-factorials for Poisson PMF (goals 0..9)
_LOG_FACT = np.array([math.lgamma(k + 1) for k in range(10)])


def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF for a single (k, lambda) pair.  No scipy needed."""
    if k < 0 or lam <= 0:
        return 0.0
    return math.exp(k * math.log(lam) - lam - _LOG_FACT[min(k, 9)])


class HMMExpert(Expert):
    """Hidden Markov Model expert: detects team performance regimes."""

    name = "hmm"

    # --- HMM parameters ---------------------------------------------------

    N_STATES = 3

    # Emission rates per state: (attack_rate, defense_concession_rate)
    EMISSION_RATES = np.array([
        [1.8, 0.8],   # state 0: dominant
        [1.3, 1.3],   # state 1: balanced
        [0.8, 1.8],   # state 2: vulnerable
    ])

    # Transition matrix — diagonal-heavy (0.80 stay, 0.10 each neighbor)
    TRANS = np.array([
        [0.80, 0.12, 0.08],   # dominant  -> dominant / balanced / vulnerable
        [0.10, 0.80, 0.10],   # balanced  -> ...
        [0.08, 0.12, 0.80],   # vulnerable -> ...
    ])

    # Uniform initial state prior
    INIT_STATE = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    # Number of goals to enumerate for Poisson convolution
    MAX_GOALS = 8

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        # Feature arrays
        hmm_dominant_h = np.zeros(n)
        hmm_dominant_a = np.zeros(n)
        hmm_vulnerable_h = np.zeros(n)
        hmm_vulnerable_a = np.zeros(n)
        hmm_state_mismatch = np.zeros(n)
        hmm_transition_h = np.zeros(n)
        hmm_transition_a = np.zeros(n)

        # Per-team state: posterior state probabilities and previous posterior
        state_post: dict[str, np.ndarray] = {}     # current posterior P(state)
        state_prev: dict[str, np.ndarray] = {}     # previous posterior (for transition detection)
        match_count: dict[str, int] = {}

        # Per-competition parameter learning from data
        _comp_goals: dict[str, list[int]] = {}
        _learned_emission: dict[str, np.ndarray] = {}
        _learned_trans: dict[str, np.ndarray] = {}

        init = self.INIT_STATE.copy()

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            # Select per-competition learned params or defaults
            comp = getattr(r, 'competition', 'default')
            emission = _learned_emission.get(comp, self.EMISSION_RATES)
            trans = _learned_trans.get(comp, self.TRANS)

            # Initialize unseen teams
            for t in (h, a):
                if t not in state_post:
                    state_post[t] = init.copy()
                    state_prev[t] = init.copy()
                    match_count[t] = 0

            post_h = state_post[h]
            post_a = state_post[a]

            # --- Features from current state posteriors ---
            hmm_dominant_h[i] = post_h[0]
            hmm_dominant_a[i] = post_a[0]
            hmm_vulnerable_h[i] = post_h[2]
            hmm_vulnerable_a[i] = post_a[2]
            hmm_state_mismatch[i] = post_h[0] - post_a[0]

            # Transition feature: KL-like measure of how much the posterior
            # changed from previous match (detects regime shifts)
            prev_h = state_prev[h]
            prev_a = state_prev[a]
            hmm_transition_h[i] = float(np.sum(np.abs(post_h - prev_h)))
            hmm_transition_a[i] = float(np.sum(np.abs(post_a - prev_a)))

            # --- Compute P(H, D, A) from state-weighted Poisson convolution ---
            # For each (home_state, away_state) pair, compute expected goals
            # and weight by joint state probability.
            p_h, p_d, p_a = 0.0, 0.0, 0.0

            for sh in range(self.N_STATES):
                for sa in range(self.N_STATES):
                    w = post_h[sh] * post_a[sa]
                    if w < 1e-12:
                        continue

                    # Home team in state sh attacking away team in state sa:
                    #   home expected goals = home_attack_rate * away_concession_factor
                    # We combine: home attacks with sh rate, away defends with sa rate
                    lambda_home = emission[sh, 0] * (emission[sa, 1] / 1.3)
                    lambda_away = emission[sa, 0] * (emission[sh, 1] / 1.3)

                    # Clamp for numerical safety
                    lambda_home = max(0.1, min(5.0, lambda_home))
                    lambda_away = max(0.1, min(5.0, lambda_away))

                    # Enumerate scorelines
                    for hg in range(self.MAX_GOALS):
                        p_hg = _poisson_pmf(hg, lambda_home)
                        for ag in range(self.MAX_GOALS):
                            p_ag = _poisson_pmf(ag, lambda_away)
                            p = w * p_hg * p_ag
                            if hg > ag:
                                p_h += p
                            elif hg == ag:
                                p_d += p
                            else:
                                p_a += p

            probs[i] = _norm3(p_h, p_d, p_a)

            # Confidence: ramps with matches seen, capped at 0.55
            min_mc = min(match_count[h], match_count[a])
            conf[i] = min(0.85, min_mc * 0.015)

            # --- HMM forward update after observing the result ---
            if _is_finished(r):
                hg = int(r.home_goals)
                ag = int(r.away_goals)

                # Collect goals for parameter learning
                _comp_goals.setdefault(comp, []).append(hg)
                _comp_goals[comp].append(ag)
                # Learn params after warmup (200+ goals per competition)
                if len(_comp_goals[comp]) >= 200 and comp not in _learned_emission:
                    from footy.models.parameter_learner import learn_hmm_params
                    params = learn_hmm_params(_comp_goals[comp])
                    rates = params['emission_rates']
                    avg = np.mean(rates)
                    _learned_emission[comp] = np.column_stack([rates, avg * 2 - rates])
                    _learned_trans[comp] = params['transition_matrix']
                    # Use newly learned params immediately
                    emission = _learned_emission[comp]
                    trans = _learned_trans[comp]

                # Update home team: observed (goals_scored=hg, goals_conceded=ag)
                self._forward_update(state_post, state_prev, h, hg, ag, trans, emission)

                # Update away team: observed (goals_scored=ag, goals_conceded=hg)
                self._forward_update(state_post, state_prev, a, ag, hg, trans, emission)

                match_count[h] += 1
                match_count[a] += 1

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "hmm_dominant_h": hmm_dominant_h,
                "hmm_dominant_a": hmm_dominant_a,
                "hmm_vulnerable_h": hmm_vulnerable_h,
                "hmm_vulnerable_a": hmm_vulnerable_a,
                "hmm_state_mismatch": hmm_state_mismatch,
                "hmm_transition_h": hmm_transition_h,
                "hmm_transition_a": hmm_transition_a,
            },
        )

    @staticmethod
    def _forward_update(
        state_post: dict[str, np.ndarray],
        state_prev: dict[str, np.ndarray],
        team: str,
        goals_scored: int,
        goals_conceded: int,
        trans: np.ndarray,
        emission: np.ndarray,
    ) -> None:
        """One-step HMM forward update for a single team.

        1. Predict: propagate current posterior through transition matrix
        2. Update: multiply by emission likelihood for observed goals
        3. Normalize to get new posterior

        This is the filtering step of the forward algorithm (Rabiner 1989).
        """
        prior = state_post[team]

        # Save current posterior as "previous" for transition detection
        state_prev[team] = prior.copy()

        # --- Predict step: P(s_t | s_{t-1}) * P(s_{t-1} | y_{1:t-1}) ---
        predicted = trans.T @ prior   # shape (N_STATES,)

        # --- Update step: P(s_t | y_t) proportional to P(y_t | s_t) * predicted ---
        likelihood = np.zeros(3)
        for s in range(3):
            # Emission: independent Poisson for goals_scored and goals_conceded
            attack_rate = emission[s, 0]
            defense_rate = emission[s, 1]
            p_scored = _poisson_pmf(goals_scored, attack_rate)
            p_conceded = _poisson_pmf(goals_conceded, defense_rate)
            likelihood[s] = p_scored * p_conceded

        posterior = predicted * likelihood

        # Normalize
        total = posterior.sum()
        if total > 1e-15:
            posterior /= total
        else:
            # Degenerate case: reset to uniform
            posterior = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        state_post[team] = posterior
