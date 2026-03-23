"""PiRatingExpert — Pi-rating system with goal-difference learning."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3
from footy.models.pi_rating import (
    INITIAL_RATING,
    SCALE_CONSTANT,
    BASE_LEARNING_RATE,
    MIN_LEARNING_RATE,
    LEARNING_DECAY_RATE,
    _learning_rate,
    _expected_goal_diff,
    _ordinal_logistic_probs,
)


class PiRatingExpert(Expert):
    """
    Pi-rating system with:
    - Separate home and away ratings per team
    - Goal-difference based learning (more granular than binary results)
    - Adaptive learning rate (faster early, slower as experience grows)
    - Implicit home advantage (captured in home/away rating difference)
    - Ordinal logistic regression for 3-way probabilities
    """
    name = "pi_rating"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        """Compute Pi-rating features over matches sorted by utc_date ASC."""
        n = len(df)

        # Main state: separate home and away ratings per team
        home_ratings: dict[str, float] = {}
        away_ratings: dict[str, float] = {}
        match_counts: dict[str, int] = {}

        # Output arrays
        out_home_rating_h = np.zeros(n)
        out_away_rating_h = np.zeros(n)
        out_away_rating_a = np.zeros(n)
        out_home_rating_a = np.zeros(n)
        out_expected_gd = np.zeros(n)
        out_ph = np.zeros(n)
        out_pd = np.zeros(n)
        out_pa = np.zeros(n)
        out_conf = np.zeros(n)
        out_rating_diff = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            # Ensure teams exist
            if h not in home_ratings:
                home_ratings[h] = INITIAL_RATING
                away_ratings[h] = INITIAL_RATING
                match_counts[h] = 0
            if a not in home_ratings:
                home_ratings[a] = INITIAL_RATING
                away_ratings[a] = INITIAL_RATING
                match_counts[a] = 0

            # Get current state
            hr_h = home_ratings[h]
            ar_h = away_ratings[h]
            matches_h = match_counts[h]

            hr_a = home_ratings[a]
            ar_a = away_ratings[a]
            matches_a = match_counts[a]

            # Expected goal difference from pi-rating system
            expected_gd = _expected_goal_diff(hr_h, ar_a, SCALE_CONSTANT)

            # Convert to 3-way probabilities
            ph, pd_, pa = _ordinal_logistic_probs(expected_gd, threshold1=-0.5, threshold2=0.5)

            # Confidence from match history: ramps up with experience
            c_h = min(1.0, matches_h / 30.0)
            c_a = min(1.0, matches_a / 30.0)
            conf = (c_h + c_a) / 2.0

            out_home_rating_h[i] = hr_h
            out_away_rating_h[i] = ar_h
            out_away_rating_a[i] = ar_a
            out_home_rating_a[i] = hr_a
            out_expected_gd[i] = expected_gd
            out_ph[i] = ph
            out_pd[i] = pd_
            out_pa[i] = pa
            out_conf[i] = conf
            out_rating_diff[i] = hr_h - ar_a

            # --- Update state AFTER recording pre-match values ---
            # Only update for finished matches (non-NaN goals)
            if _is_finished(r):
                hg, ag = int(r.home_goals), int(r.away_goals)
                actual_gd = hg - ag

                # Compute prediction errors
                error_h = actual_gd - expected_gd
                error_a = -actual_gd - _expected_goal_diff(ar_a, hr_h, SCALE_CONSTANT)

                # Adaptive learning rates
                lr_h = _learning_rate(matches_h)
                lr_a = _learning_rate(matches_a)

                # Update only venue-specific ratings:
                # Home team played at home → update home_ratings[h] only
                # Away team played away → update away_ratings[a] only
                home_ratings[h] = hr_h + lr_h * error_h
                # away_ratings[h] unchanged — team h didn't play away

                # home_ratings[a] unchanged — team a didn't play at home
                away_ratings[a] = ar_a + lr_a * error_a

                # Increment match counts
                match_counts[h] += 1
                match_counts[a] += 1

        return ExpertResult(
            probs=np.column_stack([out_ph, out_pd, out_pa]),
            confidence=out_conf,
            features={
                "pi_home_rating_h": out_home_rating_h,
                "pi_away_rating_h": out_away_rating_h,
                "pi_away_rating_a": out_away_rating_a,
                "pi_home_rating_a": out_home_rating_a,
                "pi_expected_gd": out_expected_gd,
                "pi_rating_diff": out_rating_diff,
            },
        )
