"""TransferExpert — models the impact of transfer window activity on team performance.

Teams that made major signings or lost key players show measurable performance
shifts. Uses squad value changes as a proxy for transfer activity, squad depth
for rotation capacity, and age profiles for consistency/energy signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished


def _sf(value: int | float | str | None) -> float:
    """Safe float cast — returns 0.0 for None, NaN, or unparseable values."""
    try:
        if value is None:
            return 0.0
        v = float(value)
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


class TransferExpert(Expert):
    """Capture transfer window impact on match outcomes.

    Tracks three main signals:
    1. **Value change** — large increase in squad value since season start
       implies major signings; decrease implies key departures.
    2. **Squad depth** — deeper squads cope better with fixture congestion.
    3. **Age profile** — very young squads are inconsistent; very old squads
       lack energy. The sweet spot is ~25-28.

    Additionally models an *integration period* disruption factor: newly
    assembled squads need time to gel, so matches within 3 months of a
    transfer window close carry extra uncertainty.
    """

    name = "transfer"

    def __init__(self) -> None:
        super().__init__()
        # Track initial squad values per team per season for value-change calc
        self._season_start_values: dict[tuple[str, str], float] = {}

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        # Feature arrays — all default to 0.0 (graceful degradation)
        tf_value_change_h = np.zeros(n)
        tf_value_change_a = np.zeros(n)
        tf_squad_depth_h = np.zeros(n)
        tf_squad_depth_a = np.zeros(n)
        tf_age_profile_h = np.zeros(n)
        tf_age_profile_a = np.zeros(n)
        tf_integration_period = np.zeros(n)
        tf_depth_advantage = np.zeros(n)
        tf_value_dominance = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            hv = _sf(getattr(r, "mv_home_squad_value", None))
            av = _sf(getattr(r, "mv_away_squad_value", None))
            hd = _sf(getattr(r, "mv_home_squad_depth", None))
            ad = _sf(getattr(r, "mv_away_squad_depth", None))
            ha = _sf(getattr(r, "mv_home_avg_age", None))
            aa = _sf(getattr(r, "mv_away_avg_age", None))

            # Skip rows with no Transfermarkt data
            if hv <= 0 and av <= 0:
                continue

            # --- Identify season and teams ---
            home_team = getattr(r, "home_team", "")
            away_team = getattr(r, "away_team", "")
            season = getattr(r, "season", "")

            # --- Value change relative to season start ---
            if hv > 0:
                key_h = (str(home_team), str(season))
                if key_h not in self._season_start_values:
                    self._season_start_values[key_h] = hv
                start_h = self._season_start_values[key_h]
                if start_h > 0:
                    # Fractional change: +0.2 means 20% increase in squad value
                    tf_value_change_h[i] = (hv - start_h) / start_h

            if av > 0:
                key_a = (str(away_team), str(season))
                if key_a not in self._season_start_values:
                    self._season_start_values[key_a] = av
                start_a = self._season_start_values[key_a]
                if start_a > 0:
                    tf_value_change_a[i] = (av - start_a) / start_a

            # --- Squad depth ---
            # Normalise to a 0-1 scale: typical squads are 20-35 players
            # Divide by 30 to center around 1.0 for a typical squad
            if hd > 0:
                tf_squad_depth_h[i] = hd / 30.0
            if ad > 0:
                tf_squad_depth_a[i] = ad / 30.0

            # Depth advantage: positive means home has more squad depth
            tf_depth_advantage[i] = (hd - ad) / 30.0

            # --- Age profile signal ---
            # Sweet spot is ~26.5. Deviation penalised symmetrically.
            # Signal: negative = bad (too young or too old), near zero = ideal
            _IDEAL_AGE = 26.5
            _AGE_SCALE = 3.0  # years of deviation that maps to -1.0
            if ha > 0:
                tf_age_profile_h[i] = -abs(ha - _IDEAL_AGE) / _AGE_SCALE
            if aa > 0:
                tf_age_profile_a[i] = -abs(aa - _IDEAL_AGE) / _AGE_SCALE

            # --- Integration period ---
            # 1.0 if within 3 months of transfer window close dates:
            #   Summer window closes ~Aug 31  -> Aug, Sep, Oct
            #   Winter window closes ~Jan 31  -> Jan, Feb, Mar
            utc_date = getattr(r, "utc_date", None)
            if utc_date is not None:
                try:
                    month = pd.Timestamp(utc_date).month
                except Exception:
                    month = 0
                if month in (1, 2, 3, 8, 9, 10):
                    tf_integration_period[i] = 1.0

            # --- Value dominance (bounded) ---
            if hv > 0 and av > 0:
                log_ratio = np.log(hv / av)
                tf_value_dominance[i] = float(np.tanh(log_ratio))

                # --- Probability adjustment ---
                # Teams with much higher squad value get a small boost
                shift = float(np.tanh(log_ratio * 0.25)) * 0.12

                p_h = 0.36 + shift + 0.04  # home advantage baked in
                p_d = 0.28 - abs(shift) * 0.25  # draws less likely when mismatch
                p_a = 0.36 - shift

                # Modulate by integration period: new squads add uncertainty
                if tf_integration_period[i] > 0:
                    # Pull probabilities slightly toward uniform during integration
                    damping = 0.85
                    p_h = p_h * damping + (1.0 / 3.0) * (1.0 - damping)
                    p_d = p_d * damping + (1.0 / 3.0) * (1.0 - damping)
                    p_a = p_a * damping + (1.0 / 3.0) * (1.0 - damping)

                total = p_h + p_d + p_a
                if total > 0:
                    probs[i] = [p_h / total, p_d / total, p_a / total]

                # Confidence: higher when value gap is large, reduced during integration
                base_conf = min(0.85, 0.15 + abs(log_ratio) * 0.1)
                if tf_integration_period[i] > 0:
                    base_conf *= 0.7  # less confident during integration window
                conf[i] = base_conf

            # Update season start values only for finished matches
            if _is_finished(r):
                if hv > 0:
                    self._season_start_values[(str(home_team), str(season))] = hv
                if av > 0:
                    self._season_start_values[(str(away_team), str(season))] = av

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "tf_value_change_h": tf_value_change_h,
                "tf_value_change_a": tf_value_change_a,
                "tf_squad_depth_h": tf_squad_depth_h,
                "tf_squad_depth_a": tf_squad_depth_a,
                "tf_age_profile_h": tf_age_profile_h,
                "tf_age_profile_a": tf_age_profile_a,
                "tf_integration_period": tf_integration_period,
                "tf_depth_advantage": tf_depth_advantage,
                "tf_value_dominance": tf_value_dominance,
            },
        )
