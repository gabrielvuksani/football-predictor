"""LineupExpert — Lineup strength, rotation detection, and formation analysis.

Analyzes available lineup/formation data to detect:
- Squad rotation (different lineup from usual = weaker team fielded)
- Formation changes (tactical shift signals)
- Key player availability via FPL data
- Squad strength differential

Data sources:
- formation_home/formation_away from match_extras
- FPL availability columns (fpl_available_h/a, fpl_injured_h/a, etc.)
- Squad strength metrics (fpl_squad_str_h/a)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3


class LineupExpert(Expert):
    """Detect rotation, formation changes, and squad strength differentials."""

    name = "lineup"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        lu_formation_change_h = np.zeros(n)
        lu_formation_change_a = np.zeros(n)
        lu_squad_strength_h = np.zeros(n)
        lu_squad_strength_a = np.zeros(n)
        lu_availability_h = np.zeros(n)
        lu_availability_a = np.zeros(n)
        lu_rotation_risk_h = np.zeros(n)
        lu_rotation_risk_a = np.zeros(n)

        has_formation = "formation_home" in df.columns
        has_fpl = "fpl_squad_str_h" in df.columns
        has_avail = "fpl_available_h" in df.columns
        has_injured = "fpl_injured_h" in df.columns

        # Track usual formations per team (mode of last 10)
        team_formations: dict[str, list[str]] = {}

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            for t in (h, a):
                if t not in team_formations:
                    team_formations[t] = []

            # Formation analysis
            if has_formation:
                form_h = getattr(r, "formation_home", None)
                form_a = getattr(r, "formation_away", None)

                if form_h and isinstance(form_h, str) and form_h.strip():
                    recent_h = team_formations[h][-10:]
                    if len(recent_h) >= 3:
                        # Most common formation
                        from collections import Counter
                        usual_h = Counter(recent_h).most_common(1)[0][0]
                        lu_formation_change_h[i] = 0.0 if form_h == usual_h else 1.0

                if form_a and isinstance(form_a, str) and form_a.strip():
                    recent_a = team_formations[a][-10:]
                    if len(recent_a) >= 3:
                        from collections import Counter
                        usual_a = Counter(recent_a).most_common(1)[0][0]
                        lu_formation_change_a[i] = 0.0 if form_a == usual_a else 1.0

            # FPL squad strength
            if has_fpl:
                raw_h = getattr(r, "fpl_squad_str_h", None)
                raw_a = getattr(r, "fpl_squad_str_a", None)
                if raw_h is not None and not (isinstance(raw_h, float) and np.isnan(raw_h)):
                    lu_squad_strength_h[i] = float(raw_h)
                if raw_a is not None and not (isinstance(raw_a, float) and np.isnan(raw_a)):
                    lu_squad_strength_a[i] = float(raw_a)

            # FPL availability (higher = more available players)
            if has_avail:
                raw_h = getattr(r, "fpl_available_h", None)
                raw_a = getattr(r, "fpl_available_a", None)
                if raw_h is not None and not (isinstance(raw_h, float) and np.isnan(raw_h)):
                    lu_availability_h[i] = float(raw_h) / 25.0  # normalize
                if raw_a is not None and not (isinstance(raw_a, float) and np.isnan(raw_a)):
                    lu_availability_a[i] = float(raw_a) / 25.0

            # Injured count (higher = more rotation risk)
            if has_injured:
                inj_h = getattr(r, "fpl_injured_h", None)
                inj_a = getattr(r, "fpl_injured_a", None)
                inj_h_val = float(inj_h) if inj_h is not None and not (isinstance(inj_h, float) and np.isnan(inj_h)) else 0.0
                inj_a_val = float(inj_a) if inj_a is not None and not (isinstance(inj_a, float) and np.isnan(inj_a)) else 0.0
                lu_rotation_risk_h[i] = min(1.0, inj_h_val / 5.0 + lu_formation_change_h[i] * 0.3)
                lu_rotation_risk_a[i] = min(1.0, inj_a_val / 5.0 + lu_formation_change_a[i] * 0.3)

            # Probability adjustments based on squad strength differential
            str_diff = lu_squad_strength_h[i] - lu_squad_strength_a[i]
            if abs(str_diff) > 0.05:
                adj = np.clip(str_diff * 0.10, -0.08, 0.08)
                p_h = 0.36 + adj
                p_a = 0.36 - adj
                p_d = 0.28
                s = p_h + p_d + p_a
                if s > 0:
                    probs[i] = _norm3(p_h / s, p_d / s, p_a / s)
                conf[i] = min(0.85, abs(str_diff) * 2.0)

            # Update formation history
            if _is_finished(r) and has_formation:
                form_h = getattr(r, "formation_home", None)
                form_a = getattr(r, "formation_away", None)
                if form_h and isinstance(form_h, str) and form_h.strip():
                    team_formations[h].append(form_h)
                if form_a and isinstance(form_a, str) and form_a.strip():
                    team_formations[a].append(form_a)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "lu_formation_change_h": lu_formation_change_h,
                "lu_formation_change_a": lu_formation_change_a,
                "lu_squad_strength_h": lu_squad_strength_h,
                "lu_squad_strength_a": lu_squad_strength_a,
                "lu_availability_h": lu_availability_h,
                "lu_availability_a": lu_availability_a,
                "lu_rotation_risk_h": lu_rotation_risk_h,
                "lu_rotation_risk_a": lu_rotation_risk_a,
            },
        )
