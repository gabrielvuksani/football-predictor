"""InjuryAvailabilityExpert — FPL + API-Football + Transfermarkt injury/suspension data."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _f, _norm3


class InjuryAvailabilityExpert(Expert):
    """
    Incorporates player availability from FPL + API-Football + Transfermarkt data.

    v16: Added Transfermarkt injury fallback for all 18 leagues. Previously
    only PL had injury data (FPL). Now reads tm_inj_count_h/a from the
    transfermarkt_injuries table as a universal fallback.
    """
    name = "injury"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)

        # FPL-sourced features
        inj_score_h = np.zeros(n)
        inj_score_a = np.zeros(n)
        squad_str_h = np.ones(n)
        squad_str_a = np.ones(n)
        injured_h = np.zeros(n)
        injured_a = np.zeros(n)
        doubtful_h = np.zeros(n)
        doubtful_a = np.zeros(n)
        suspended_h = np.zeros(n)
        suspended_a = np.zeros(n)
        fpl_fdr_h = np.full(n, 3.0)
        fpl_fdr_a = np.full(n, 3.0)
        fpl_fdr6_h = np.full(n, 3.0)
        fpl_fdr6_a = np.full(n, 3.0)
        # API-Football injury counts
        af_inj_h = np.zeros(n)
        af_inj_a = np.zeros(n)
        # Derived
        inj_diff = np.zeros(n)
        squad_str_diff = np.zeros(n)
        # NEW: combined availability (injuries + suspensions + doubtfuls)
        unavailable_h = np.zeros(n)
        unavailable_a = np.zeros(n)
        # NEW: FDR future schedule asymmetry
        fdr_future_diff = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            # FPL availability (only available for PL)
            _is_h = _f(getattr(r, 'fpl_inj_score_h', None))
            _is_a = _f(getattr(r, 'fpl_inj_score_a', None))
            _ss_h = getattr(r, 'fpl_squad_str_h', None)
            _ss_a = getattr(r, 'fpl_squad_str_a', None)
            _inj_h = _f(getattr(r, 'fpl_injured_h', None))
            _inj_a = _f(getattr(r, 'fpl_injured_a', None))
            _dbt_h = _f(getattr(r, 'fpl_doubtful_h', None))
            _dbt_a = _f(getattr(r, 'fpl_doubtful_a', None))
            # NEW: read suspended data
            _sus_h = _f(getattr(r, 'fpl_suspended_h', None))
            _sus_a = _f(getattr(r, 'fpl_suspended_a', None))

            inj_score_h[i] = _is_h if _is_h else 0.0
            inj_score_a[i] = _is_a if _is_a else 0.0
            squad_str_h[i] = float(_ss_h) if _ss_h is not None and not (isinstance(_ss_h, float) and math.isnan(_ss_h)) else 1.0
            squad_str_a[i] = float(_ss_a) if _ss_a is not None and not (isinstance(_ss_a, float) and math.isnan(_ss_a)) else 1.0
            injured_h[i] = _inj_h
            injured_a[i] = _inj_a
            doubtful_h[i] = _dbt_h
            doubtful_a[i] = _dbt_a
            suspended_h[i] = _sus_h
            suspended_a[i] = _sus_a

            # Combined unavailability: injuries + suspensions + 50% of doubtfuls
            unavailable_h[i] = _inj_h + _sus_h + _dbt_h * 0.5
            unavailable_a[i] = _inj_a + _sus_a + _dbt_a * 0.5

            # FPL fixture difficulty
            _fdr_h = getattr(r, 'fpl_fdr_h', None)
            _fdr_a = getattr(r, 'fpl_fdr_a', None)
            fpl_fdr_h[i] = float(_fdr_h) if _fdr_h is not None and not (isinstance(_fdr_h, float) and math.isnan(_fdr_h)) else 3.0
            fpl_fdr_a[i] = float(_fdr_a) if _fdr_a is not None and not (isinstance(_fdr_a, float) and math.isnan(_fdr_a)) else 3.0

            # NEW: FDR6 — 6-match fixture difficulty lookahead
            _fdr6_h = getattr(r, 'fpl_fdr6_h', None)
            _fdr6_a = getattr(r, 'fpl_fdr6_a', None)
            fpl_fdr6_h[i] = float(_fdr6_h) if _fdr6_h is not None and not (isinstance(_fdr6_h, float) and math.isnan(_fdr6_h)) else 3.0
            fpl_fdr6_a[i] = float(_fdr6_a) if _fdr6_a is not None and not (isinstance(_fdr6_a, float) and math.isnan(_fdr6_a)) else 3.0
            fdr_future_diff[i] = fpl_fdr6_h[i] - fpl_fdr6_a[i]

            # API-Football injury counts
            _af_h = _f(getattr(r, 'af_inj_h', None))
            _af_a = _f(getattr(r, 'af_inj_a', None))
            af_inj_h[i] = _af_h
            af_inj_a[i] = _af_a

            # v16: Transfermarkt injury counts (available for all leagues)
            _tm_h = _f(getattr(r, 'tm_inj_count_h', None))
            _tm_a = _f(getattr(r, 'tm_inj_count_a', None))

            # Use Transfermarkt as fallback when FPL/AF data unavailable
            has_fpl_data = (_is_h > 0 or _is_a > 0 or (_ss_h is not None and not (isinstance(_ss_h, float) and math.isnan(_ss_h))))
            has_af_data = (_af_h > 0 or _af_a > 0)
            if not has_fpl_data and not has_af_data and (_tm_h > 0 or _tm_a > 0):
                # Approximate FPL-scale injury_score from Transfermarkt count
                inj_score_h[i] = min(_tm_h / 10.0, 1.0)
                inj_score_a[i] = min(_tm_a / 10.0, 1.0)
                unavailable_h[i] = _tm_h
                unavailable_a[i] = _tm_a

            # Combined injury differential: FPL score + AF counts + TM counts + suspensions
            total_h = inj_score_h[i] + _af_h * 0.1 + _sus_h * 0.15 + (0 if has_fpl_data else _tm_h * 0.08)
            total_a = inj_score_a[i] + _af_a * 0.1 + _sus_a * 0.15 + (0 if has_fpl_data else _tm_a * 0.08)
            inj_diff[i] = total_h - total_a
            squad_str_diff[i] = squad_str_h[i] - squad_str_a[i]

            # Probability adjustment — INCREASED impact (0.04 per unit, up from 0.02)
            delta = (total_a - total_h) * 0.04 + squad_str_diff[i] * 0.06
            p_h = 0.333 + delta
            p_a = 0.333 - delta
            p_d = 1.0 - p_h - p_a
            probs[i] = list(_norm3(p_h, p_d, p_a))

            # Confidence: based on data availability — higher ceiling
            has_fpl_flag = 1.0 if has_fpl_data else 0.0
            has_af_flag = 1.0 if has_af_data else 0.0
            has_sus = 1.0 if (_sus_h > 0 or _sus_a > 0) else 0.0
            has_tm = 1.0 if (_tm_h > 0 or _tm_a > 0) else 0.0
            conf[i] = min(0.6, has_fpl_flag * 0.3 + has_af_flag * 0.2 + has_sus * 0.1 + has_tm * 0.2)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "inj_score_h": inj_score_h, "inj_score_a": inj_score_a,
                "inj_squad_str_h": squad_str_h, "inj_squad_str_a": squad_str_a,
                "inj_injured_h": injured_h, "inj_injured_a": injured_a,
                "inj_doubtful_h": doubtful_h, "inj_doubtful_a": doubtful_a,
                "inj_suspended_h": suspended_h, "inj_suspended_a": suspended_a,
                "inj_fdr_h": fpl_fdr_h, "inj_fdr_a": fpl_fdr_a,
                "inj_fdr6_h": fpl_fdr6_h, "inj_fdr6_a": fpl_fdr6_a,
                "inj_fdr_future_diff": fdr_future_diff,
                "inj_af_h": af_inj_h, "inj_af_a": af_inj_a,
                "inj_diff": inj_diff,
                "inj_squad_diff": squad_str_diff,
                "inj_unavailable_h": unavailable_h, "inj_unavailable_a": unavailable_a,
            },
        )
