"""BettingMovementExpert — Betting line movement analysis.

Tracks opening-to-closing odds movements which signal:
- Smart money (sharp bettors moving the line)
- New information (injury news, lineup changes)
- Market overreaction (steam moves that reverse)

The MarketExpert uses Shin-adjusted closing odds for probability.
This expert focuses on the MOVEMENT from open to close — the delta
contains unique information about late-breaking factors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _norm3


class BettingMovementExpert(Expert):
    """Analyze betting line movements for smart money signals."""

    name = "betting_movement"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        bm_move_h = np.zeros(n)
        bm_move_d = np.zeros(n)
        bm_move_a = np.zeros(n)
        bm_magnitude = np.zeros(n)
        bm_direction_change = np.zeros(n)
        bm_late_shift = np.zeros(n)
        bm_sharp_signal = np.zeros(n)

        has_open = "b365h" in df.columns
        has_close = "b365ch" in df.columns
        has_pin = "psh" in df.columns
        has_avg = "avgh" in df.columns

        for i, r in enumerate(df.itertuples(index=False)):
            # Get opening odds (B365 primary)
            open_h = getattr(r, "b365h", None) if has_open else None
            open_d = getattr(r, "b365d", None) if has_open else None
            open_a = getattr(r, "b365a", None) if has_open else None

            # Get closing odds
            close_h = getattr(r, "b365ch", None) if has_close else None
            close_d = getattr(r, "b365cd", None) if has_close else None
            close_a = getattr(r, "b365ca", None) if has_close else None

            # Get Pinnacle odds (sharpest bookmaker)
            pin_h = getattr(r, "psh", None) if has_pin else None
            pin_d = getattr(r, "psd", None) if has_pin else None
            pin_a = getattr(r, "psa", None) if has_pin else None

            # Get average market odds
            avg_h = getattr(r, "avgh", None) if has_avg else None
            avg_d = getattr(r, "avgd", None) if has_avg else None
            avg_a = getattr(r, "avga", None) if has_avg else None

            def _safe(v):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return 0.0
                return float(v)

            oh, od, oa = _safe(open_h), _safe(open_d), _safe(open_a)
            ch, cd, ca = _safe(close_h), _safe(close_d), _safe(close_a)
            ph, pd_, pa = _safe(pin_h), _safe(pin_d), _safe(pin_a)
            ah, ad, aa = _safe(avg_h), _safe(avg_d), _safe(avg_a)

            # Use closing if available, else use opening as "close"
            if ch == 0 and oh > 0:
                ch, cd, ca = oh, od, oa

            # Need both open and close to compute movement
            if oh > 1.0 and ch > 1.0:
                # Convert to implied probabilities
                open_total = (1/oh + 1/od + 1/oa) if od > 0 and oa > 0 else 1.0
                close_total = (1/ch + 1/cd + 1/ca) if cd > 0 and ca > 0 else 1.0

                if open_total > 0 and close_total > 0:
                    op_h = (1/oh) / open_total
                    op_d = (1/od) / open_total if od > 0 else 0.33
                    op_a = (1/oa) / open_total if oa > 0 else 0.33

                    cl_h = (1/ch) / close_total
                    cl_d = (1/cd) / close_total if cd > 0 else 0.33
                    cl_a = (1/ca) / close_total if ca > 0 else 0.33

                    # Movement: positive = market moved toward that outcome
                    bm_move_h[i] = cl_h - op_h
                    bm_move_d[i] = cl_d - op_d
                    bm_move_a[i] = cl_a - op_a

                    # Magnitude of total movement
                    bm_magnitude[i] = abs(bm_move_h[i]) + abs(bm_move_d[i]) + abs(bm_move_a[i])

                    # Did the favourite change?
                    open_fav = np.argmax([op_h, op_d, op_a])
                    close_fav = np.argmax([cl_h, cl_d, cl_a])
                    bm_direction_change[i] = 1.0 if open_fav != close_fav else 0.0

                    # Late shift: closing vs average (if avg available)
                    if ah > 1.0 and ad > 1.0 and aa > 1.0:
                        avg_total = (1/ah + 1/ad + 1/aa)
                        if avg_total > 0:
                            av_h = (1/ah) / avg_total
                            bm_late_shift[i] = cl_h - av_h

                    # Sharp signal: Pinnacle vs average disagreement
                    if ph > 1.0 and ah > 1.0 and pd_ > 1.0 and ad > 1.0:
                        pin_total = (1/ph + 1/pd_ + 1/pa) if pa > 0 else 1.0
                        avg_total2 = (1/ah + 1/ad + 1/aa) if aa > 0 else 1.0
                        if pin_total > 0 and avg_total2 > 0:
                            pin_imp_h = (1/ph) / pin_total
                            avg_imp_h = (1/ah) / avg_total2
                            bm_sharp_signal[i] = pin_imp_h - avg_imp_h

                    # Probability adjustment based on movement
                    move = bm_move_h[i]
                    if abs(move) > 0.02:
                        p_h = 0.36 + move * 0.5
                        p_a = 0.36 - move * 0.5
                        p_d = 0.28
                        s = p_h + p_d + p_a
                        if s > 0:
                            probs[i] = _norm3(p_h / s, p_d / s, p_a / s)
                        conf[i] = min(0.85, abs(move) * 8.0 + bm_magnitude[i] * 2.0)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "bm_move_h": bm_move_h,
                "bm_move_d": bm_move_d,
                "bm_move_a": bm_move_a,
                "bm_magnitude": bm_magnitude,
                "bm_direction_change": bm_direction_change,
                "bm_late_shift": bm_late_shift,
                "bm_sharp_signal": bm_sharp_signal,
            },
        )
