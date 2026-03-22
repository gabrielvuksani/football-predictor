"""MomentumIndicatorsExpert — Financial momentum indicators applied to football.

Applies RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence),
and Bollinger Bands to team points-per-game series to detect momentum shifts.

References:
    Wilder (1978) "New Concepts in Technical Trading Systems" (RSI)
    Appel (2005) "Technical Analysis: Power Tools for Active Investors" (MACD)
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3, _pts


class MomentumIndicatorsExpert(Expert):
    """Financial momentum indicators (RSI, MACD, Bollinger Bands) on PPG series.

    Tracks per team:
    - Points per game (PPG) series (rolling)
    - RSI-14 (14-game RSI based on PPG changes)
    - MACD (12-game EMA minus 26-game EMA of PPG, signal = 9-game EMA of MACD)
    - Bollinger Bands (20-game SMA +/- 2*std of PPG)
    """

    name = "momentum_indicators"

    # RSI lookback
    RSI_PERIOD = 14
    # MACD parameters
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    # Bollinger Bands parameters
    BB_PERIOD = 20
    BB_STD_MULT = 2.0

    # ---------------------------------------------------------------------------
    # Static indicator helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _ema(values: list[float], span: int) -> float:
        """Compute exponential moving average over *values* with given *span*."""
        if not values:
            return 0.0
        alpha = 2.0 / (span + 1)
        result = values[0]
        for v in values[1:]:
            result = alpha * v + (1.0 - alpha) * result
        return result

    @staticmethod
    def _rsi(ppg_series: list[float], period: int = 14) -> float:
        """Compute RSI from a PPG series.

        RSI = 100 - 100 / (1 + avg_gain / avg_loss)
        Gains/losses are computed on PPG changes (current - previous).
        """
        if len(ppg_series) < period + 1:
            return 50.0  # neutral when insufficient data

        # Compute changes over the lookback window
        recent = ppg_series[-(period + 1):]
        gains: list[float] = []
        losses: list[float] = []
        for j in range(1, len(recent)):
            delta = recent[j] - recent[j - 1]
            if delta > 0:
                gains.append(delta)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(delta))

        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0

        if avg_loss < 1e-12:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def _macd(ppg_series: list[float], fast: int, slow: int, signal: int) -> tuple[float, float, float]:
        """Compute MACD line, signal line, and histogram.

        Returns (macd_line, signal_line, histogram).
        """
        if len(ppg_series) < slow:
            return 0.0, 0.0, 0.0

        # Fast and slow EMAs of PPG
        alpha_f = 2.0 / (fast + 1)
        alpha_s = 2.0 / (slow + 1)
        ema_fast = ppg_series[0]
        ema_slow = ppg_series[0]
        macd_history: list[float] = []

        for v in ppg_series[1:]:
            ema_fast = alpha_f * v + (1.0 - alpha_f) * ema_fast
            ema_slow = alpha_s * v + (1.0 - alpha_s) * ema_slow
            macd_history.append(ema_fast - ema_slow)

        macd_line = ema_fast - ema_slow

        # Signal line: EMA of MACD history
        if len(macd_history) < signal:
            signal_line = macd_line
        else:
            alpha_sig = 2.0 / (signal + 1)
            sig = macd_history[0]
            for m in macd_history[1:]:
                sig = alpha_sig * m + (1.0 - alpha_sig) * sig
            signal_line = sig

        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _bollinger_position(ppg_series: list[float], period: int, mult: float) -> float:
        """Compute position within Bollinger Bands.

        Returns value in [-1, +1]:
            -1 = at lower band
             0 = at SMA
            +1 = at upper band
        Values beyond bands are clipped.
        """
        if len(ppg_series) < period:
            return 0.0

        window = ppg_series[-period:]
        sma = sum(window) / period
        std = float(np.std(window, ddof=0))

        if std < 1e-12:
            return 0.0

        current = ppg_series[-1]
        band_width = mult * std
        position = (current - sma) / band_width  # -1 to +1 when within bands
        return float(np.clip(position, -1.0, 1.0))

    # ---------------------------------------------------------------------------
    # Main compute
    # ---------------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)

        # Per-team PPG series (rolling, one entry per match played)
        team_ppg: dict[str, list[float]] = defaultdict(list)

        # Output arrays
        rsi_h = np.zeros(n)
        rsi_a = np.zeros(n)
        macd_h = np.zeros(n)
        macd_a = np.zeros(n)
        macd_signal_h = np.zeros(n)
        macd_signal_a = np.zeros(n)
        macd_hist_h = np.zeros(n)
        macd_hist_a = np.zeros(n)
        bb_pos_h = np.zeros(n)
        bb_pos_a = np.zeros(n)
        rsi_div = np.zeros(n)
        mom_agree = np.zeros(n)

        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            h_ppg = team_ppg[h]
            a_ppg = team_ppg[a]

            # --- Pre-match feature extraction ---

            # RSI
            h_rsi = self._rsi(h_ppg, self.RSI_PERIOD)
            a_rsi = self._rsi(a_ppg, self.RSI_PERIOD)
            rsi_h[i] = h_rsi
            rsi_a[i] = a_rsi
            rsi_div[i] = h_rsi - a_rsi

            # MACD
            h_macd, h_sig, h_hist = self._macd(h_ppg, self.MACD_FAST, self.MACD_SLOW, self.MACD_SIGNAL)
            a_macd, a_sig, a_hist = self._macd(a_ppg, self.MACD_FAST, self.MACD_SLOW, self.MACD_SIGNAL)
            macd_h[i] = h_macd
            macd_a[i] = a_macd
            macd_signal_h[i] = h_sig
            macd_signal_a[i] = a_sig
            macd_hist_h[i] = h_hist
            macd_hist_a[i] = a_hist

            # Bollinger position
            bb_pos_h[i] = self._bollinger_position(h_ppg, self.BB_PERIOD, self.BB_STD_MULT)
            bb_pos_a[i] = self._bollinger_position(a_ppg, self.BB_PERIOD, self.BB_STD_MULT)

            # Momentum agreement: do RSI and MACD agree on direction?
            # RSI > 50 = bullish, MACD histogram > 0 = bullish
            rsi_bullish_h = h_rsi > 50.0
            macd_bullish_h = h_hist > 0.0
            rsi_bullish_a = a_rsi > 50.0
            macd_bullish_a = a_hist > 0.0
            # Agreement = 1 if both indicators agree for both teams on who is stronger
            home_bullish = rsi_bullish_h and macd_bullish_h
            away_bullish = rsi_bullish_a and macd_bullish_a
            # Agreement means the indicators don't conflict — either home is clearly
            # favoured by both or away is clearly favoured by both
            if (home_bullish and not away_bullish) or (away_bullish and not home_bullish):
                mom_agree[i] = 1.0

            # --- Probability estimation ---
            # Shift from flat prior using RSI divergence + MACD
            rsi_adj = (h_rsi - a_rsi) / 200.0  # scaled, max ±0.5
            rsi_adj = float(np.clip(rsi_adj, -0.08, 0.08))

            macd_adj = float(np.clip((h_hist - a_hist) * 0.5, -0.04, 0.04))
            total_adj = rsi_adj + macd_adj

            p_h = 1.0 / 3.0 + total_adj
            p_a = 1.0 / 3.0 - total_adj
            p_d = 1.0 / 3.0

            probs[i] = _norm3(max(0.05, p_h), max(0.10, p_d), max(0.05, p_a))

            # Confidence scales with data availability
            data_count = len(h_ppg) + len(a_ppg)
            conf[i] = min(1.0, data_count / (2.0 * self.MACD_SLOW))

            # --- Update state (only for finished matches) ---
            if not _is_finished(r):
                continue

            hg, ag = int(r.home_goals), int(r.away_goals)
            h_pts_val = float(_pts(hg, ag)) / 3.0  # PPG: 0, 1/3, or 1
            a_pts_val = float(_pts(ag, hg)) / 3.0
            team_ppg[h].append(h_pts_val)
            team_ppg[a].append(a_pts_val)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "mind_home_rsi": rsi_h,
                "mind_away_rsi": rsi_a,
                "mind_home_macd": macd_h,
                "mind_away_macd": macd_a,
                "mind_home_macd_signal": macd_signal_h,
                "mind_away_macd_signal": macd_signal_a,
                "mind_home_macd_hist": macd_hist_h,
                "mind_away_macd_hist": macd_hist_a,
                "mind_home_bb_position": bb_pos_h,
                "mind_away_bb_position": bb_pos_a,
                "mind_rsi_divergence": rsi_div,
                "mind_momentum_agreement": mom_agree,
            },
        )
