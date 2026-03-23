"""NewsSentimentExpert — News and media sentiment impact on team performance.

Tracks recent news sentiment per team. Negative news spikes (manager sacking,
financial problems, player scandal) correlate with poor performance. High news
volume signals uncertainty.

Data sources:
- GDELT news data (news table in DB)
- news_tone / news_count columns if joined to matches
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult


class NewsSentimentExpert(Expert):
    """Detect disruption signals from news sentiment."""

    name = "news_sentiment"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        ns_tone_h = np.zeros(n)
        ns_tone_a = np.zeros(n)
        ns_tone_diff = np.zeros(n)
        ns_volume_h = np.zeros(n)
        ns_volume_a = np.zeros(n)
        ns_disruption_h = np.zeros(n)
        ns_disruption_a = np.zeros(n)

        has_tone = "news_tone_h" in df.columns and "news_tone_a" in df.columns
        has_volume = "news_count_h" in df.columns and "news_count_a" in df.columns

        for i, r in enumerate(df.itertuples(index=False)):
            tone_h = 0.0
            tone_a = 0.0
            vol_h = 0.0
            vol_a = 0.0

            if has_tone:
                raw_h = getattr(r, "news_tone_h", None)
                raw_a = getattr(r, "news_tone_a", None)
                if raw_h is not None and not (isinstance(raw_h, float) and np.isnan(raw_h)):
                    tone_h = float(raw_h)
                if raw_a is not None and not (isinstance(raw_a, float) and np.isnan(raw_a)):
                    tone_a = float(raw_a)

            if has_volume:
                raw_vh = getattr(r, "news_count_h", None)
                raw_va = getattr(r, "news_count_a", None)
                if raw_vh is not None and not (isinstance(raw_vh, float) and np.isnan(raw_vh)):
                    vol_h = float(raw_vh)
                if raw_va is not None and not (isinstance(raw_va, float) and np.isnan(raw_va)):
                    vol_a = float(raw_va)

            ns_tone_h[i] = tone_h
            ns_tone_a[i] = tone_a
            ns_tone_diff[i] = tone_h - tone_a
            ns_volume_h[i] = vol_h
            ns_volume_a[i] = vol_a

            # Disruption = high volume AND negative tone
            ns_disruption_h[i] = max(0.0, -tone_h) * min(vol_h / 50.0, 1.0) if vol_h > 5 else 0.0
            ns_disruption_a[i] = max(0.0, -tone_a) * min(vol_a / 50.0, 1.0) if vol_a > 5 else 0.0

            # Probability adjustment: disrupted team performs worse
            if ns_disruption_h[i] > 0.1 or ns_disruption_a[i] > 0.1:
                adj = (ns_disruption_a[i] - ns_disruption_h[i]) * 0.08
                p_h = 0.36 + adj
                p_a = 0.36 - adj
                p_d = 0.28
                s = p_h + p_d + p_a
                if s > 0:
                    probs[i] = [p_h / s, p_d / s, p_a / s]
                conf[i] = min(0.85, abs(adj) * 5.0)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "ns_tone_h": ns_tone_h,
                "ns_tone_a": ns_tone_a,
                "ns_tone_diff": ns_tone_diff,
                "ns_volume_h": ns_volume_h,
                "ns_volume_a": ns_volume_a,
                "ns_disruption_h": ns_disruption_h,
                "ns_disruption_a": ns_disruption_a,
            },
        )
