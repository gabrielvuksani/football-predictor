"""WeatherExpert — venue weather and pitch-condition context from Open-Meteo."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult


def _safe_float(value: int | float | str | None) -> float:
    try:
        return float(value) if value is not None else 0.0
    except Exception:
        return 0.0


def _normalize3(a: float, b: float, c: float) -> tuple[float, float, float]:
    s = a + b + c
    if s <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (a / s, b / s, c / s)


class WeatherExpert(Expert):
    """Model the effect of weather on outcome shape.

    Weather usually matters more for variance and draw rate than for raw team strength.
    This expert therefore mostly adjusts draw probability and slightly dampens/extents
    home edge in extreme conditions.
    """

    name = "weather"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        temp_c = np.zeros(n)
        precipitation = np.zeros(n)
        wind_kmh = np.zeros(n)
        wind_gusts = np.zeros(n)
        humidity = np.zeros(n)
        cloud_cover = np.zeros(n)
        rain_24h = np.zeros(n)
        rain_48h = np.zeros(n)
        extreme_temp = np.zeros(n)
        bad_weather = np.zeros(n)
        pitch_heavy = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            temp_c[i] = _safe_float(getattr(r, "wx_temp_c", None))
            precipitation[i] = _safe_float(getattr(r, "wx_precip_mm", None))
            wind_kmh[i] = _safe_float(getattr(r, "wx_wind_kmh", None))
            wind_gusts[i] = _safe_float(getattr(r, "wx_gusts_kmh", None))
            humidity[i] = _safe_float(getattr(r, "wx_humidity_pct", None))
            cloud_cover[i] = _safe_float(getattr(r, "wx_cloud_cover_pct", None))
            rain_24h[i] = _safe_float(getattr(r, "wx_rain_24h_mm", None))
            rain_48h[i] = _safe_float(getattr(r, "wx_rain_48h_mm", None))

            extreme_temp[i] = 1.0 if temp_c[i] < 0 or temp_c[i] > 30 else 0.0
            bad_weather[i] = 1.0 if precipitation[i] > 2.5 or wind_kmh[i] > 28 or wind_gusts[i] > 40 else 0.0
            pitch_heavy[i] = 1.0 if rain_24h[i] > 8 or rain_48h[i] > 15 else 0.0

            draw_boost = 0.0
            home_delta = 0.0

            # High wind and heavy rain reduce shot quality and increase chaos/draw likelihood.
            draw_boost += min(0.08, precipitation[i] * 0.01)
            draw_boost += min(0.07, max(0.0, wind_kmh[i] - 20.0) * 0.0035)
            draw_boost += pitch_heavy[i] * 0.03
            draw_boost += extreme_temp[i] * 0.015

            # Home familiarity matters a touch more on heavy surfaces / harsh conditions.
            home_delta += pitch_heavy[i] * 0.02
            home_delta += bad_weather[i] * 0.01

            p_h = 0.36 + home_delta
            p_d = 0.28 + draw_boost
            p_a = 0.36 - home_delta * 0.7
            probs[i] = _normalize3(p_h, p_d, p_a)

            observed = sum(
                1.0
                for val in (temp_c[i], precipitation[i], wind_kmh[i], humidity[i], rain_24h[i])
                if val > 0
            )
            conf[i] = min(0.45, observed * 0.08 + bad_weather[i] * 0.05)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "wx_temp_c": temp_c,
                "wx_precip_mm": precipitation,
                "wx_wind_kmh": wind_kmh,
                "wx_gusts_kmh": wind_gusts,
                "wx_humidity_pct": humidity,
                "wx_cloud_cover_pct": cloud_cover,
                "wx_rain_24h_mm": rain_24h,
                "wx_rain_48h_mm": rain_48h,
                "wx_extreme_temp": extreme_temp,
                "wx_bad_weather": bad_weather,
                "wx_pitch_heavy": pitch_heavy,
            },
        )
