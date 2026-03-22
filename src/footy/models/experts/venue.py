"""VenueExpert — stadium, travel, and venue-specific features."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult


def _sf(value: int | float | str | None) -> float:
    try:
        return float(value) if value is not None else 0.0
    except Exception:
        return 0.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    if lat1 == 0 or lon1 == 0 or lat2 == 0 or lon2 == 0:
        return 0.0
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class VenueExpert(Expert):
    """Venue-based features: capacity, altitude, travel distance, surface type."""

    name = "venue"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        capacity = np.zeros(n)
        altitude = np.zeros(n)
        travel_km = np.zeros(n)
        is_synthetic = np.zeros(n)
        has_venue = np.zeros(n)
        large_stadium = np.zeros(n)
        high_altitude = np.zeros(n)
        long_travel = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            cap = _sf(getattr(r, "venue_capacity", None))
            alt = _sf(getattr(r, "venue_altitude_m", None))
            surface = getattr(r, "venue_surface", None) or ""
            home_lat = _sf(getattr(r, "venue_lat", None))
            home_lon = _sf(getattr(r, "venue_lon", None))
            away_lat = _sf(getattr(r, "away_lat", None))
            away_lon = _sf(getattr(r, "away_lon", None))

            if cap > 0 or home_lat != 0 or alt > 0:
                has_venue[i] = 1.0

            capacity[i] = cap
            altitude[i] = alt
            is_synthetic[i] = 1.0 if "artif" in surface.lower() or "synth" in surface.lower() else 0.0
            large_stadium[i] = 1.0 if cap > 50000 else 0.0
            high_altitude[i] = 1.0 if alt > 800 else 0.0

            if home_lat != 0 and away_lat != 0:
                travel_km[i] = _haversine_km(away_lat, away_lon, home_lat, home_lon)
                long_travel[i] = 1.0 if travel_km[i] > 500 else 0.0

            # Venue-based probability adjustments
            home_boost = 0.0

            # Large stadiums amplify home advantage
            if cap > 60000:
                home_boost += 0.025
            elif cap > 40000:
                home_boost += 0.015

            # High altitude disadvantages visiting teams
            if alt > 1000:
                home_boost += 0.03
            elif alt > 500:
                home_boost += 0.01

            # Long travel fatigues away team
            if travel_km[i] > 1000:
                home_boost += 0.02
            elif travel_km[i] > 500:
                home_boost += 0.01

            # Synthetic pitch familiarity
            if is_synthetic[i]:
                home_boost += 0.015

            p_h = 0.36 + home_boost
            p_d = 0.28
            p_a = 0.36 - home_boost * 0.8

            s = p_h + p_d + p_a
            if s > 0:
                probs[i] = [p_h / s, p_d / s, p_a / s]

            conf[i] = min(0.30, has_venue[i] * 0.08 + (1.0 if travel_km[i] > 0 else 0.0) * 0.07)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "venue_capacity": capacity,
                "venue_altitude_m": altitude,
                "venue_travel_km": travel_km,
                "venue_is_synthetic": is_synthetic,
                "venue_has_data": has_venue,
                "venue_large_stadium": large_stadium,
                "venue_high_altitude": high_altitude,
                "venue_long_travel": long_travel,
            },
        )
