from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from footy.providers.base import BaseProvider


class OpenMeteoProvider(BaseProvider):
    name = "open_meteo"
    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def fetch_kickoff_weather(
        self,
        latitude: float,
        longitude: float,
        kickoff: datetime,
        *,
        timezone_name: str = "UTC",
    ) -> dict[str, Any]:
        kickoff_utc = kickoff.astimezone(timezone.utc)
        payload = self.fetch_json(
            self.BASE_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "timezone": timezone_name,
                "hourly": ",".join([
                    "temperature_2m",
                    "apparent_temperature",
                    "precipitation",
                    "wind_speed_10m",
                    "wind_gusts_10m",
                    "relative_humidity_2m",
                    "cloud_cover",
                    "weather_code",
                ]),
                "daily": "precipitation_sum",
                "forecast_days": 3,
                "past_days": 2,
            },
            ttl=15 * 60,
        )
        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return {"source": self.name, "raw_json": payload}

        def _parse(ts: str) -> datetime:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        nearest_idx = min(range(len(times)), key=lambda i: abs((_parse(times[i]) - kickoff_utc).total_seconds()))
        daily = payload.get("daily", {})
        return {
            "source": self.name,
            "raw_json": payload,
            "kickoff_temperature_c": self._get_series(hourly, "temperature_2m", nearest_idx),
            "kickoff_apparent_temperature_c": self._get_series(hourly, "apparent_temperature", nearest_idx),
            "kickoff_precipitation_mm": self._get_series(hourly, "precipitation", nearest_idx),
            "kickoff_wind_speed_kmh": self._get_series(hourly, "wind_speed_10m", nearest_idx),
            "kickoff_wind_gusts_kmh": self._get_series(hourly, "wind_gusts_10m", nearest_idx),
            "kickoff_humidity_pct": self._get_series(hourly, "relative_humidity_2m", nearest_idx),
            "kickoff_cloud_cover_pct": self._get_series(hourly, "cloud_cover", nearest_idx),
            "kickoff_weather_code": self._get_series(hourly, "weather_code", nearest_idx),
            "rainfall_prev_24h_mm": self._sum_recent(hourly, "precipitation", nearest_idx, 24),
            "rainfall_prev_48h_mm": self._sum_recent(hourly, "precipitation", nearest_idx, 48),
            "daily_precipitation_sum": (daily.get("precipitation_sum") or [None])[0],
        }

    @staticmethod
    def _get_series(hourly: dict[str, Any], name: str, idx: int) -> float | int | None:
        vals = hourly.get(name) or []
        if idx >= len(vals):
            return None
        return vals[idx]

    @staticmethod
    def _sum_recent(hourly: dict[str, Any], name: str, idx: int, hours: int) -> float | None:
        vals = hourly.get(name) or []
        if not vals:
            return None
        lo = max(0, idx - hours)
        return float(sum(v or 0.0 for v in vals[lo:idx + 1]))
