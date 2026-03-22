from __future__ import annotations

from datetime import date, timedelta
from typing import Any, cast

from footy.normalize import canonical_team_name
from footy.providers.base import BaseProvider

SOFASCORE_COMPETITIONS: dict[str, tuple[str, ...]] = {
    "PL": ("premier league",),
    "ELC": ("championship",),
    "PD": ("laliga", "la liga"),
    "SA": ("serie a",),
    "BL1": ("bundesliga",),
    "FL1": ("ligue 1",),
    "DED": ("eredivisie",),
    "PPL": ("liga portugal", "liga portugal betclic"),
    "TR1": ("super lig", "süper lig"),
    "BEL": ("pro league",),
    "SL": ("premiership", "scottish premiership"),
    "A1": ("bundesliga, championship round", "austrian bundesliga"),
    "GR1": ("super league",),
    "SWS": ("super league", "swiss super league"),
    "DK1": ("superliga",),
    "SE1": ("allsvenskan",),
    "NO1": ("eliteserien",),
    "PL1": ("ekstraklasa",),
}


class SofaScoreProvider(BaseProvider):
    name = "sofascore"
    BASE_URL = "https://api.sofascore.com/api/v1"

    def fetch_scheduled_events(self, target_date: date) -> list[dict[str, Any]]:
        payload = self.fetch_json(
            f"{self.BASE_URL}/sport/football/scheduled-events/{target_date.isoformat()}",
            ttl=5 * 60,
        )
        payload_dict = cast(dict[str, Any], payload if isinstance(payload, dict) else {})
        return cast(list[dict[str, Any]], payload_dict.get("events", []))

    def fetch_event(self, event_id: int) -> dict[str, Any]:
        payload = self.fetch_json(f"{self.BASE_URL}/event/{event_id}", ttl=10 * 60)
        return cast(dict[str, Any], payload.get("event", payload) if isinstance(payload, dict) else {})

    def fetch_window(self, start_date: date, end_date: date) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cursor = start_date
        while cursor <= end_date:
            out.extend(self.fetch_scheduled_events(cursor))
            cursor += timedelta(days=1)
        return out

    def normalize_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        home_team = cast(dict[str, Any], event.get("homeTeam") or {})
        away_team = cast(dict[str, Any], event.get("awayTeam") or {})
        tournament = cast(dict[str, Any], event.get("tournament") or {})
        unique_tournament = cast(dict[str, Any], tournament.get("uniqueTournament") or {})
        season = cast(dict[str, Any], event.get("season") or {})
        slug = str(unique_tournament.get("slug") or "").strip().lower()
        name = str(unique_tournament.get("name") or tournament.get("name") or "").strip().lower()
        competition = self._map_competition(slug, name)
        if not competition:
            return None

        start_timestamp = event.get("startTimestamp")
        utc_date = None
        if start_timestamp is not None:
            try:
                import datetime as _dt
                utc_date = _dt.datetime.fromtimestamp(int(start_timestamp), tz=_dt.timezone.utc)
            except Exception:
                utc_date = None

        home_score = cast(dict[str, Any], event.get("homeScore") or {})
        away_score = cast(dict[str, Any], event.get("awayScore") or {})
        status_obj = cast(dict[str, Any], event.get("status") or {})
        status_desc = str(status_obj.get("description") or status_obj.get("type") or "SCHEDULED").upper()
        finished = status_desc in {"FT", "FINISHED"}

        return {
            "match_id": int(event.get("id") or 0),
            "provider": self.name,
            "competition": competition,
            "season": season.get("year"),
            "utc_date": utc_date,
            "status": "FINISHED" if finished else "SCHEDULED",
            "home_team": canonical_team_name(home_team.get("name")),
            "away_team": canonical_team_name(away_team.get("name")),
            "home_goals": home_score.get("current") if finished else None,
            "away_goals": away_score.get("current") if finished else None,
            "raw_json": event,
            "venue_name": cast(dict[str, Any], event.get("venue") or {}).get("stadium", {}).get("name") if isinstance(event.get("venue"), dict) else None,
        }

    @staticmethod
    def _map_competition(slug: str, name: str) -> str | None:
        for code, candidates in SOFASCORE_COMPETITIONS.items():
            if slug in candidates or name in candidates:
                return code
        if name == "premier league":
            return "PL"
        return None
