from __future__ import annotations

from typing import Any, cast

from footy.normalize import canonical_team_name
from footy.providers.base import BaseProvider

OPENFOOTBALL_COMPETITIONS: dict[str, str] = {
    "PL": "en.1",
    "ELC": "en.2",
    "BL1": "de.1",
    "PD": "es.1",
    "SA": "it.1",
    "FL1": "fr.1",
    "DED": "nl.1",
    "PPL": "pt.1",
    "TR1": "tr.1",
    "BEL": "be.1",
    "SL": "sco.1",
    "A1": "at.1",
    "GR1": "gr.1",
    "SWS": "ch.1",
    "DK1": "dk.1",
    "SE1": "se.1",
    "NO1": "no.1",
    "PL1": "pl.1",
}


class OpenFootballProvider(BaseProvider):
    name = "openfootball"
    BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"

    def season_url(self, competition: str, season: str) -> str:
        comp = OPENFOOTBALL_COMPETITIONS.get(competition)
        if not comp:
            raise KeyError(f"Unsupported openfootball competition: {competition}")
        return f"{self.BASE_URL}/{season}/{comp}.json"

    def fetch_season_matches(self, competition: str, season: str) -> list[dict[str, Any]]:
        url = self.season_url(competition, season)
        payload = self.fetch_json(url, ttl=6 * 60 * 60)
        payload_dict = cast(dict[str, Any], payload if isinstance(payload, dict) else {})
        matches = cast(list[dict[str, Any]], payload_dict.get("matches", []))
        out: list[dict[str, Any]] = []
        for item in matches:
            score = cast(dict[str, Any], item.get("score") or {})
            ft = cast(list[Any] | None, score.get("ft"))
            home_goals = ft[0] if isinstance(ft, list) and len(ft) == 2 else None
            away_goals = ft[1] if isinstance(ft, list) and len(ft) == 2 else None
            out.append({
                "competition": competition,
                "season": season,
                "round": item.get("round"),
                "utc_date": item.get("date"),
                "home_team": canonical_team_name(item.get("team1")),
                "away_team": canonical_team_name(item.get("team2")),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "status": "FINISHED" if home_goals is not None and away_goals is not None else "SCHEDULED",
                "raw_json": item,
            })
        return out
