from __future__ import annotations

import codecs
import json
import re
from typing import Any

from footy.providers.base import BaseProvider, ProviderError


class UnderstatProvider(BaseProvider):
    name = "understat"
    BASE_URL = "https://understat.com"
    LEAGUE_MAP = {
        "PL": "EPL",
        "PD": "La_liga",
        "BL1": "Bundesliga",
        "SA": "Serie_A",
        "FL1": "Ligue_1",
        "RU1": "RFPL",
    }

    def fetch_match_shots(self, match_id: int) -> dict[str, Any]:
        html = self.fetch_text(f"{self.BASE_URL}/match/{match_id}", ttl=30 * 60)
        home = self._extract_json_parse_blob(html, "shotsData", "h")
        away = self._extract_json_parse_blob(html, "shotsData", "a")
        return {"home_shots": home or [], "away_shots": away or []}

    def fetch_league_results(self, competition: str, season: int | str) -> list[dict[str, Any]]:
        league = self.LEAGUE_MAP.get(competition, competition)
        html = self.fetch_text(f"{self.BASE_URL}/league/{league}/{season}", ttl=6 * 60 * 60)
        dates = self._extract_json_parse_blob(html, "datesData")
        if not isinstance(dates, list):
            raise ProviderError(f"Understat datesData not found for {competition} {season}")
        return dates

    def extract_match_team_xg(self, match_id: int) -> dict[str, Any]:
        shots = self.fetch_match_shots(match_id)
        home_total = sum(float(s.get("xG") or 0.0) for s in shots["home_shots"])
        away_total = sum(float(s.get("xG") or 0.0) for s in shots["away_shots"])
        return {
            "understat_xg_home": round(home_total, 4),
            "understat_xg_away": round(away_total, 4),
            "understat_shots_home": len(shots["home_shots"]),
            "understat_shots_away": len(shots["away_shots"]),
            "raw_json": shots,
        }

    @staticmethod
    def _extract_json_parse_blob(html: str, var_name: str, nested_key: str | None = None) -> Any:
        patterns = [
            rf"{re.escape(var_name)}\s*=\s*JSON\.parse\('(?P<data>.*?)'\)",
            rf"\('{re.escape(var_name)}'\)\.text\('\s*'\s*\+\s*JSON\.parse\('(?P<data>.*?)'\)",
            rf"{re.escape(var_name)}\s*=\s*(?P<data>\{{.*?\}}|\[.*?\])",
        ]
        match = None
        for pattern in patterns:
            match = re.search(pattern, html, flags=re.DOTALL)
            if match:
                break
        if not match:
            return None
        data = match.group("data")
        if data.startswith("{") or data.startswith("["):
            parsed = json.loads(data)
        else:
            unescaped = codecs.decode(data.encode("utf-8"), "unicode_escape")
            unescaped = unescaped.replace("\\/", "/")
            parsed = json.loads(unescaped)
        if nested_key is not None and isinstance(parsed, dict):
            return parsed.get(nested_key)
        return parsed
