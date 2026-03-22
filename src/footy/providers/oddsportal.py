"""OddsPortal Scraper — historical and live odds from 40+ bookmakers.

Free, no API key. Scraping-based with polite delays.
Replaces The Odds API for zero-cost operation.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from bs4 import BeautifulSoup

from footy.providers.base import BaseProvider, ProviderError

log = logging.getLogger(__name__)

_BASE_URL = "https://www.oddsportal.com"
_REQUEST_DELAY = 3.0

# Competition mappings
ODDSPORTAL_LEAGUES: dict[str, dict[str, str]] = {
    "PL":  {"country": "england", "slug": "premier-league"},
    "PD":  {"country": "spain", "slug": "laliga"},
    "SA":  {"country": "italy", "slug": "serie-a"},
    "BL1": {"country": "germany", "slug": "bundesliga"},
    "FL1": {"country": "france", "slug": "ligue-1"},
    "DED": {"country": "netherlands", "slug": "eredivisie"},
    "PPL": {"country": "portugal", "slug": "liga-portugal"},
    "ELC": {"country": "england", "slug": "championship"},
    "TR1": {"country": "turkey", "slug": "super-lig"},
    "BEL": {"country": "belgium", "slug": "first-division-a"},
    "SL":  {"country": "scotland", "slug": "premiership"},
    "A1":  {"country": "austria", "slug": "tipico-bundesliga"},
    "GR1": {"country": "greece", "slug": "super-league"},
    "SWS": {"country": "switzerland", "slug": "super-league"},
    "DK1": {"country": "denmark", "slug": "superliga"},
    "SE1": {"country": "sweden", "slug": "allsvenskan"},
    "NO1": {"country": "norway", "slug": "eliteserien"},
    "PL1": {"country": "poland", "slug": "ekstraklasa"},
}


class OddsPortalProvider(BaseProvider):
    """Scrapes OddsPortal for odds from multiple bookmakers."""

    name = "oddsportal"

    def __init__(self, *, enabled: bool = True):
        super().__init__(
            enabled=enabled,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.oddsportal.com/",
            },
        )
        self._last_request_time: float = 0.0

    def _polite_delay(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < _REQUEST_DELAY:
            time.sleep(_REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _parse_odds(self, text: str) -> float:
        """Parse decimal odds from text."""
        try:
            text = text.strip()
            if not text or text == "-":
                return 0.0
            return float(text)
        except ValueError:
            return 0.0

    def fetch_upcoming_odds(self, competition: str) -> list[dict[str, Any]]:
        """Fetch upcoming match odds for a competition.

        Returns list of dicts with match info and 1X2 odds from multiple bookmakers.
        """
        if not self.enabled:
            raise ProviderError("OddsPortal provider is disabled")

        league_info = ODDSPORTAL_LEAGUES.get(competition)
        if not league_info:
            log.warning("OddsPortal: no mapping for competition %s", competition)
            return []

        url = f"{_BASE_URL}/football/{league_info['country']}/{league_info['slug']}/"
        try:
            self._polite_delay()
            html = self.fetch_text(url, ttl=1800)  # cache 30 minutes
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            log.warning("OddsPortal: failed to fetch odds for %s: %s", competition, exc)
            return []

        results: list[dict[str, Any]] = []

        # OddsPortal uses various table/div structures
        # Look for match rows with odds data
        match_rows = soup.find_all("div", class_=re.compile(r"eventRow"))
        if not match_rows:
            # Fallback: try table-based layout
            match_rows = soup.find_all("tr", class_=re.compile(r"deactivate"))

        for row in match_rows:
            try:
                # Extract team names
                teams = row.find_all("a", class_=re.compile(r"participant-name"))
                if len(teams) < 2:
                    # Try alternative selectors
                    teams = row.find_all("span", class_=re.compile(r"team"))
                if len(teams) < 2:
                    continue

                home_team = teams[0].text.strip()
                away_team = teams[1].text.strip()

                if not home_team or not away_team:
                    continue

                # Extract odds
                odds_cells = row.find_all("span", class_=re.compile(r"odds-val"))
                if not odds_cells:
                    odds_cells = row.find_all("td", class_=re.compile(r"odds"))

                odds_h = self._parse_odds(odds_cells[0].text) if len(odds_cells) > 0 else 0.0
                odds_d = self._parse_odds(odds_cells[1].text) if len(odds_cells) > 1 else 0.0
                odds_a = self._parse_odds(odds_cells[2].text) if len(odds_cells) > 2 else 0.0

                # Extract date/time if available
                date_elem = row.find("span", class_=re.compile(r"date|time"))
                match_date = date_elem.text.strip() if date_elem else ""

                match_data: dict[str, Any] = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "competition": competition,
                    "date": match_date,
                    "odds_h": odds_h,
                    "odds_d": odds_d,
                    "odds_a": odds_a,
                    "source": "oddsportal",
                }

                # Try to extract over/under odds
                ou_elem = row.find("span", {"data-market": "ou"})
                if ou_elem:
                    match_data["odds_over25"] = self._parse_odds(ou_elem.get("data-over", "0"))  # type: ignore[arg-type]
                    match_data["odds_under25"] = self._parse_odds(ou_elem.get("data-under", "0"))  # type: ignore[arg-type]

                results.append(match_data)
            except Exception as exc:
                log.debug("OddsPortal: failed to parse match row: %s", exc)
                continue

        return results

    def fetch_historical_odds(self, competition: str, season: str = "2025-2026") -> list[dict[str, Any]]:
        """Fetch historical opening/closing odds for completed matches."""
        if not self.enabled:
            raise ProviderError("OddsPortal provider is disabled")

        league_info = ODDSPORTAL_LEAGUES.get(competition)
        if not league_info:
            return []

        url = f"{_BASE_URL}/football/{league_info['country']}/{league_info['slug']}-{season}/results/"
        try:
            self._polite_delay()
            html = self.fetch_text(url, ttl=3600)
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            log.warning("OddsPortal: failed to fetch historical odds for %s: %s", competition, exc)
            return []

        results: list[dict[str, Any]] = []
        match_rows = soup.find_all("div", class_=re.compile(r"eventRow"))
        if not match_rows:
            match_rows = soup.find_all("tr", class_=re.compile(r"deactivate"))

        for row in match_rows:
            try:
                teams = row.find_all("a", class_=re.compile(r"participant-name"))
                if len(teams) < 2:
                    teams = row.find_all("span", class_=re.compile(r"team"))
                if len(teams) < 2:
                    continue

                home_team = teams[0].text.strip()
                away_team = teams[1].text.strip()

                # Extract score
                score_elem = row.find("span", class_=re.compile(r"score"))
                score_text = score_elem.text.strip() if score_elem else ""
                home_goals, away_goals = None, None
                score_match = re.match(r"(\d+)\s*[:-]\s*(\d+)", score_text)
                if score_match:
                    home_goals = int(score_match.group(1))
                    away_goals = int(score_match.group(2))

                # Extract odds
                odds_cells = row.find_all("span", class_=re.compile(r"odds-val"))
                if not odds_cells:
                    odds_cells = row.find_all("td", class_=re.compile(r"odds"))

                odds_h = self._parse_odds(odds_cells[0].text) if len(odds_cells) > 0 else 0.0
                odds_d = self._parse_odds(odds_cells[1].text) if len(odds_cells) > 1 else 0.0
                odds_a = self._parse_odds(odds_cells[2].text) if len(odds_cells) > 2 else 0.0

                results.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "competition": competition,
                    "odds_h": odds_h,
                    "odds_d": odds_d,
                    "odds_a": odds_a,
                    "is_closing": True,
                    "source": "oddsportal",
                })
            except Exception as exc:
                log.debug("OddsPortal: failed to parse historical row: %s", exc)
                continue

        return results
