"""Transfermarkt Provider — squad market values, injuries, and squad depth.

Free, no API key. Scraping-based with polite delays.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from bs4 import BeautifulSoup

from footy.providers.base import BaseProvider, ProviderError

log = logging.getLogger(__name__)

_BASE_URL = "https://www.transfermarkt.com"
_REQUEST_DELAY = 4.0  # polite scraping delay

# Competition code → Transfermarkt URL slugs
TRANSFERMARKT_LEAGUES: dict[str, dict[str, str]] = {
    "PL":  {"slug": "premier-league", "id": "GB1"},
    "PD":  {"slug": "laliga", "id": "ES1"},
    "SA":  {"slug": "serie-a", "id": "IT1"},
    "BL1": {"slug": "1-bundesliga", "id": "L1"},
    "FL1": {"slug": "ligue-1", "id": "FR1"},
    "DED": {"slug": "eredivisie", "id": "NL1"},
    "PPL": {"slug": "liga-nos", "id": "PO1"},
    "ELC": {"slug": "championship", "id": "GB2"},
    "TR1": {"slug": "super-lig", "id": "TR1"},
    "BEL": {"slug": "jupiler-pro-league", "id": "BE1"},
    "SL":  {"slug": "scottish-premiership", "id": "SC1"},
    "A1":  {"slug": "bundesliga", "id": "A1"},
    "GR1": {"slug": "super-league-1", "id": "GR1"},
    "SWS": {"slug": "super-league", "id": "C1"},
    "DK1": {"slug": "superligaen", "id": "DK1"},
    "SE1": {"slug": "allsvenskan", "id": "SE1"},
    "NO1": {"slug": "eliteserien", "id": "NO1"},
    "PL1": {"slug": "pko-bp-ekstraklasa", "id": "PL1"},
}


class TransfermarktProvider(BaseProvider):
    """Scrapes Transfermarkt for squad values, injuries, and depth data."""

    name = "transfermarkt"

    def __init__(self, *, enabled: bool = True):
        super().__init__(
            enabled=enabled,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-GB,en;q=0.9",
            },
        )
        self._last_request_time: float = 0.0

    def _polite_delay(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < _REQUEST_DELAY:
            time.sleep(_REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _parse_market_value(self, text: str | None) -> float:
        """Parse Transfermarkt market value strings like '€120.50m' or '€800k'."""
        if not text:
            return 0.0
        text = text.strip().replace("€", "").replace("$", "").replace("£", "")
        try:
            if "bn" in text.lower():
                return float(re.sub(r"[^\d.]", "", text)) * 1_000_000_000
            elif "m" in text.lower():
                return float(re.sub(r"[^\d.]", "", text)) * 1_000_000
            elif "k" in text.lower() or "th" in text.lower():
                return float(re.sub(r"[^\d.]", "", text)) * 1_000
            else:
                return float(re.sub(r"[^\d.]", "", text))
        except (ValueError, TypeError):
            return 0.0

    def fetch_squad_values(self, competition: str, season: str = "2025") -> list[dict[str, Any]]:
        """Fetch squad market values for all teams in a competition.

        Returns list of dicts with team name, total value, avg value, squad size, avg age.
        """
        if not self.enabled:
            raise ProviderError("Transfermarkt provider is disabled")

        league_info = TRANSFERMARKT_LEAGUES.get(competition)
        if not league_info:
            log.warning("Transfermarkt: no mapping for competition %s", competition)
            return []

        url = f"{_BASE_URL}/{league_info['slug']}/startseite/wettbewerb/{league_info['id']}/plus/?saison_id={season}"
        try:
            self._polite_delay()
            html = self.fetch_text(url, ttl=86400)  # cache 24 hours
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            log.warning("Transfermarkt: failed to fetch squad values for %s: %s", competition, exc)
            return []

        results: list[dict[str, Any]] = []
        # Look for the main table
        table = soup.find("table", class_="items")
        if not table:
            log.warning("Transfermarkt: no items table found for %s", competition)
            return results

        tbody = table.find("tbody")  # type: ignore[union-attr]
        if not tbody:
            return results

        for row in tbody.find_all("tr", recursive=False):  # type: ignore[union-attr]
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            # Extract team name
            team_link = row.find("a", class_="vereinprofil_tooltip")
            if not team_link:
                continue
            team_name = team_link.text.strip()
            if not team_name:
                continue

            # Parse available data from cells
            stats: dict[str, Any] = {
                "team": team_name,
                "competition": competition,
                "source": "transfermarkt",
            }

            # Try to extract numeric data from cells
            for cell in cells:
                text = cell.text.strip()
                # Squad size (small integer)
                if re.match(r"^\d{1,2}$", text) and "squad_size" not in stats:
                    stats["squad_size"] = int(text)
                # Average age (decimal number like 25.4)
                elif re.match(r"^\d{2}\.\d$", text) and "average_age" not in stats:
                    stats["average_age"] = float(text)
                # Market value
                elif ("€" in text or "m" in text.lower()) and "squad_market_value_eur" not in stats:
                    val = self._parse_market_value(text)
                    if val > 0:
                        stats["squad_market_value_eur"] = val

            # Compute average player value
            if "squad_market_value_eur" in stats and "squad_size" in stats and stats["squad_size"] > 0:
                stats["average_player_value_eur"] = stats["squad_market_value_eur"] / stats["squad_size"]

            results.append(stats)

        return results

    def fetch_injuries(self, competition: str) -> list[dict[str, Any]]:
        """Fetch current injuries for all teams in a competition.

        Returns list of dicts with team, player, injury type, and expected return.
        """
        if not self.enabled:
            raise ProviderError("Transfermarkt provider is disabled")

        league_info = TRANSFERMARKT_LEAGUES.get(competition)
        if not league_info:
            return []

        url = f"{_BASE_URL}/{league_info['slug']}/verletztespieler/wettbewerb/{league_info['id']}"
        try:
            self._polite_delay()
            html = self.fetch_text(url, ttl=3600)  # cache 1 hour
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            log.warning("Transfermarkt: failed to fetch injuries for %s: %s", competition, exc)
            return []

        results: list[dict[str, Any]] = []
        table = soup.find("table", class_="items")
        if not table:
            return results

        tbody = table.find("tbody")  # type: ignore[union-attr]
        if not tbody:
            return results

        current_team = ""
        for row in tbody.find_all("tr"):  # type: ignore[union-attr]
            # Team header rows
            team_header = row.find("td", class_="hauptlink")
            if team_header:
                team_link = team_header.find("a")
                if team_link:
                    current_team = team_link.text.strip()
                continue

            cells = row.find_all("td")
            if len(cells) < 3 or not current_team:
                continue

            # Player name
            player_link = row.find("a", class_="spielprofil_tooltip")
            if not player_link:
                continue

            player_name = player_link.text.strip()
            injury_type = ""
            return_date = ""

            for cell in cells:
                text = cell.text.strip()
                # Injury description (typically contains common injury terms)
                if any(term in text.lower() for term in
                       ("injury", "knee", "muscle", "ankle", "hamstring", "calf",
                        "groin", "back", "shoulder", "concussion", "illness",
                        "thigh", "achilles", "ligament", "fracture", "bruise")):
                    injury_type = text
                # Return date (date-like patterns)
                elif re.match(r"^\w+ \d+, \d{4}$", text) or re.match(r"^\d{2}/\d{2}/\d{4}$", text):
                    return_date = text

            results.append({
                "team": current_team,
                "player_name": player_name,
                "injury_type": injury_type,
                "return_date": return_date,
                "competition": competition,
                "source": "transfermarkt",
            })

        return results
