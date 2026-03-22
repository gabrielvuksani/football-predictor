"""FBref Scraper — advanced team stats from fbref.com (StatsBomb/Opta data).

Free, no API key. Respectful 3.5-second delay between requests.
Provides: shooting, passing, possession, defensive actions per team.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from bs4 import BeautifulSoup

from footy.providers.base import BaseProvider, ProviderError

log = logging.getLogger(__name__)

# FBref league slugs mapping to our competition codes
FBREF_LEAGUES: dict[str, dict[str, str]] = {
    "PL":  {"id": "9", "slug": "Premier-League", "country": "eng"},
    "PD":  {"id": "12", "slug": "La-Liga", "country": "esp"},
    "SA":  {"id": "11", "slug": "Serie-A", "country": "ita"},
    "BL1": {"id": "20", "slug": "Bundesliga", "country": "ger"},
    "FL1": {"id": "13", "slug": "Ligue-1", "country": "fra"},
    "DED": {"id": "23", "slug": "Eredivisie", "country": "ned"},
    "PPL": {"id": "32", "slug": "Primeira-Liga", "country": "por"},
    "ELC": {"id": "10", "slug": "Championship", "country": "eng"},
    "TR1": {"id": "26", "slug": "Super-Lig", "country": "tur"},
    "BEL": {"id": "37", "slug": "Belgian-Pro-League", "country": "bel"},
    "SL":  {"id": "40", "slug": "Scottish-Premiership", "country": "sco"},
    "A1":  {"id": "56", "slug": "Austrian-Bundesliga", "country": "aut"},
    "GR1": {"id": "27", "slug": "Super-League-Greece", "country": "gre"},
    "SWS": {"id": "57", "slug": "Swiss-Super-League", "country": "che"},
    "DK1": {"id": "50", "slug": "Danish-Superliga", "country": "den"},
    "SE1": {"id": "29", "slug": "Allsvenskan", "country": "swe"},
    "NO1": {"id": "28", "slug": "Eliteserien", "country": "nor"},
    "PL1": {"id": "36", "slug": "Ekstraklasa", "country": "pol"},
}

_BASE_URL = "https://fbref.com/en/comps"
_REQUEST_DELAY = 3.5  # seconds between requests (polite scraping)


class FBrefProvider(BaseProvider):
    """Scrapes FBref for advanced team-level statistics."""

    name = "fbref"

    def __init__(self, *, enabled: bool = True):
        super().__init__(enabled=enabled)
        self._last_request_time: float = 0.0

    def _polite_delay(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < _REQUEST_DELAY:
            time.sleep(_REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _fetch_page(self, url: str) -> BeautifulSoup:
        self._polite_delay()
        html = self.fetch_text(url, ttl=3600)  # cache 1 hour
        # FBref wraps many tables in HTML comments; uncomment them for parsing
        html = re.sub(r'<!--\s*(<div.*?</div>)\s*-->', r'\1', html, flags=re.DOTALL)
        return BeautifulSoup(html, "html.parser")

    def _parse_float(self, text: str | None) -> float:
        if not text:
            return 0.0
        text = text.strip().replace(",", "")
        try:
            return float(text)
        except ValueError:
            return 0.0

    def _parse_int(self, text: str | None) -> int:
        if not text:
            return 0
        text = text.strip().replace(",", "")
        try:
            return int(text)
        except ValueError:
            return 0

    def fetch_team_stats(self, competition: str, season: str = "2025-2026") -> list[dict[str, Any]]:
        """Fetch squad standard stats for a league season.

        Returns list of dicts with team-level stats: goals, xG, assists, xA,
        possession, passes completed, progressive passes, etc.
        """
        if not self.enabled:
            raise ProviderError("FBref provider is disabled")

        league_info = FBREF_LEAGUES.get(competition)
        if not league_info:
            log.warning("FBref: no mapping for competition %s", competition)
            return []

        url = f"{_BASE_URL}/{league_info['id']}/{season}/stats/{season}-{league_info['slug']}-Stats"
        try:
            soup = self._fetch_page(url)
        except Exception as exc:
            log.warning("FBref: failed to fetch stats for %s: %s", competition, exc)
            return []

        results: list[dict[str, Any]] = []
        table = soup.find("table", id="stats_squads_standard_for")
        if not table:
            # Try alternate table ID patterns
            table = soup.find("table", id=re.compile(r"stats_squads"))
            if not table:
                log.warning("FBref: no stats table found for %s", competition)
                return results

        tbody = table.find("tbody")
        if not tbody:
            return results

        for row in tbody.find_all("tr"):
            if row.get("class") and "thead" in row.get("class", []):
                continue
            cells = row.find_all(["th", "td"])
            if len(cells) < 10:
                continue

            team_cell = cells[0]
            team_link = team_cell.find("a")
            team_name = team_link.text.strip() if team_link else team_cell.text.strip()

            stats: dict[str, Any] = {
                "team": team_name,
                "competition": competition,
                "source": "fbref",
            }

            # Parse stat cells by data-stat attribute
            for cell in cells:
                stat_name = cell.get("data-stat", "")
                value = cell.text.strip()
                if stat_name in ("players_used",):
                    stats[stat_name] = self._parse_int(value)
                elif stat_name in ("games", "games_starts", "minutes"):
                    stats[stat_name] = self._parse_int(value)
                elif stat_name in ("goals", "assists", "goals_pens", "pens_made", "pens_att",
                                   "cards_yellow", "cards_red"):
                    stats[stat_name] = self._parse_int(value)
                elif stat_name in ("goals_per90", "assists_per90", "goals_assists_per90",
                                   "goals_pens_per90", "xg", "xg_assist", "xg_per90",
                                   "xg_assist_per90", "xg_xg_assist_per90",
                                   "npxg", "npxg_per90", "npxg_xg_assist_per90"):
                    stats[stat_name] = self._parse_float(value)

            results.append(stats)

        return results

    def fetch_shooting_stats(self, competition: str, season: str = "2025-2026") -> list[dict[str, Any]]:
        """Fetch squad shooting stats: shots, SoT, shot distance, FK shots, etc."""
        if not self.enabled:
            raise ProviderError("FBref provider is disabled")

        league_info = FBREF_LEAGUES.get(competition)
        if not league_info:
            return []

        url = f"{_BASE_URL}/{league_info['id']}/{season}/shooting/{season}-{league_info['slug']}-Stats"
        try:
            soup = self._fetch_page(url)
        except Exception as exc:
            log.warning("FBref: failed to fetch shooting for %s: %s", competition, exc)
            return []

        results: list[dict[str, Any]] = []
        table = soup.find("table", id="stats_squads_shooting_for")
        if not table:
            table = soup.find("table", id=re.compile(r"stats_squads_shooting"))
        if not table:
            return results

        tbody = table.find("tbody")
        if not tbody:
            return results

        for row in tbody.find_all("tr"):
            if row.get("class") and "thead" in row.get("class", []):
                continue
            cells = row.find_all(["th", "td"])
            if len(cells) < 8:
                continue

            team_cell = cells[0]
            team_link = team_cell.find("a")
            team_name = team_link.text.strip() if team_link else team_cell.text.strip()

            stats: dict[str, Any] = {
                "team": team_name,
                "competition": competition,
                "source": "fbref",
            }

            for cell in cells:
                stat_name = cell.get("data-stat", "")
                value = cell.text.strip()
                if stat_name in ("shots", "shots_on_target", "shots_free_kicks",
                                 "pens_made", "pens_att"):
                    stats[stat_name] = self._parse_int(value)
                elif stat_name in ("shots_per90", "shots_on_target_per90",
                                   "shots_on_target_pct", "goals_per_shot",
                                   "goals_per_shot_on_target", "average_shot_distance",
                                   "xg", "npxg", "npxg_per_shot", "xg_net"):
                    stats[stat_name] = self._parse_float(value)

            results.append(stats)

        return results

    def fetch_passing_stats(self, competition: str, season: str = "2025-2026") -> list[dict[str, Any]]:
        """Fetch squad passing stats: total/short/medium/long pass completion, progressive passes."""
        if not self.enabled:
            raise ProviderError("FBref provider is disabled")

        league_info = FBREF_LEAGUES.get(competition)
        if not league_info:
            return []

        url = f"{_BASE_URL}/{league_info['id']}/{season}/passing/{season}-{league_info['slug']}-Stats"
        try:
            soup = self._fetch_page(url)
        except Exception as exc:
            log.warning("FBref: failed to fetch passing for %s: %s", competition, exc)
            return []

        results: list[dict[str, Any]] = []
        table = soup.find("table", id="stats_squads_passing_for")
        if not table:
            table = soup.find("table", id=re.compile(r"stats_squads_passing"))
        if not table:
            return results

        tbody = table.find("tbody")
        if not tbody:
            return results

        for row in tbody.find_all("tr"):
            if row.get("class") and "thead" in row.get("class", []):
                continue
            cells = row.find_all(["th", "td"])
            if len(cells) < 8:
                continue

            team_cell = cells[0]
            team_link = team_cell.find("a")
            team_name = team_link.text.strip() if team_link else team_cell.text.strip()

            stats: dict[str, Any] = {
                "team": team_name,
                "competition": competition,
                "source": "fbref",
            }

            for cell in cells:
                stat_name = cell.get("data-stat", "")
                value = cell.text.strip()
                if stat_name in ("passes_completed", "passes", "passes_short",
                                 "passes_completed_short", "passes_medium",
                                 "passes_completed_medium", "passes_long",
                                 "passes_completed_long", "passes_progressive"):
                    stats[stat_name] = self._parse_int(value)
                elif stat_name in ("passes_pct", "passes_pct_short", "passes_pct_medium",
                                   "passes_pct_long", "xa", "xa_net",
                                   "passes_progressive_distance"):
                    stats[stat_name] = self._parse_float(value)

            results.append(stats)

        return results

    def fetch_defense_stats(self, competition: str, season: str = "2025-2026") -> list[dict[str, Any]]:
        """Fetch squad defensive stats: tackles, interceptions, blocks, clearances."""
        if not self.enabled:
            raise ProviderError("FBref provider is disabled")

        league_info = FBREF_LEAGUES.get(competition)
        if not league_info:
            return []

        url = f"{_BASE_URL}/{league_info['id']}/{season}/defense/{season}-{league_info['slug']}-Stats"
        try:
            soup = self._fetch_page(url)
        except Exception as exc:
            log.warning("FBref: failed to fetch defense for %s: %s", competition, exc)
            return []

        results: list[dict[str, Any]] = []
        table = soup.find("table", id="stats_squads_defense_for")
        if not table:
            table = soup.find("table", id=re.compile(r"stats_squads_defense"))
        if not table:
            return results

        tbody = table.find("tbody")
        if not tbody:
            return results

        for row in tbody.find_all("tr"):
            if row.get("class") and "thead" in row.get("class", []):
                continue
            cells = row.find_all(["th", "td"])
            if len(cells) < 8:
                continue

            team_cell = cells[0]
            team_link = team_cell.find("a")
            team_name = team_link.text.strip() if team_link else team_cell.text.strip()

            stats: dict[str, Any] = {
                "team": team_name,
                "competition": competition,
                "source": "fbref",
            }

            for cell in cells:
                stat_name = cell.get("data-stat", "")
                value = cell.text.strip()
                if stat_name in ("tackles", "tackles_won", "tackles_def_3rd",
                                 "tackles_mid_3rd", "tackles_att_3rd",
                                 "interceptions", "blocks", "clearances", "errors"):
                    stats[stat_name] = self._parse_int(value)
                elif stat_name in ("tackles_per90", "tackles_won_pct",
                                   "interceptions_per90", "blocks_per90"):
                    stats[stat_name] = self._parse_float(value)

            results.append(stats)

        return results

    def fetch_all_stats(self, competition: str, season: str = "2025-2026") -> dict[str, list[dict[str, Any]]]:
        """Fetch all stat categories for a competition."""
        return {
            "standard": self.fetch_team_stats(competition, season),
            "shooting": self.fetch_shooting_stats(competition, season),
            "passing": self.fetch_passing_stats(competition, season),
            "defense": self.fetch_defense_stats(competition, season),
        }
