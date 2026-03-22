from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from footy.config import settings
from footy.db import connect
from footy.providers import (
    ClubEloProvider,
    FBrefProvider,
    OddsPortalProvider,
    OpenFootballProvider,
    OpenMeteoProvider,
    SofaScoreProvider,
    TransfermarktProvider,
    UnderstatProvider,
)

log = logging.getLogger(__name__)


class DataOrchestrator:
    """Coordinator for the new zero-cost provider stack.

    This is intentionally additive: it gives the codebase a new orchestration path
    without forcing every legacy CLI/API entry point to change on day one.
    """

    def __init__(self):
        self.settings = settings()
        self.openfootball = OpenFootballProvider(enabled=self.settings.enable_openfootball)
        self.sofascore = SofaScoreProvider(enabled=self.settings.enable_sofascore)
        self.clubelo = ClubEloProvider(enabled=self.settings.enable_clubelo)
        self.weather = OpenMeteoProvider(enabled=self.settings.enable_open_meteo)
        self.understat = UnderstatProvider(enabled=self.settings.enable_understat)
        self.fbref = FBrefProvider(enabled=self.settings.enable_fbref)
        self.transfermarkt = TransfermarktProvider(enabled=self.settings.enable_transfermarkt)
        self.oddsportal = OddsPortalProvider(enabled=self.settings.enable_oddsportal)

    def close(self) -> None:
        for provider in (self.openfootball, self.sofascore, self.clubelo, self.weather,
                         self.understat, self.fbref, self.transfermarkt, self.oddsportal):
            provider.close()

    def seed_sofascore_schedule(self, days_back: int = 0, days_forward: int = 7) -> int:
        from datetime import date, timedelta

        con = connect()
        rows = self.sofascore.fetch_window(date.today() - timedelta(days=days_back), date.today() + timedelta(days=days_forward))
        inserted = 0
        for event in rows:
            normalized = self.sofascore.normalize_event(event)
            if not normalized or normalized["competition"] not in self.settings.tracked_competitions:
                continue
            con.execute(
                """INSERT OR REPLACE INTO matches
                   (match_id, provider, competition, season, utc_date, status, home_team, away_team, home_goals, away_goals, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    normalized["match_id"],
                    normalized["provider"],
                    normalized["competition"],
                    normalized["season"],
                    normalized["utc_date"],
                    normalized["status"],
                    normalized["home_team"],
                    normalized["away_team"],
                    normalized["home_goals"],
                    normalized["away_goals"],
                    str(normalized["raw_json"]),
                ],
            )
            if normalized.get("venue_name"):
                con.execute(
                    "INSERT OR REPLACE INTO match_extras(match_id, provider, competition, venue_name) VALUES (?, ?, ?, ?)",
                    [normalized["match_id"], normalized["provider"], normalized["competition"], normalized["venue_name"]],
                )
            inserted += 1
        return inserted

    def seed_openfootball_schedule(self, competition: str, season: str) -> int:
        rows = self.openfootball.fetch_season_matches(competition, season)
        con = connect()
        inserted = 0
        for row in rows:
            match_id = int(hashlib.blake2b(str(("openfootball", competition, season, row["utc_date"], row["home_team"], row["away_team"])).encode(), digest_size=8).hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF
            con.execute(
                """INSERT OR REPLACE INTO matches
                   (match_id, provider, competition, season, utc_date, status, home_team, away_team, home_goals, away_goals, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    int(match_id),
                    "openfootball",
                    competition,
                    int(str(season)[:4]) if str(season)[:4].isdigit() else None,
                    row["utc_date"],
                    row["status"],
                    row["home_team"],
                    row["away_team"],
                    row["home_goals"],
                    row["away_goals"],
                    str(row["raw_json"]),
                ],
            )
            inserted += 1
        return inserted

    def refresh_clubelo_snapshot(self, target_date: str | None = None) -> int:
        rows = self.clubelo.fetch_ratings(target_date)
        con = connect()
        updated = 0
        for row in rows:
            con.execute(
                "INSERT OR REPLACE INTO elo_state(team, rating) VALUES (?, ?)",
                [row["team"], row["elo"]],
            )
            updated += 1
        return updated

    def enrich_match_weather(self, match_id: int, latitude: float, longitude: float, kickoff: datetime, venue_name: str | None = None) -> dict[str, Any]:
        weather = self.weather.fetch_kickoff_weather(latitude, longitude, kickoff)
        con = connect()
        con.execute(
            """INSERT OR REPLACE INTO weather_data
               (match_id, venue_name, latitude, longitude,
                kickoff_temperature_c, kickoff_apparent_temperature_c,
                kickoff_precipitation_mm, kickoff_wind_speed_kmh, kickoff_wind_gusts_kmh,
                kickoff_humidity_pct, kickoff_cloud_cover_pct, kickoff_weather_code,
                rainfall_prev_24h_mm, rainfall_prev_48h_mm, source, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                match_id,
                venue_name,
                latitude,
                longitude,
                weather.get("kickoff_temperature_c"),
                weather.get("kickoff_apparent_temperature_c"),
                weather.get("kickoff_precipitation_mm"),
                weather.get("kickoff_wind_speed_kmh"),
                weather.get("kickoff_wind_gusts_kmh"),
                weather.get("kickoff_humidity_pct"),
                weather.get("kickoff_cloud_cover_pct"),
                weather.get("kickoff_weather_code"),
                weather.get("rainfall_prev_24h_mm"),
                weather.get("rainfall_prev_48h_mm"),
                weather.get("source"),
                str(weather.get("raw_json")),
            ],
        )
        return weather

    def backfill_understat_xg(self, match_ids: Iterable[int]) -> int:
        con = connect()
        updated = 0
        for match_id in match_ids:
            try:
                data = self.understat.extract_match_team_xg(int(match_id))
            except Exception as e:
                log.debug("understat failed for %s: %s", match_id, e)
                continue
            con.execute(
                """UPDATE match_extras
                   SET understat_xg_home=?, understat_xg_away=?,
                       understat_shots_home=?, understat_shots_away=?
                   WHERE match_id=?""",
                [
                    data.get("understat_xg_home"),
                    data.get("understat_xg_away"),
                    data.get("understat_shots_home"),
                    data.get("understat_shots_away"),
                    int(match_id),
                ],
            )
            updated += 1
        return updated

    def refresh_fbref_stats(self, competition: str, season: str = "2025-2026") -> int:
        """Fetch and store FBref stats for a competition."""
        try:
            stats = self.fbref.fetch_team_stats(competition, season)
        except Exception as exc:
            log.warning("FBref refresh failed for %s: %s", competition, exc)
            return 0
        con = connect()
        stored = 0
        for team_stats in stats:
            team = team_stats.get("team", "")
            xg = team_stats.get("xg", 0)
            npxg = team_stats.get("npxg", 0)
            games = team_stats.get("games", 0)
            if team and games > 0:
                con.execute(
                    """INSERT OR REPLACE INTO provider_status
                       (match_id, provider, status, detail)
                       VALUES (?, 'fbref', 'ok', ?)""",
                    [int(hashlib.blake2b(str((competition, team, season)).encode(), digest_size=8).hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF, f"{team}: xG={xg}, npxG={npxg}, games={games}"],
                )
                stored += 1
        return stored

    def refresh_transfermarkt_values(self, competition: str, season: str = "2025") -> int:
        """Fetch and store squad market values from Transfermarkt."""
        try:
            values = self.transfermarkt.fetch_squad_values(competition, season)
        except Exception as exc:
            log.warning("Transfermarkt refresh failed for %s: %s", competition, exc)
            return 0
        con = connect()
        stored = 0
        for team_data in values:
            team = team_data.get("team", "")
            squad_value = team_data.get("squad_market_value_eur", 0)
            avg_value = team_data.get("average_player_value_eur", 0)
            squad_size = team_data.get("squad_size", 0)
            avg_age = team_data.get("average_age", 0)
            if team:
                con.execute(
                    """INSERT OR REPLACE INTO market_values
                       (team, competition, squad_market_value_eur, average_player_value_eur,
                        squad_size, average_age, source)
                       VALUES (?, ?, ?, ?, ?, ?, 'transfermarkt')""",
                    [team, competition, squad_value, avg_value, squad_size, avg_age],
                )
                stored += 1
        return stored

    def refresh_transfermarkt_injuries(self, competition: str) -> int:
        """Fetch and store injury data from Transfermarkt."""
        try:
            injuries = self.transfermarkt.fetch_injuries(competition)
        except Exception as exc:
            log.warning("Transfermarkt injuries failed for %s: %s", competition, exc)
            return 0
        con = connect()
        stored = 0
        for inj in injuries:
            team = inj.get("team", "")
            player = inj.get("player_name", "")
            if team and player:
                con.execute(
                    """INSERT OR REPLACE INTO transfermarkt_injuries
                       (team, player_name, injury_type, return_date, source)
                       VALUES (?, ?, ?, ?, 'transfermarkt')""",
                    [team, player, inj.get("injury_type", ""), inj.get("return_date", "")],
                )
                stored += 1
        return stored

    def refresh_oddsportal_odds(self, competition: str) -> int:
        """Fetch and store upcoming odds from OddsPortal."""
        try:
            odds = self.oddsportal.fetch_upcoming_odds(competition)
        except Exception as exc:
            log.warning("OddsPortal refresh failed for %s: %s", competition, exc)
            return 0
        con = connect()
        stored = 0
        for match_odds in odds:
            home = match_odds.get("home_team", "")
            away = match_odds.get("away_team", "")
            odds_h = match_odds.get("odds_h", 0)
            odds_d = match_odds.get("odds_d", 0)
            odds_a = match_odds.get("odds_a", 0)
            if home and away and odds_h > 0:
                match_id = int(hashlib.blake2b(str((competition, home, away, match_odds.get("date", ""))).encode(), digest_size=8).hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF
                con.execute(
                    """INSERT OR REPLACE INTO odds_history
                       (match_id, bookmaker, market, outcome, price, source)
                       VALUES (?, 'oddsportal_consensus', '1x2', 'H', ?, 'oddsportal')""",
                    [match_id, odds_h],
                )
                con.execute(
                    """INSERT OR REPLACE INTO odds_history
                       (match_id, bookmaker, market, outcome, price, source)
                       VALUES (?, 'oddsportal_consensus', '1x2', 'D', ?, 'oddsportal')""",
                    [match_id, odds_d],
                )
                con.execute(
                    """INSERT OR REPLACE INTO odds_history
                       (match_id, bookmaker, market, outcome, price, source)
                       VALUES (?, 'oddsportal_consensus', '1x2', 'A', ?, 'oddsportal')""",
                    [match_id, odds_a],
                )
                stored += 1
        return stored

    def full_refresh_all_providers(self) -> dict[str, Any]:
        """Run a full refresh across all enabled providers for all tracked competitions."""
        results: dict[str, Any] = {}

        # Club Elo snapshot
        try:
            results["clubelo"] = self.refresh_clubelo_snapshot()
        except Exception as exc:
            results["clubelo"] = {"error": str(exc)}

        # Per-competition enrichments
        for comp in self.settings.tracked_competitions:
            comp_results: dict[str, Any] = {}

            try:
                comp_results["fbref"] = self.refresh_fbref_stats(comp)
            except Exception as exc:
                comp_results["fbref"] = {"error": str(exc)}

            try:
                comp_results["transfermarkt_values"] = self.refresh_transfermarkt_values(comp)
            except Exception as exc:
                comp_results["transfermarkt_values"] = {"error": str(exc)}

            try:
                comp_results["transfermarkt_injuries"] = self.refresh_transfermarkt_injuries(comp)
            except Exception as exc:
                comp_results["transfermarkt_injuries"] = {"error": str(exc)}

            try:
                comp_results["oddsportal"] = self.refresh_oddsportal_odds(comp)
            except Exception as exc:
                comp_results["oddsportal"] = {"error": str(exc)}

            results[comp] = comp_results

        return results
