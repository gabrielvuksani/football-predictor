"""Unified soccerdata library provider -- replaces 5 custom scrapers.

Wraps: FBref, Understat, ClubElo, MatchHistory (football-data.co.uk), SoFIFA.
All free, cached locally by soccerdata, consistent DataFrames.

Install: pip install soccerdata

The soccerdata library handles:
- Rate limiting / polite delays automatically
- Local caching (SQLite-backed, configurable TTL)
- Consistent column naming across sources
- Automatic season/league resolution

This provider exposes a clean interface that the rest of the pipeline
can call without knowing which underlying source is being queried.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# League mapping: our competition codes -> soccerdata league identifiers
# ---------------------------------------------------------------------------
LEAGUE_MAP: dict[str, str] = {
    "PL": "ENG-Premier League",
    "PD": "ESP-La Liga",
    "SA": "ITA-Serie A",
    "BL1": "GER-Bundesliga",
    "FL1": "FRA-Ligue 1",
    "DED": "NED-Eredivisie",
    "PPL": "POR-Primeira Liga",
    "ELC": "ENG-Championship",
}

# Understat uses its own league keys (fewer leagues supported)
UNDERSTAT_MAP: dict[str, str] = {
    "PL": "EPL",
    "PD": "La Liga",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
    "RU1": "RFPL",
}

# football-data.co.uk division codes used by soccerdata.MatchHistory
MATCH_HISTORY_MAP: dict[str, str] = {
    "PL": "ENG-Premier League",
    "PD": "ESP-La Liga",
    "SA": "ITA-Serie A",
    "BL1": "GER-Bundesliga",
    "FL1": "FRA-Ligue 1",
    "DED": "NED-Eredivisie",
    "ELC": "ENG-Championship",
    "PPL": "POR-Primeira Liga",
    "TR1": "TUR-Super Lig",
    "BEL": "BEL-Jupiler Pro League",
    "GR1": "GRE-Super League",
}

# FBref stat_type values accepted by sd.FBref.read_team_season_stats()
FBREF_STAT_TYPES: set[str] = {
    "standard",
    "shooting",
    "passing",
    "passing_types",
    "goal_shot_creation",
    "defense",
    "possession",
    "playing_time",
    "misc",
    "keeper",
    "keeper_adv",
}

# FBref per-match stat_type values accepted by sd.FBref.read_team_match_stats()
FBREF_MATCH_STAT_TYPES: set[str] = {
    "schedule",
    "shooting",
    "keeper",
}


def _empty_df() -> pd.DataFrame:
    """Return a typed empty DataFrame as a safe fallback."""
    return pd.DataFrame()


class SoccerDataProvider:
    """Unified wrapper around the soccerdata library.

    Lazy-initializes each underlying reader on first use so that import
    failures (missing optional deps, network issues) never crash the app.
    """

    def __init__(self, seasons: list[str] | None = None):
        self._seasons = seasons or ["2024-2025", "2023-2024"]
        self._fbref = None
        self._understat = None
        self._clubelo = None
        self._match_history = None
        self._sofifa = None

    # ------------------------------------------------------------------
    # Lazy initializers
    # ------------------------------------------------------------------

    def _get_fbref(self, competition: str | None = None):
        """Lazy-init FBref reader scoped to requested leagues."""
        if self._fbref is None:
            try:
                import soccerdata as sd

                leagues = [LEAGUE_MAP[competition]] if competition and competition in LEAGUE_MAP else list(LEAGUE_MAP.values())
                self._fbref = sd.FBref(leagues=leagues, seasons=self._seasons)
            except Exception as exc:
                log.warning("FBref init failed: %s", exc)
        return self._fbref

    def _get_understat(self):
        """Lazy-init Understat reader."""
        if self._understat is None:
            try:
                import soccerdata as sd

                leagues = list(UNDERSTAT_MAP.values())
                self._understat = sd.Understat(leagues=leagues, seasons=self._seasons)
            except Exception as exc:
                log.warning("Understat init failed: %s", exc)
        return self._understat

    def _get_clubelo(self):
        """Lazy-init ClubElo reader (no league/season scoping needed)."""
        if self._clubelo is None:
            try:
                import soccerdata as sd

                self._clubelo = sd.ClubElo()
            except Exception as exc:
                log.warning("ClubElo init failed: %s", exc)
        return self._clubelo

    def _get_match_history(self, competition: str | None = None):
        """Lazy-init football-data.co.uk MatchHistory reader."""
        if self._match_history is None:
            try:
                import soccerdata as sd

                leagues = (
                    [MATCH_HISTORY_MAP[competition]]
                    if competition and competition in MATCH_HISTORY_MAP
                    else list(MATCH_HISTORY_MAP.values())
                )
                self._match_history = sd.MatchHistory(
                    leagues=leagues, seasons=self._seasons
                )
            except Exception as exc:
                log.warning("MatchHistory init failed: %s", exc)
        return self._match_history

    def _get_sofifa(self, competition: str | None = None):
        """Lazy-init SoFIFA reader."""
        if self._sofifa is None:
            try:
                import soccerdata as sd

                leagues = (
                    [LEAGUE_MAP[competition]]
                    if competition and competition in LEAGUE_MAP
                    else list(LEAGUE_MAP.values())
                )
                self._sofifa = sd.SoFIFA(leagues=leagues, seasons=self._seasons)
            except Exception as exc:
                log.warning("SoFIFA init failed: %s", exc)
        return self._sofifa

    # ------------------------------------------------------------------
    # FBref: aggregated team season stats
    # ------------------------------------------------------------------

    def get_team_season_stats(
        self, competition: str, stat_type: str = "standard"
    ) -> pd.DataFrame:
        """Get aggregated team stats for a season from FBref.

        stat_type is one of: standard, shooting, passing, passing_types,
        goal_shot_creation, defense, possession, playing_time, misc,
        keeper, keeper_adv.

        Returns a DataFrame indexed by (league, season, team) with the
        stat columns specific to that stat_type.
        """
        if stat_type not in FBREF_STAT_TYPES:
            log.warning(
                "Unknown FBref stat_type '%s'; valid: %s",
                stat_type,
                sorted(FBREF_STAT_TYPES),
            )
            return _empty_df()

        sd_league = LEAGUE_MAP.get(competition)
        if sd_league is None:
            log.warning(
                "No soccerdata league mapping for competition '%s'", competition
            )
            return _empty_df()

        fbref = self._get_fbref(competition)
        if fbref is None:
            return _empty_df()

        try:
            df = fbref.read_team_season_stats(stat_type=stat_type)
            # Filter to the requested competition only
            if isinstance(df.index, pd.MultiIndex) and "league" in df.index.names:
                df = df.xs(sd_league, level="league", drop_level=False)
            return df
        except KeyError:
            log.warning(
                "FBref: league '%s' not found in results for stat_type '%s'",
                sd_league,
                stat_type,
            )
            return _empty_df()
        except Exception as exc:
            log.warning(
                "FBref team_season_stats failed for %s/%s: %s",
                competition,
                stat_type,
                exc,
            )
            return _empty_df()

    # ------------------------------------------------------------------
    # FBref: per-match team stats
    # ------------------------------------------------------------------

    def get_team_match_stats(
        self, competition: str, stat_type: str = "shooting"
    ) -> pd.DataFrame:
        """Get per-match team stats from FBref.

        stat_type is one of: schedule, shooting, keeper.

        Returns a DataFrame with one row per team-match containing the
        requested stat columns.
        """
        if stat_type not in FBREF_MATCH_STAT_TYPES:
            log.warning(
                "Unknown FBref match stat_type '%s'; valid: %s",
                stat_type,
                sorted(FBREF_MATCH_STAT_TYPES),
            )
            return _empty_df()

        sd_league = LEAGUE_MAP.get(competition)
        if sd_league is None:
            log.warning(
                "No soccerdata league mapping for competition '%s'", competition
            )
            return _empty_df()

        fbref = self._get_fbref(competition)
        if fbref is None:
            return _empty_df()

        try:
            df = fbref.read_team_match_stats(stat_type=stat_type)
            if isinstance(df.index, pd.MultiIndex) and "league" in df.index.names:
                df = df.xs(sd_league, level="league", drop_level=False)
            return df
        except KeyError:
            log.warning(
                "FBref: league '%s' not found in match results for stat_type '%s'",
                sd_league,
                stat_type,
            )
            return _empty_df()
        except Exception as exc:
            log.warning(
                "FBref team_match_stats failed for %s/%s: %s",
                competition,
                stat_type,
                exc,
            )
            return _empty_df()

    # ------------------------------------------------------------------
    # Understat: xG data
    # ------------------------------------------------------------------

    def get_xg_data(self, competition: str) -> pd.DataFrame:
        """Get match-level xG data from Understat.

        Supported leagues: EPL, La Liga, Bundesliga, Serie A, Ligue 1, RFPL.

        Returns a DataFrame with columns including xG_home, xG_away,
        team names, scores, and match dates.
        """
        sd_league = UNDERSTAT_MAP.get(competition)
        if sd_league is None:
            log.warning(
                "Understat does not cover competition '%s'; supported: %s",
                competition,
                list(UNDERSTAT_MAP.keys()),
            )
            return _empty_df()

        understat = self._get_understat()
        if understat is None:
            return _empty_df()

        try:
            df = understat.read_team_season_stats()
            # Understat returns stats indexed by (league, season, team)
            if isinstance(df.index, pd.MultiIndex) and "league" in df.index.names:
                df = df.xs(sd_league, level="league", drop_level=False)
            return df
        except KeyError:
            log.warning("Understat: league '%s' not found in results", sd_league)
            return _empty_df()
        except Exception as exc:
            log.warning(
                "Understat xG fetch failed for %s: %s", competition, exc
            )
            return _empty_df()

    # ------------------------------------------------------------------
    # ClubElo: team Elo ratings
    # ------------------------------------------------------------------

    def get_elo_ratings(self, target_date: str | date | None = None) -> pd.DataFrame:
        """Get ClubElo ratings for all teams on a given date.

        If target_date is None, returns the most recent available ratings.

        Returns a DataFrame with columns: team, country, level, elo.
        """
        clubelo = self._get_clubelo()
        if clubelo is None:
            return _empty_df()

        try:
            if target_date is None:
                target_date = date.today()
            elif isinstance(target_date, str):
                target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

            df = clubelo.read_by_date(target_date)
            return df
        except Exception as exc:
            log.warning("ClubElo fetch failed for date %s: %s", target_date, exc)
            return _empty_df()

    # ------------------------------------------------------------------
    # football-data.co.uk: historical odds & results
    # ------------------------------------------------------------------

    def get_historical_odds(self, competition: str) -> pd.DataFrame:
        """Get historical match results with betting odds from football-data.co.uk.

        Returns a DataFrame with match results and columns for various
        bookmaker odds (B365H, B365D, B365A, BWH, BWD, BWA, etc.),
        total goals, half-time scores, shots, corners, cards, and more.
        """
        sd_league = MATCH_HISTORY_MAP.get(competition)
        if sd_league is None:
            log.warning(
                "No football-data.co.uk mapping for competition '%s'; supported: %s",
                competition,
                list(MATCH_HISTORY_MAP.keys()),
            )
            return _empty_df()

        mh = self._get_match_history(competition)
        if mh is None:
            return _empty_df()

        try:
            df = mh.read_games()
            # Filter to requested competition
            if isinstance(df.index, pd.MultiIndex) and "league" in df.index.names:
                df = df.xs(sd_league, level="league", drop_level=False)
            return df
        except KeyError:
            log.warning(
                "MatchHistory: league '%s' not found in results", sd_league
            )
            return _empty_df()
        except Exception as exc:
            log.warning(
                "MatchHistory fetch failed for %s: %s", competition, exc
            )
            return _empty_df()

    # ------------------------------------------------------------------
    # SoFIFA: FIFA game player/team ratings
    # ------------------------------------------------------------------

    def get_player_ratings(self, competition: str) -> pd.DataFrame:
        """Get FIFA game player ratings from SoFIFA.

        Returns a DataFrame with player-level attributes including
        overall rating, potential, positions, pace, shooting, passing,
        dribbling, defending, and physical stats.
        """
        sd_league = LEAGUE_MAP.get(competition)
        if sd_league is None:
            log.warning(
                "No SoFIFA league mapping for competition '%s'", competition
            )
            return _empty_df()

        sofifa = self._get_sofifa(competition)
        if sofifa is None:
            return _empty_df()

        try:
            df = sofifa.read_players()
            if isinstance(df.index, pd.MultiIndex) and "league" in df.index.names:
                df = df.xs(sd_league, level="league", drop_level=False)
            return df
        except KeyError:
            log.warning(
                "SoFIFA: league '%s' not found in results", sd_league
            )
            return _empty_df()
        except Exception as exc:
            log.warning(
                "SoFIFA fetch failed for %s: %s", competition, exc
            )
            return _empty_df()

    # ------------------------------------------------------------------
    # Convenience aggregators
    # ------------------------------------------------------------------

    def get_full_team_profile(self, competition: str) -> dict[str, pd.DataFrame]:
        """Fetch all available stat categories for a competition.

        Returns a dict keyed by stat category name -> DataFrame.
        Empty DataFrames are included for categories that failed.
        """
        profile: dict[str, pd.DataFrame] = {}

        # FBref season-level stats (all 11 types)
        for stat_type in sorted(FBREF_STAT_TYPES):
            profile[f"fbref_{stat_type}"] = self.get_team_season_stats(
                competition, stat_type=stat_type
            )

        # FBref match-level stats
        for stat_type in sorted(FBREF_MATCH_STAT_TYPES):
            profile[f"fbref_match_{stat_type}"] = self.get_team_match_stats(
                competition, stat_type=stat_type
            )

        # Understat xG
        profile["understat_xg"] = self.get_xg_data(competition)

        # Historical odds
        profile["historical_odds"] = self.get_historical_odds(competition)

        # Player ratings
        profile["sofifa_players"] = self.get_player_ratings(competition)

        # Elo ratings (not competition-specific, but included for completeness)
        profile["clubelo"] = self.get_elo_ratings()

        return profile

    def get_match_features(self, competition: str) -> pd.DataFrame:
        """Build a match-level feature table by joining schedule, shooting, and odds.

        Merges FBref match schedule with shooting stats and historical odds
        to produce a single DataFrame suitable for model training. Returns
        an empty DataFrame if the join produces no rows.
        """
        schedule = self.get_team_match_stats(competition, stat_type="schedule")
        if schedule.empty:
            log.warning("No schedule data for %s; cannot build match features", competition)
            return _empty_df()

        shooting = self.get_team_match_stats(competition, stat_type="shooting")
        odds = self.get_historical_odds(competition)

        result = schedule.copy()

        # Merge shooting stats if available
        if not shooting.empty:
            # Use common index levels for the join
            shared_idx = [
                c for c in shooting.index.names if c in result.index.names
            ]
            if shared_idx:
                try:
                    result = result.join(shooting, rsuffix="_shooting")
                except Exception as exc:
                    log.debug("Shooting join failed: %s", exc)

        # Merge odds if available
        if not odds.empty:
            try:
                # Reset to flat columns for a merge on date + team columns
                odds_flat = odds.reset_index() if isinstance(odds.index, pd.MultiIndex) else odds
                result_flat = result.reset_index() if isinstance(result.index, pd.MultiIndex) else result

                # Attempt merge on date and team columns
                date_cols = [c for c in odds_flat.columns if "date" in c.lower()]
                team_cols = [c for c in odds_flat.columns if "team" in c.lower() or "home" in c.lower()]

                if date_cols and team_cols:
                    merge_on = [date_cols[0], team_cols[0]]
                    shared = [c for c in merge_on if c in result_flat.columns and c in odds_flat.columns]
                    if shared:
                        result = result_flat.merge(
                            odds_flat, on=shared, how="left", suffixes=("", "_odds")
                        )
            except Exception as exc:
                log.debug("Odds merge failed: %s", exc)

        return result
