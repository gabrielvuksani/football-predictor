"""Understat xG provider – NOT YET IMPLEMENTED.

This module is a placeholder.  All public functions return empty results.
A real implementation would scrape or query the Understat API for
expected-goals data.
"""
from __future__ import annotations
from typing import Optional


class UnderstatProvider:
    """Stub – returns empty data for every query."""

    def get_provider_status(self) -> dict:
        return {"provider": "Understat", "status": "not_implemented"}

    def get_team_season_stats(self, team: str, *, season: int = 2024) -> Optional[dict]:
        return None

    def get_match_xg(self, match_id: int) -> Optional[dict]:
        return None

    def compute_team_rolling_xg(self, team: str, matches_window: int = 5) -> Optional[dict]:
        return None


def get_understat_provider() -> UnderstatProvider:
    """Get or create Understat provider instance."""
    return UnderstatProvider()
