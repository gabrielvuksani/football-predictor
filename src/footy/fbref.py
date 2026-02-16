"""FBRef advanced statistics provider – NOT YET IMPLEMENTED.

All public functions return empty results.  A real implementation would
scrape fbref.com for shooting, possession, defense, and passing metrics.
"""
from __future__ import annotations
from typing import Optional, Dict, Any


class FBRefProvider:
    """Stub – returns empty data for every query."""

    def get_provider_status(self) -> dict:
        return {"status": "not_implemented", "mode": "stub"}
    def get_team_shooting_stats(self, team: str, season: int, **kw) -> Optional[dict]:
        return None
    def get_team_possession_stats(self, team: str, season: int, **kw) -> Optional[dict]:
        return None
    def get_team_defense_stats(self, team: str, season: int, **kw) -> Optional[dict]:
        return None
    def get_team_passing_stats(self, team: str, season: int, **kw) -> Optional[dict]:
        return None
    def get_all_team_stats(self, team: str, season: int) -> Optional[Dict[str, Any]]:
        return None
    def compute_team_stats_comparison(self, t1: str, t2: str, season: int) -> Dict[str, Any]:
        return {}

def get_fbref_provider() -> FBRefProvider:
    """Get singleton FBRef provider instance."""
    return FBRefProvider()
