"""Shared league table tracker for experts that need league position context.

Eliminates duplicate league table maintenance across Context, LeagueTable,
and Motivation experts. Each expert registers as a consumer and receives
consistent table state without maintaining their own copy.

Usage:
    tracker = LeagueTableTracker()
    # In expert compute loop:
    table = tracker.get_table(season_key)
    entry = tracker.get_entry(season_key, team)
    # After finished match:
    tracker.update(season_key, home, away, home_goals, away_goals)
"""
from __future__ import annotations

import numpy as np

from footy.models.experts._base import _pts


class _TableEntry:
    __slots__ = ("pts", "gf", "ga", "played", "pos")

    def __init__(self):
        self.pts = 0
        self.gf = 0
        self.ga = 0
        self.played = 0
        self.pos = 1

    def to_dict(self) -> dict:
        return {
            "pts": self.pts,
            "gf": self.gf,
            "ga": self.ga,
            "played": self.played,
            "pos": self.pos,
        }


class LeagueTableTracker:
    """Shared, efficient league table tracker.

    Maintains overall, home-only, away-only, and form tables per
    competition-season key. All experts share the same instance.
    """

    def __init__(self):
        # season_key -> {team: _TableEntry}
        self._tables: dict[str, dict[str, _TableEntry]] = {}
        self._home_tables: dict[str, dict[str, _TableEntry]] = {}
        self._away_tables: dict[str, dict[str, _TableEntry]] = {}
        # season_key -> {team: [pts, pts, ...]} for form table
        self._recent_results: dict[str, dict[str, list[int]]] = {}
        # Track if table has been re-ranked since last update
        self._dirty: dict[str, bool] = {}

    @staticmethod
    def season_key(r) -> str:
        """Derive season key from a match row."""
        comp = getattr(r, "competition", "UNK")
        dt = r.utc_date
        try:
            yr = dt.year
            m = dt.month
            season = yr if m >= 7 else yr - 1
        except Exception:
            season = 2024
        return f"{comp}_{season}"

    def _ensure_table(self, sk: str) -> dict[str, _TableEntry]:
        if sk not in self._tables:
            self._tables[sk] = {}
            self._dirty[sk] = False
        return self._tables[sk]

    def _rank(self, sk: str) -> None:
        """Re-rank table entries by pts, GD, GF."""
        table = self._tables.get(sk)
        if not table:
            return
        ranked = sorted(
            table.values(),
            key=lambda e: (-e.pts, -(e.gf - e.ga), -e.gf),
        )
        for idx, entry in enumerate(ranked):
            entry.pos = idx + 1
        self._dirty[sk] = False

    def get_table(self, sk: str) -> dict[str, _TableEntry]:
        """Get the overall table for a season key, ranked."""
        table = self._ensure_table(sk)
        if self._dirty.get(sk, False):
            self._rank(sk)
        return table

    def get_entry(self, sk: str, team: str) -> _TableEntry:
        """Get a team's table entry, creating a default if missing."""
        table = self.get_table(sk)
        if team not in table:
            entry = _TableEntry()
            entry.pos = len(table) + 1
            table[team] = entry
        return table[team]

    def get_home_table(self, sk: str) -> dict[str, _TableEntry]:
        if sk not in self._home_tables:
            self._home_tables[sk] = {}
        return self._home_tables[sk]

    def get_away_table(self, sk: str) -> dict[str, _TableEntry]:
        if sk not in self._away_tables:
            self._away_tables[sk] = {}
        return self._away_tables[sk]

    def n_teams(self, sk: str) -> int:
        return max(len(self._tables.get(sk, {})), 1)

    def all_pts_sorted(self, sk: str) -> list[int]:
        """Get all team points sorted descending."""
        table = self.get_table(sk)
        if not table:
            return [0]
        return sorted([e.pts for e in table.values()], reverse=True)

    def leader_pts(self, sk: str) -> int:
        pts = self.all_pts_sorted(sk)
        return pts[0] if pts else 0

    def relegation_boundary_pts(self, sk: str) -> int:
        """Points at the relegation boundary position."""
        n = self.n_teams(sk)
        rel_count = 3 if n >= 20 else 2
        pts = self.all_pts_sorted(sk)
        idx = max(0, n - rel_count - 1)
        return pts[min(idx, len(pts) - 1)] if pts else 0

    def euro_boundary_pts(self, sk: str) -> int:
        """Points at the European qualification boundary."""
        n = self.n_teams(sk)
        pts = self.all_pts_sorted(sk)
        idx = min(max(3, n // 5), len(pts) - 1)
        return pts[idx] if pts else 0

    def form_table(self, sk: str, n_recent: int = 6) -> dict[str, dict]:
        """Build mini form table from recent results."""
        rr = self._recent_results.get(sk, {})
        if not rr:
            return {}
        form_tbl = {}
        for team, pts_list in rr.items():
            recent = pts_list[-n_recent:]
            if recent:
                form_tbl[team] = {
                    "pts": sum(recent),
                    "ppg": sum(recent) / len(recent),
                }
        ranked = sorted(form_tbl.items(), key=lambda x: -x[1]["pts"])
        for idx, (team, entry) in enumerate(ranked):
            entry["pos"] = idx + 1
            entry["rel_pos"] = (
                1.0 - idx / max(len(ranked) - 1, 1) if len(ranked) > 1 else 0.5
            )
        return form_tbl

    def update(
        self,
        sk: str,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
    ) -> None:
        """Update all tables after a finished match. Call ONCE per match."""
        # Overall table
        table = self._ensure_table(sk)
        for team, gf, ga in [(home, home_goals, away_goals), (away, away_goals, home_goals)]:
            if team not in table:
                table[team] = _TableEntry()
            e = table[team]
            e.gf += gf
            e.ga += ga
            e.played += 1
            e.pts += _pts(gf, ga)
        self._dirty[sk] = True

        # Home-only table (home team only)
        ht = self.get_home_table(sk)
        if home not in ht:
            ht[home] = _TableEntry()
        he = ht[home]
        he.gf += home_goals
        he.ga += away_goals
        he.played += 1
        he.pts += _pts(home_goals, away_goals)

        # Away-only table (away team only)
        at = self.get_away_table(sk)
        if away not in at:
            at[away] = _TableEntry()
        ae = at[away]
        ae.gf += away_goals
        ae.ga += home_goals
        ae.played += 1
        ae.pts += _pts(away_goals, home_goals)

        # Recent results for form table
        if sk not in self._recent_results:
            self._recent_results[sk] = {}
        h_pts = _pts(home_goals, away_goals)
        a_pts = _pts(away_goals, home_goals)
        self._recent_results[sk].setdefault(home, []).append(h_pts)
        self._recent_results[sk].setdefault(away, []).append(a_pts)
