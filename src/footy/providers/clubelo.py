from __future__ import annotations

import csv
import io
from datetime import date
from typing import Any

from footy.normalize import canonical_team_name
from footy.providers.base import BaseProvider


class ClubEloProvider(BaseProvider):
    name = "clubelo"
    BASE_URL = "https://api.clubelo.com"

    def fetch_ratings(self, target_date: date | str | None = None) -> list[dict[str, Any]]:
        slug = str(target_date) if target_date else ""
        url = f"{self.BASE_URL}/{slug}".rstrip("/")
        raw = self.fetch_text(url, ttl=60 * 60)
        reader = csv.DictReader(io.StringIO(raw))
        rows: list[dict[str, Any]] = []
        for row in reader:
            team = canonical_team_name(row.get("Club") or row.get("Team"))
            if not team:
                continue
            rows.append({
                "team": team,
                "country": row.get("Country"),
                "level": row.get("Level"),
                "elo": float(row.get("Elo") or 0.0),
                "from": row.get("From"),
                "to": row.get("To"),
            })
        return rows

    def fetch_team_history(self, team: str) -> list[dict[str, Any]]:
        url = f"{self.BASE_URL}/{team.replace(' ', '%20')}"
        raw = self.fetch_text(url, ttl=6 * 60 * 60)
        reader = csv.DictReader(io.StringIO(raw))
        history: list[dict[str, Any]] = []
        for row in reader:
            if not row:
                continue
            try:
                elo = float(row.get("Elo") or 0.0)
            except (TypeError, ValueError):
                continue
            history.append({
                "date": row.get("From") or row.get("Date"),
                "team": canonical_team_name(row.get("Club") or row.get("Team") or team),
                "elo": elo,
            })
        return history

    def get_team_rating(self, team: str, target_date: date | str | None = None) -> float | None:
        canonical = canonical_team_name(team)
        for row in self.fetch_ratings(target_date):
            if row["team"] == canonical:
                return row["elo"]
        return None
