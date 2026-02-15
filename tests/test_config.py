"""Unit tests — Config & Settings."""
import os
import pytest


class TestSettings:
    """Config loading and validation."""

    def test_settings_loads(self):
        from footy.config import settings

        s = settings()
        assert s.football_data_org_token
        assert s.api_football_key
        assert isinstance(s.tracked_competitions, list)
        assert len(s.tracked_competitions) > 0

    def test_tracked_competitions_default(self):
        from footy.config import settings

        s = settings()
        assert "PL" in s.tracked_competitions

    def test_db_path_has_value(self):
        from footy.config import settings

        s = settings()
        assert s.db_path  # non-empty

    def test_ollama_defaults(self):
        from footy.config import settings

        s = settings()
        assert "localhost" in s.ollama_host or "11434" in s.ollama_host
        assert s.ollama_model  # non-empty
