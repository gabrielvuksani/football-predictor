"""Unit tests — Config & Settings."""
import os
import pytest


class TestSettings:
    """Config loading and validation."""

    def test_settings_loads(self):
        from footy.config import settings

        s = settings()
        assert isinstance(s.tracked_competitions, (list, tuple))
        assert len(s.tracked_competitions) > 0
        assert isinstance(s.enable_scraper_stack, bool)
        assert isinstance(s.enable_understat, bool)

    def test_tracked_competitions_default(self):
        from footy.config import settings

        s = settings()
        assert "PL" in s.tracked_competitions
        assert "DED" in s.tracked_competitions

    def test_db_path_has_value(self):
        from footy.config import settings

        s = settings()
        assert s.db_path  # non-empty

    def test_ollama_defaults(self):
        from footy.config import settings

        s = settings()
        assert "localhost" in s.ollama_host or "11434" in s.ollama_host
        assert s.ollama_model  # non-empty

    def test_new_llm_defaults(self):
        from footy.config import settings

        s = settings()
        assert s.groq_model
        assert "ollama" in s.llm_provider_order
