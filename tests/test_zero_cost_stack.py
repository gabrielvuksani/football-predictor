from __future__ import annotations

from datetime import datetime, timezone


def test_openfootball_competition_mapping_imports():
    from footy.providers.openfootball import OPENFOOTBALL_COMPETITIONS

    assert OPENFOOTBALL_COMPETITIONS["PL"] == "en.1"
    assert "DED" in OPENFOOTBALL_COMPETITIONS


def test_llm_provider_router_falls_back(monkeypatch):
    from footy.llm.providers import chat

    monkeypatch.setattr("footy.llm.groq_client.chat", lambda *args, **kwargs: "")
    monkeypatch.setattr("footy.llm.ollama_client.chat", lambda *args, **kwargs: "ok")

    assert chat([{"role": "user", "content": "hi"}]) == "ok"


def test_open_meteo_sum_recent_helper():
    from footy.providers.open_meteo import OpenMeteoProvider

    hourly = {"precipitation": [0.0, 1.0, 2.0, 3.0]}
    assert OpenMeteoProvider._sum_recent(hourly, "precipitation", 3, 2) == 6.0


def test_data_orchestrator_imports():
    from footy.data_orchestrator import DataOrchestrator

    orch = DataOrchestrator()
    assert orch.settings.enable_scraper_stack is True
    orch.close()


def test_openfootball_provider_url_builder():
    from footy.providers.openfootball import OpenFootballProvider

    provider = OpenFootballProvider()
    url = provider.season_url("PL", "2025-26")
    assert url.endswith("/2025-26/en.1.json")
    provider.close()
