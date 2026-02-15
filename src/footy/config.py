from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def _get(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return v if v not in ("", None) else default

@dataclass(frozen=True)
class Settings:
    football_data_org_token: str
    api_football_key: str
    tracked_competitions: list[str]
    lookahead_days: int
    db_path: str

    thesportsdb_key: str | None
    sportapi_ai_key: str | None
    the_odds_api_key: str | None

    ollama_host: str
    ollama_model: str

def settings() -> Settings:
    tok = _get("FOOTBALL_DATA_ORG_TOKEN")
    afk = _get("API_FOOTBALL_KEY")
    if not tok or not afk:
        raise RuntimeError("Missing FOOTBALL_DATA_ORG_TOKEN or API_FOOTBALL_KEY. Copy .env.example to .env and fill it.")

    comps = (_get("TRACKED_COMPETITIONS", "PL,PD,SA,BL1") or "").split(",")
    comps = [c.strip() for c in comps if c.strip()]

    return Settings(
        football_data_org_token=tok,
        api_football_key=afk,
        tracked_competitions=comps,
        lookahead_days=int(_get("LOOKAHEAD_DAYS", "7") or "7"),
        db_path=_get("DB_PATH", "./data/footy.duckdb") or "./data/footy.duckdb",
        thesportsdb_key=_get("THESPORTSDB_KEY"),
        sportapi_ai_key=_get("SPORTAPI_AI_KEY"),
        the_odds_api_key=_get("THE_ODDS_API_KEY"),
        ollama_host=_get("OLLAMA_HOST", "http://localhost:11434") or "http://localhost:11434",
        ollama_model=_get("OLLAMA_MODEL", "llama3.2:3b") or "llama3.2:3b",
    )
