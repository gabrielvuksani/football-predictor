from __future__ import annotations
import functools
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def _get(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return v if v not in ("", None) else default


def _get_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class Settings:
    football_data_org_token: str | None
    api_football_key: str | None
    tracked_competitions: tuple[str, ...]
    lookahead_days: int
    db_path: str

    thesportsdb_key: str | None
    the_odds_api_key: str | None

    enable_scraper_stack: bool
    enable_understat: bool
    enable_fbref: bool
    enable_sofascore: bool
    enable_transfermarkt: bool
    enable_oddsportal: bool
    enable_clubelo: bool
    enable_openfootball: bool
    enable_open_meteo: bool

    request_timeout_seconds: int
    scraper_cache_ttl_seconds: int
    pre_match_refresh_minutes: int
    odds_movement_threshold: float

    ollama_host: str
    ollama_model: str
    groq_api_key: str | None
    groq_model: str
    llm_provider_order: tuple[str, ...]
    open_meteo_commercial_use: bool

    # Prediction aggregation weights
    prediction_weight_council: float
    prediction_weight_bayesian: float
    prediction_weight_statistical: float
    prediction_weight_market: float

    # Self-learning thresholds
    self_learning_accuracy_threshold: float
    self_learning_ece_threshold: float

    # Drift detection threshold
    drift_detection_threshold: float

    # Kelly criterion fraction
    kelly_fraction: float

    # Calibration settings
    calibration_method: str

    # Retraining parameters
    retrain_threshold_matches: int
    retrain_improvement_threshold: float

    # Data lookback windows
    xg_lookback_days: int
    walkforward_embargo_days: int

    def __repr__(self) -> str:
        def _mask(v: str | None) -> str:
            if not v:
                return repr(v)
            return repr(v[:4] + "***") if len(v) > 4 else repr("***")
        fields = ", ".join(
            f"{f}={_mask(getattr(self, f)) if 'key' in f.lower() or 'token' in f.lower() else repr(getattr(self, f))}"
            for f in self.__dataclass_fields__
        )
        return f"Settings({fields})"

@functools.lru_cache(maxsize=1)
def settings() -> Settings:
    tok = _get("FOOTBALL_DATA_ORG_TOKEN")
    afk = _get("API_FOOTBALL_KEY")

    comps = (_get(
        "TRACKED_COMPETITIONS",
        "PL,PD,SA,BL1,FL1,DED,ELC,PPL,TR1,BEL,GR1"
    ) or "").split(",")
    comps = tuple(c.strip() for c in comps if c.strip())

    llm_order = (_get("LLM_PROVIDER_ORDER", "groq,ollama") or "groq,ollama").split(",")
    llm_order = tuple(p.strip().lower() for p in llm_order if p.strip())

    return Settings(
        football_data_org_token=tok,
        api_football_key=afk,
        tracked_competitions=comps,
        lookahead_days=int(_get("LOOKAHEAD_DAYS", "7") or "7"),
        db_path=_get("DB_PATH", "./data/footy.duckdb") or "./data/footy.duckdb",
        thesportsdb_key=_get("THESPORTSDB_KEY"),
        the_odds_api_key=_get("THE_ODDS_API_KEY"),
        enable_scraper_stack=_get_bool("ENABLE_SCRAPER_STACK", True),
        enable_understat=_get_bool("ENABLE_UNDERSTAT", True),
        enable_fbref=_get_bool("ENABLE_FBREF", True),
        enable_sofascore=_get_bool("ENABLE_SOFASCORE", True),
        enable_transfermarkt=_get_bool("ENABLE_TRANSFERMARKT", True),
        enable_oddsportal=_get_bool("ENABLE_ODDSPORTAL", True),
        enable_clubelo=_get_bool("ENABLE_CLUBELO", True),
        enable_openfootball=_get_bool("ENABLE_OPENFOOTBALL", True),
        enable_open_meteo=_get_bool("ENABLE_OPEN_METEO", True),
        request_timeout_seconds=int(_get("REQUEST_TIMEOUT_SECONDS", "20") or "20"),
        scraper_cache_ttl_seconds=int(_get("SCRAPER_CACHE_TTL_SECONDS", "900") or "900"),
        pre_match_refresh_minutes=int(_get("PRE_MATCH_REFRESH_MINUTES", "15") or "15"),
        odds_movement_threshold=float(_get("ODDS_MOVEMENT_THRESHOLD", "0.05") or "0.05"),
        ollama_host=_get("OLLAMA_HOST", "http://localhost:11434") or "http://localhost:11434",
        ollama_model=_get("OLLAMA_MODEL", "llama3.2:3b") or "llama3.2:3b",
        groq_api_key=_get("GROQ_API_KEY"),
        groq_model=_get("GROQ_MODEL", "llama-3.3-70b-versatile") or "llama-3.3-70b-versatile",
        llm_provider_order=llm_order,
        open_meteo_commercial_use=_get_bool("OPEN_METEO_COMMERCIAL_USE", False),
        # v12: Market weight increased from 0.15 to 0.30 based on research
        # showing Pinnacle closing line has r²=0.997 across 400K matches.
        # Council weight reduced proportionally.
        prediction_weight_council=float(_get("PREDICTION_WEIGHT_COUNCIL", "0.35") or "0.35"),
        prediction_weight_bayesian=float(_get("PREDICTION_WEIGHT_BAYESIAN", "0.15") or "0.15"),
        prediction_weight_statistical=float(_get("PREDICTION_WEIGHT_STATISTICAL", "0.20") or "0.20"),
        prediction_weight_market=float(_get("PREDICTION_WEIGHT_MARKET", "0.30") or "0.30"),
        self_learning_accuracy_threshold=float(_get("SELF_LEARNING_ACCURACY_THRESHOLD", "0.38") or "0.38"),
        self_learning_ece_threshold=float(_get("SELF_LEARNING_ECE_THRESHOLD", "0.12") or "0.12"),
        drift_detection_threshold=float(_get("DRIFT_DETECTION_THRESHOLD", "0.05") or "0.05"),
        kelly_fraction=float(_get("KELLY_FRACTION", "0.25") or "0.25"),
        calibration_method=_get("CALIBRATION_METHOD", "auto") or "auto",
        retrain_threshold_matches=int(_get("RETRAIN_THRESHOLD_MATCHES", "20") or "20"),
        retrain_improvement_threshold=float(_get("RETRAIN_IMPROVEMENT_THRESHOLD", "0.005") or "0.005"),
        xg_lookback_days=int(_get("XG_LOOKBACK_DAYS", "180") or "180"),
        walkforward_embargo_days=int(_get("WALKFORWARD_EMBARGO_DAYS", "3") or "3"),
    )
