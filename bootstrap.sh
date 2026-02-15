#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/football-predictor"
mkdir -p "$ROOT"/{src/footy/{providers,models,llm},ui,data,logs}

cat > "$ROOT/.env.example" <<'ENV'
# --- Required (you already have these) ---
FOOTBALL_DATA_ORG_TOKEN=put_your_token_here
API_FOOTBALL_KEY=put_your_key_here

# --- Optional free additions ---
THE_ODDS_API_KEY=
SPORTAPI_AI_KEY=
THESPORTSDB_KEY=3  # dev/test key mentioned by TheSportsDB community posts

# --- App config ---
TRACKED_COMPETITIONS=PL,PD,SA,BL1,FL1,DED
LOOKAHEAD_DAYS=7
DB_PATH=./data/footy.duckdb

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
ENV

cat > "$ROOT/pyproject.toml" <<'TOML'
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "footy"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "httpx>=0.27,<1.0",
  "python-dotenv>=1.0,<2.0",
  "pydantic>=2.7,<3.0",
  "duckdb>=1.0,<2.0",
  "pandas>=2.2.3,<3.0",
  "numpy>=2.0,<3.0",
  "scipy>=1.11,<2.0",
  "scikit-learn>=1.6,<1.8",
  "typer>=0.12,<1.0",
  "rich>=13.7,<14.0",
  "streamlit>=1.29,<2.0",
  "plotly>=5.18,<6.0",
  "gdeltdoc>=1.5,<2.0",
]

[project.scripts]
footy = "footy.cli:app"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
TOML

cat > "$ROOT/src/footy/__init__.py" <<'PY'
__all__ = ["config", "db"]
PY

cat > "$ROOT/src/footy/config.py" <<'PY'
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

    comps = (_get("TRACKED_COMPETITIONS", "PL,PD,SA,BL1,FL1,DED") or "").split(",")
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
PY

cat > "$ROOT/src/footy/db.py" <<'PY'
from __future__ import annotations
import duckdb
from footy.config import settings

SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS matches (
  match_id BIGINT PRIMARY KEY,
  provider VARCHAR NOT NULL,
  competition VARCHAR,
  season INT,
  utc_date TIMESTAMP,
  status VARCHAR,
  home_team VARCHAR,
  away_team VARCHAR,
  home_goals INT,
  away_goals INT,
  raw_json VARCHAR
);

CREATE TABLE IF NOT EXISTS predictions (
  match_id BIGINT,
  model_version VARCHAR,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  p_home DOUBLE,
  p_draw DOUBLE,
  p_away DOUBLE,
  eg_home DOUBLE,
  eg_away DOUBLE,
  notes VARCHAR,
  PRIMARY KEY(match_id, model_version)
);

CREATE TABLE IF NOT EXISTS metrics (
  model_version VARCHAR PRIMARY KEY,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  n_matches INT,
  logloss DOUBLE,
  brier DOUBLE,
  accuracy DOUBLE
);

CREATE TABLE IF NOT EXISTS elo_state (
  team VARCHAR PRIMARY KEY,
  rating DOUBLE
);

CREATE TABLE IF NOT EXISTS poisson_state (
  key VARCHAR PRIMARY KEY,
  value VARCHAR
);

CREATE TABLE IF NOT EXISTS news (
  team VARCHAR,
  seendate TIMESTAMP,
  title VARCHAR,
  url VARCHAR,
  domain VARCHAR,
  tone DOUBLE,
  source VARCHAR,
  fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def connect() -> duckdb.DuckDBPyConnection:
    s = settings()
    con = duckdb.connect(s.db_path)
    con.execute(SCHEMA_SQL)
    return con
PY

cat > "$ROOT/src/footy/providers/ratelimit.py" <<'PY'
from __future__ import annotations
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_calls: int, period_seconds: int):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls = deque()

    def wait(self):
        now = time.monotonic()
        while self.calls and now - self.calls[0] > self.period:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            sleep_for = self.period - (now - self.calls[0]) + 0.01
            time.sleep(max(0.0, sleep_for))
        self.calls.append(time.monotonic())
PY

cat > "$ROOT/src/footy/providers/football_data_org.py" <<'PY'
from __future__ import annotations
import json
import httpx
from datetime import date
from footy.config import settings
from footy.providers.ratelimit import RateLimiter

BASE = "https://api.football-data.org/v4"

_rl = RateLimiter(max_calls=10, period_seconds=60)  # free plan 10/min :contentReference[oaicite:11]{index=11}

def _client():
    s = settings()
    return httpx.Client(
        headers={"X-Auth-Token": s.football_data_org_token},
        timeout=30.0
    )

def fetch_matches(date_from: date, date_to: date) -> dict:
    # One call returns matches in your subscribed competitions (most efficient)
    # Example documented: /v4/matches :contentReference[oaicite:12]{index=12}
    _rl.wait()
    with _client() as c:
        r = c.get(f"{BASE}/matches", params={"dateFrom": str(date_from), "dateTo": str(date_to)})
        r.raise_for_status()
        return r.json()

def normalize_match(m: dict) -> dict:
    score = (m.get("score") or {}).get("fullTime") or {}
    home_goals = score.get("home")
    away_goals = score.get("away")
    comp = (m.get("competition") or {}).get("code")
    season = (m.get("season") or {}).get("startDate", "")[:4]
    return {
        "match_id": int(m["id"]),
        "provider": "football-data.org",
        "competition": comp,
        "season": int(season) if season.isdigit() else None,
        "utc_date": m.get("utcDate"),
        "status": m.get("status"),
        "home_team": (m.get("homeTeam") or {}).get("name"),
        "away_team": (m.get("awayTeam") or {}).get("name"),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "raw_json": json.dumps(m),
    }
PY

cat > "$ROOT/src/footy/providers/football_data_co_uk.py" <<'PY'
from __future__ import annotations
import pandas as pd

# Data download page shows it is actively updated (e.g., Feb 12, 2026) :contentReference[oaicite:13]{index=13}
FIXTURES_CSV = "https://www.football-data.co.uk/fixtures.csv"

def fetch_fixtures() -> pd.DataFrame:
    # The fixtures CSV format can change; we keep this as optional fallback.
    df = pd.read_csv(FIXTURES_CSV)
    return df
PY

cat > "$ROOT/src/footy/providers/clubelo.py" <<'PY'
from __future__ import annotations
import pandas as pd

# ClubElo CSV API endpoint :contentReference[oaicite:14]{index=14}
BASE = "http://api.clubelo.com"

def fetch_latest() -> pd.DataFrame:
    # Base URL returns latest ratings in CSV
    return pd.read_csv(BASE)
PY

cat > "$ROOT/src/footy/providers/news_gdelt.py" <<'PY'
from __future__ import annotations
from datetime import datetime, timedelta
import pandas as pd
from gdeltdoc import GdeltDoc, Filters

# GDELT provides live JSON APIs including DOC :contentReference[oaicite:15]{index=15}
# gdeltdoc Article List returns url/title/seendate/etc :contentReference[oaicite:16]{index=16}

def fetch_team_news(team: str, days_back: int = 3, max_records: int = 50) -> pd.DataFrame:
    end = datetime.utcnow().date()
    start = end - timedelta(days=days_back)
    f = Filters(
        keyword=team,
        start_date=str(start),
        end_date=str(end),
        num_records=max_records,
    )
    gd = GdeltDoc()
    df = gd.article_search(f)
    # df columns: url, title, seendate, domain, language, sourcecountry, etc.
    return df
PY

cat > "$ROOT/src/footy/llm/ollama_client.py" <<'PY'
from __future__ import annotations
import httpx
from footy.config import settings

# Ollama API docs show /api/chat :contentReference[oaicite:17]{index=17}

def chat(messages: list[dict], model: str | None = None) -> str:
    s = settings()
    payload = {"model": model or s.ollama_model, "messages": messages, "stream": False}
    with httpx.Client(timeout=60.0) as c:
        r = c.post(f"{s.ollama_host}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "").strip()
PY

cat > "$ROOT/src/footy/llm/news_extractor.py" <<'PY'
from __future__ import annotations
import json
from pydantic import BaseModel, Field, ValidationError
from footy.llm.ollama_client import chat

class NewsSignal(BaseModel):
    availability_score: float = Field(..., ge=-1.0, le=1.0, description="Negative = bad availability/news, positive = good")
    likely_absences: list[str] = Field(default_factory=list)
    key_notes: list[str] = Field(default_factory=list)
    short_summary: str

def extract_news_signal(team: str, headlines: list[dict]) -> NewsSignal:
    # headlines: [{"title":..., "domain":..., "seendate":...}, ...]
    prompt = {
        "role": "user",
        "content": (
            f"You are extracting structured team-news signals for {team}.\n"
            "Given ONLY these headlines (no browsing), infer availability/news impact.\n"
            "Return STRICT JSON matching this schema:\n"
            "{availability_score: number -1..1, likely_absences: string[], key_notes: string[], short_summary: string}\n\n"
            f"HEADLINES:\n{json.dumps(headlines, ensure_ascii=False)[:6000]}"
        )
    }
    txt = chat([prompt])
    # best-effort JSON parse + validate
    try:
        obj = json.loads(txt)
        return NewsSignal.model_validate(obj)
    except (json.JSONDecodeError, ValidationError):
        # fallback: ask model to output only JSON
        txt2 = chat([{"role":"user","content":"Output ONLY valid JSON. No prose.\n\n"+prompt["content"]}])
        obj2 = json.loads(txt2)
        return NewsSignal.model_validate(obj2)
PY

cat > "$ROOT/src/footy/models/elo.py" <<'PY'
from __future__ import annotations
import math
from footy.db import connect

DEFAULT_RATING = 1500.0
HOME_ADV = 60.0
K = 20.0

def _expected(r_home: float, r_away: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_home - r_away) / 400.0))

def ensure_team(con, team: str):
    con.execute("INSERT OR IGNORE INTO elo_state(team, rating) VALUES (?, ?)", [team, DEFAULT_RATING])

def get_rating(con, team: str) -> float:
    ensure_team(con, team)
    return float(con.execute("SELECT rating FROM elo_state WHERE team=?", [team]).fetchone()[0])

def set_rating(con, team: str, rating: float):
    con.execute("UPDATE elo_state SET rating=? WHERE team=?", [rating, team])

def update_from_match(con, home: str, away: str, hg: int, ag: int):
    r_home = get_rating(con, home) + HOME_ADV
    r_away = get_rating(con, away)
    exp_home = _expected(r_home, r_away)

    if hg > ag:
        s_home = 1.0
    elif hg == ag:
        s_home = 0.5
    else:
        s_home = 0.0

    delta = K * (s_home - exp_home)
    set_rating(con, home, get_rating(con, home) + delta)
    set_rating(con, away, get_rating(con, away) - delta)

def predict_probs(con, home: str, away: str) -> tuple[float, float, float]:
    r_home = get_rating(con, home) + HOME_ADV
    r_away = get_rating(con, away)
    p_home = _expected(r_home, r_away)
    # simple draw heuristic (improved later by the meta-model)
    p_draw = 0.26
    p_home_adj = p_home * (1.0 - p_draw)
    p_away_adj = (1.0 - p_home) * (1.0 - p_draw)
    s = p_home_adj + p_draw + p_away_adj
    return (p_home_adj/s, p_draw/s, p_away_adj/s)
PY

cat > "$ROOT/src/footy/models/poisson.py" <<'PY'
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def fit_poisson(matches: pd.DataFrame, halflife_days: float = 180.0) -> dict:
    """
    Fits team attack/defense + home advantage using weighted Poisson likelihood.
    matches columns: home_team, away_team, home_goals, away_goals, utc_date
    """
    df = matches.dropna(subset=["home_goals","away_goals","utc_date"]).copy()
    if df.empty:
        return {"teams": [], "attack": [], "defense": [], "home_adv": 0.0, "mu": 0.0}

    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)
    tmax = df["utc_date"].max()
    age_days = (tmax - df["utc_date"]).dt.total_seconds() / 86400.0
    w = np.exp(-np.log(2) * age_days / halflife_days).to_numpy()

    teams = pd.Index(sorted(set(df["home_team"]) | set(df["away_team"])))
    idx = {t:i for i,t in enumerate(teams)}
    n = len(teams)

    h = df["home_team"].map(idx).to_numpy()
    a = df["away_team"].map(idx).to_numpy()
    hg = df["home_goals"].to_numpy(dtype=float)
    ag = df["away_goals"].to_numpy(dtype=float)

    # parameters: [mu, home_adv, attack(0..n-2), defense(0..n-2)]
    # last team fixed at 0 for identifiability
    x0 = np.zeros(2 + 2*(n-1), dtype=float)

    def unpack(x):
        mu = x[0]
        ha = x[1]
        att = np.zeros(n)
        deff = np.zeros(n)
        att[:-1] = x[2:2+(n-1)]
        deff[:-1] = x[2+(n-1):]
        return mu, ha, att, deff

    def nll(x):
        mu, ha, att, deff = unpack(x)
        lam_h = np.exp(mu + ha + att[h] - deff[a])
        lam_a = np.exp(mu + att[a] - deff[h])
        # Poisson loglik: k*log(lam) - lam - log(k!)
        ll = (hg*np.log(lam_h) - lam_h) + (ag*np.log(lam_a) - lam_a)
        return -np.sum(w * ll)

    def reg(x, l2=0.05):
        return l2 * np.sum(x[2:]**2)

    def obj(x):
        return nll(x) + reg(x)

    res = minimize(obj, x0, method="L-BFGS-B")
    mu, ha, att, deff = unpack(res.x)
    return {"teams": teams.tolist(), "attack": att.tolist(), "defense": deff.tolist(), "home_adv": float(ha), "mu": float(mu)}

def expected_goals(state: dict, home_team: str, away_team: str) -> tuple[float,float]:
    teams = state["teams"]
    if not teams:
        return (1.3, 1.1)
    idx = {t:i for i,t in enumerate(teams)}
    if home_team not in idx or away_team not in idx:
        return (1.3, 1.1)
    mu = state["mu"]; ha = state["home_adv"]
    att = np.array(state["attack"]); deff = np.array(state["defense"])
    i = idx[home_team]; j = idx[away_team]
    lam_h = float(np.exp(mu + ha + att[i] - deff[j]))
    lam_a = float(np.exp(mu + att[j] - deff[i]))
    return lam_h, lam_a

def scoreline_probs(lam_h: float, lam_a: float, max_goals: int = 8) -> np.ndarray:
    # matrix [hg, ag]
    hg = np.arange(max_goals+1)
    ag = np.arange(max_goals+1)
    ph = np.exp(-lam_h) * np.power(lam_h, hg) / np.array([math.factorial(int(k)) for k in hg])
    pa = np.exp(-lam_a) * np.power(lam_a, ag) / np.array([math.factorial(int(k)) for k in ag])
    return np.outer(ph, pa)

def outcome_probs(lam_h: float, lam_a: float) -> tuple[float,float,float]:
    M = scoreline_probs(lam_h, lam_a, max_goals=8)
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())
    s = p_home + p_draw + p_away
    return (p_home/s, p_draw/s, p_away/s)
PY

cat > "$ROOT/src/footy/pipeline.py" <<'PY'
from __future__ import annotations
import json
from datetime import date, timedelta
import pandas as pd
from footy.db import connect
from footy.config import settings
from footy.providers.football_data_org import fetch_matches, normalize_match
from footy.providers.news_gdelt import fetch_team_news
from footy.models import elo
from footy.models.poisson import fit_poisson, expected_goals, outcome_probs

MODEL_VERSION = "v1_elo_poisson"

def ingest(days_back: int = 30, days_forward: int = 7) -> int:
    con = connect()
    d0 = date.today() - timedelta(days=days_back)
    d1 = date.today() + timedelta(days=days_forward)
    raw = fetch_matches(d0, d1)
    matches = raw.get("matches", [])
    s = settings()
    n = 0
    for m in matches:
        nm = normalize_match(m)
        # filter to tracked competitions
        if nm["competition"] and nm["competition"] not in s.tracked_competitions:
            continue
        con.execute(
            """INSERT OR REPLACE INTO matches
               (match_id, provider, competition, season, utc_date, status, home_team, away_team, home_goals, away_goals, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                nm["match_id"], nm["provider"], nm["competition"], nm["season"],
                nm["utc_date"], nm["status"], nm["home_team"], nm["away_team"],
                nm["home_goals"], nm["away_goals"], nm["raw_json"]
            ],
        )
        n += 1
    return n

def update_elo_from_finished() -> int:
    con = connect()
    rows = con.execute(
        """SELECT match_id, home_team, away_team, home_goals, away_goals
           FROM matches
           WHERE status='FINISHED' AND home_goals IS NOT NULL AND away_goals IS NOT NULL
           ORDER BY utc_date ASC"""
    ).fetchall()
    count = 0
    for _, home, away, hg, ag in rows:
        elo.update_from_match(con, home, away, int(hg), int(ag))
        count += 1
    return count

def refit_poisson() -> dict:
    con = connect()
    df = con.execute(
        """SELECT home_team, away_team, home_goals, away_goals, utc_date
           FROM matches
           WHERE status='FINISHED' AND home_goals IS NOT NULL AND away_goals IS NOT NULL"""
    ).df()
    state = fit_poisson(df)
    con.execute("INSERT OR REPLACE INTO poisson_state(key, value) VALUES ('state', ?)", [json.dumps(state)])
    return state

def load_poisson() -> dict:
    con = connect()
    row = con.execute("SELECT value FROM poisson_state WHERE key='state'").fetchone()
    if not row:
        return {"teams": [], "attack": [], "defense": [], "home_adv": 0.0, "mu": 0.0}
    return json.loads(row[0])

def predict_upcoming(lookahead_days: int | None = None) -> int:
    con = connect()
    s = settings()
    look = lookahead_days or s.lookahead_days
    df = con.execute(
        f"""SELECT match_id, home_team, away_team, utc_date
            FROM matches
            WHERE status='SCHEDULED' AND utc_date <= (CURRENT_TIMESTAMP + INTERVAL {look} DAY)
            ORDER BY utc_date ASC"""
    ).df()

    state = load_poisson()
    n = 0
    for _, r in df.iterrows():
        mid = int(r["match_id"])
        home = r["home_team"]; away = r["away_team"]

        pE = elo.predict_probs(con, home, away)
        lam_h, lam_a = expected_goals(state, home, away)
        pP = outcome_probs(lam_h, lam_a)

        # simple blend (meta-model comes later)
        p_home = 0.45*pE[0] + 0.55*pP[0]
        p_draw = 0.45*pE[1] + 0.55*pP[1]
        p_away = 0.45*pE[2] + 0.55*pP[2]
        ssum = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home/ssum, p_draw/ssum, p_away/ssum

        con.execute(
            """INSERT OR REPLACE INTO predictions
               (match_id, model_version, p_home, p_draw, p_away, eg_home, eg_away, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [mid, MODEL_VERSION, p_home, p_draw, p_away, lam_h, lam_a, "blend(elo,poisson)"]
        )
        n += 1
    return n

def backtest_metrics() -> dict:
    con = connect()
    df = con.execute(
        """SELECT p.match_id, p.p_home, p.p_draw, p.p_away,
                  m.home_goals, m.away_goals
           FROM predictions p
           JOIN matches m USING(match_id)
           WHERE p.model_version=? AND m.status='FINISHED'
             AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL""",
        [MODEL_VERSION]
    ).df()
    if df.empty:
        return {"n": 0, "logloss": None, "brier": None, "accuracy": None}

    y = []
    P = df[["p_home","p_draw","p_away"]].to_numpy()
    for hg, ag in zip(df["home_goals"], df["away_goals"]):
        if hg > ag: y.append(0)
        elif hg == ag: y.append(1)
        else: y.append(2)
    y = pd.Series(y).to_numpy()

    # logloss
    eps = 1e-12
    ll = 0.0
    for i, yi in enumerate(y):
        ll += -float(np.log(P[i, yi] + eps))
    logloss = ll / len(y)

    # brier (multi-class)
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    brier = float(np.mean(np.sum((P - Y)**2, axis=1)))

    # accuracy
    acc = float(np.mean(np.argmax(P, axis=1) == y))

    con.execute("INSERT OR REPLACE INTO metrics(model_version, n_matches, logloss, brier, accuracy) VALUES (?, ?, ?, ?, ?)",
                [MODEL_VERSION, int(len(y)), float(logloss), float(brier), float(acc)])
    return {"n": int(len(y)), "logloss": float(logloss), "brier": float(brier), "accuracy": float(acc)}

def ingest_news_for_teams(days_back: int = 3, max_records: int = 30) -> int:
    con = connect()
    teams = con.execute(
        """SELECT DISTINCT home_team FROM matches
           UNION
           SELECT DISTINCT away_team FROM matches"""
    ).fetchall()
    n = 0
    for (t,) in teams:
        if not t:
            continue
        df = fetch_team_news(t, days_back=days_back, max_records=max_records)
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            con.execute(
                "INSERT INTO news(team, seendate, title, url, domain, tone, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [t, r.get("seendate"), r.get("title"), r.get("url"), r.get("domain"), None, "gdelt"]
            )
            n += 1
    return n
PY

cat > "$ROOT/src/footy/cli.py" <<'PY'
from __future__ import annotations
import typer
from rich import print
from footy import pipeline

app = typer.Typer(add_completion=False)

@app.command()
def ingest(days_back: int = 30, days_forward: int = 7):
    n = pipeline.ingest(days_back=days_back, days_forward=days_forward)
    print(f"[green]Inserted/updated[/green] {n} matches")

@app.command()
def train():
    n = pipeline.update_elo_from_finished()
    print(f"[green]Elo updated from[/green] {n} finished matches")
    state = pipeline.refit_poisson()
    print(f"[green]Poisson refit[/green] teams={len(state.get('teams', []))}")

@app.command()
def predict():
    n = pipeline.predict_upcoming()
    print(f"[green]Predictions written[/green] {n} upcoming matches")

@app.command()
def metrics():
    m = pipeline.backtest_metrics()
    print(m)

@app.command()
def update():
    # one-shot: ingest -> train -> predict -> metrics
    pipeline.ingest()
    pipeline.update_elo_from_finished()
    pipeline.refit_poisson()
    pipeline.predict_upcoming()
    m = pipeline.backtest_metrics()
    print(f"[cyan]metrics[/cyan] {m}")

@app.command()
def news(days_back: int = 3, max_records: int = 30):
    n = pipeline.ingest_news_for_teams(days_back=days_back, max_records=max_records)
    print(f"[green]Inserted[/green] {n} news rows")

if __name__ == "__main__":
    app()
PY

cat > "$ROOT/ui/app.py" <<'PY'
from __future__ import annotations
import duckdb
import streamlit as st
import pandas as pd
from footy.config import settings
from footy.db import connect
from footy.llm.news_extractor import extract_news_signal

st.set_page_config(page_title="Footy Predictor", layout="wide")

@st.cache_resource
def db():
    return connect()

con = db()
s = settings()

st.title("⚽ Footy Predictor (Local)")

tabs = st.tabs(["Upcoming", "Match detail", "Backtest", "Data/Health"])

with tabs[0]:
    st.subheader("Upcoming matches")
    df = con.execute(
        f"""SELECT m.utc_date, m.competition, m.home_team, m.away_team,
                   p.p_home, p.p_draw, p.p_away, p.eg_home, p.eg_away, m.match_id
            FROM matches m
            LEFT JOIN predictions p ON p.match_id=m.match_id AND p.model_version='v1_elo_poisson'
            WHERE m.status='SCHEDULED'
              AND m.utc_date <= (CURRENT_TIMESTAMP + INTERVAL {s.lookahead_days} DAY)
            ORDER BY m.utc_date ASC"""
    ).df()
    if df.empty:
        st.info("No upcoming matches in DB yet. Run: footy update")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Match detail")
    mids = con.execute(
        "SELECT match_id, utc_date, home_team, away_team FROM matches WHERE status='SCHEDULED' ORDER BY utc_date ASC"
    ).df()
    if mids.empty:
        st.info("No scheduled matches. Run: footy ingest / footy update")
    else:
        label = mids.apply(lambda r: f'{r.utc_date} — {r.home_team} vs {r.away_team} (#{int(r.match_id)})', axis=1)
        choice = st.selectbox("Choose match", options=list(range(len(mids))), format_func=lambda i: label.iloc[i])
        row = mids.iloc[int(choice)]
        mid = int(row.match_id)
        home = row.home_team; away = row.away_team

        pred = con.execute(
            """SELECT p_home, p_draw, p_away, eg_home, eg_away, notes
               FROM predictions WHERE match_id=? AND model_version='v1_elo_poisson'""",
            [mid]
        ).fetchone()

        c1, c2 = st.columns(2)
        if pred:
            p_home, p_draw, p_away, eg_h, eg_a, notes = pred
            c1.metric("Home win", f"{p_home:.1%}")
            c1.metric("Draw", f"{p_draw:.1%}")
            c1.metric("Away win", f"{p_away:.1%}")
            c2.metric("Expected goals (H)", f"{eg_h:.2f}")
            c2.metric("Expected goals (A)", f"{eg_a:.2f}")
            st.caption(f"Model notes: {notes}")
        else:
            st.warning("No prediction yet. Run: footy predict (or footy update)")

        st.divider()
        st.subheader("Team news (GDELT) + optional Ollama summary")
        for team in [home, away]:
            st.markdown(f"### {team}")
            news = con.execute(
                """SELECT seendate, title, domain, url
                   FROM news WHERE team=? ORDER BY seendate DESC LIMIT 12""",
                [team]
            ).df()
            if news.empty:
                st.write("No news cached yet. Run: footy news")
            else:
                st.dataframe(news[["seendate","title","domain","url"]], use_container_width=True, hide_index=True)

                if st.button(f"Summarize with Ollama: {team}", key=f"oll_{team}"):
                    headlines = [{"title": t, "domain": d, "seendate": str(sd)} for sd, t, d in zip(news["seendate"], news["title"], news["domain"])]
                    sig = extract_news_signal(team, headlines=headlines)
                    st.json(sig.model_dump())

with tabs[2]:
    st.subheader("Backtest metrics")
    met = con.execute("SELECT * FROM metrics ORDER BY created_at DESC").df()
    if met.empty:
        st.info("No metrics yet. Run: footy metrics (or footy update).")
    else:
        st.dataframe(met, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Data/Health")
    st.write("Tracked competitions:", s.tracked_competitions)
    counts = con.execute(
        """SELECT status, COUNT(*) AS n
           FROM matches GROUP BY status ORDER BY n DESC"""
    ).df()
    st.dataframe(counts, use_container_width=True, hide_index=True)
PY

chmod +x "$ROOT/bootstrap.sh"

echo "Bootstrap complete."
echo "Next:"
echo "  1) cp .env.example .env && edit .env"
echo "  2) pip install -e ."
echo "  3) footy update"
echo "  4) streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501"
