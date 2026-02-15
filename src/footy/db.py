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

-- Per-match scoring once actual result is known
CREATE TABLE IF NOT EXISTS prediction_scores (
  match_id BIGINT,
  model_version VARCHAR,
  scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  outcome INT,              -- 0=H,1=D,2=A
  logloss DOUBLE,
  brier DOUBLE,
  correct BOOLEAN,
  -- Goal prediction accuracy
  goals_mae DOUBLE,         -- mean absolute error of predicted goals
  eg_home DOUBLE,           -- predicted expected goals home
  eg_away DOUBLE,           -- predicted expected goals away
  -- Market prediction accuracy
  btts_correct BOOLEAN,     -- was BTTS prediction correct?
  ou25_correct BOOLEAN,     -- was O/U 2.5 prediction correct?
  score_correct BOOLEAN,    -- was exact score prediction correct?
  p_btts DOUBLE,            -- predicted BTTS probability
  p_o25 DOUBLE,             -- predicted Over 2.5 probability
  predicted_score_h INT,    -- predicted home score
  predicted_score_a INT,    -- predicted away score
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

CREATE TABLE IF NOT EXISTS elo_applied (
  match_id BIGINT PRIMARY KEY,
  applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS poisson_state (
  key VARCHAR PRIMARY KEY,
  value VARCHAR
);

-- Extra stats/odds from football-data.co.uk rows (used only for rolling aggregates; never used directly for same match)
CREATE TABLE IF NOT EXISTS match_extras (
  match_id BIGINT PRIMARY KEY,
  provider VARCHAR,
  competition VARCHAR,
  season_code VARCHAR,
  div_code VARCHAR,
  -- 1X2 odds: opening B365
  b365h DOUBLE, b365d DOUBLE, b365a DOUBLE,
  -- 1X2 odds: closing B365
  b365ch DOUBLE, b365cd DOUBLE, b365ca DOUBLE,
  -- 1X2 odds: Pinnacle (sharpest book)
  psh DOUBLE, psd DOUBLE, psa DOUBLE,
  -- 1X2 odds: market average & max
  avgh DOUBLE, avgd DOUBLE, avga DOUBLE,
  maxh DOUBLE, maxd DOUBLE, maxa DOUBLE,
  -- Over/Under 2.5 odds
  b365_o25 DOUBLE, b365_u25 DOUBLE,
  avg_o25 DOUBLE, avg_u25 DOUBLE,
  max_o25 DOUBLE, max_u25 DOUBLE,
  -- Asian Handicap
  b365ahh DOUBLE, b365ahha DOUBLE, b365ahaw DOUBLE,
  -- Half-time scores
  hthg INT, htag INT,
  -- basic stats (common but not guaranteed across seasons/leagues)
  hs DOUBLE, as_ DOUBLE, hst DOUBLE, ast DOUBLE,
  hc DOUBLE, ac DOUBLE,
  hy DOUBLE, ay DOUBLE,
  hr DOUBLE, ar DOUBLE,
  raw_json VARCHAR
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
CREATE UNIQUE INDEX IF NOT EXISTS news_team_url_idx ON news(team, url);

-- Cached LLM insights per match+model
CREATE TABLE IF NOT EXISTS llm_insights (
  match_id BIGINT,
  model_version VARCHAR,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  insight_json VARCHAR,
  PRIMARY KEY(match_id, model_version)
);

-- Cached expert council breakdowns (pre-computed during prediction)
CREATE TABLE IF NOT EXISTS expert_cache (
  match_id BIGINT PRIMARY KEY,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  breakdown_json VARCHAR
);

-- Head-to-head statistics (any venue)
CREATE TABLE IF NOT EXISTS h2h_stats (
  team_a VARCHAR NOT NULL,
  team_b VARCHAR NOT NULL,
  team_a_canonical VARCHAR,
  team_b_canonical VARCHAR,
  total_matches INT DEFAULT 0,
  team_a_wins INT DEFAULT 0,
  team_b_wins INT DEFAULT 0,
  draws INT DEFAULT 0,
  team_a_goals_for INT DEFAULT 0,
  team_a_goals_against INT DEFAULT 0,
  team_b_goals_for INT DEFAULT 0,
  team_b_goals_against INT DEFAULT 0,
  team_a_avg_goals_for DOUBLE DEFAULT 0.0,
  team_a_avg_goals_against DOUBLE DEFAULT 0.0,
  team_b_avg_goals_for DOUBLE DEFAULT 0.0,
  team_b_avg_goals_against DOUBLE DEFAULT 0.0,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(team_a, team_b)
);

-- Venue-specific H2H (home team hosts away team)
CREATE TABLE IF NOT EXISTS h2h_venue_stats (
  home_team VARCHAR NOT NULL,
  away_team VARCHAR NOT NULL,
  home_team_canonical VARCHAR,
  away_team_canonical VARCHAR,
  total_matches INT DEFAULT 0,
  home_wins INT DEFAULT 0,
  away_wins INT DEFAULT 0,
  draws INT DEFAULT 0,
  home_goals_for INT DEFAULT 0,
  home_goals_against INT DEFAULT 0,
  away_goals_for INT DEFAULT 0,
  away_goals_against INT DEFAULT 0,
  home_avg_goals_for DOUBLE DEFAULT 0.0,
  home_avg_goals_against DOUBLE DEFAULT 0.0,
  away_avg_goals_for DOUBLE DEFAULT 0.0,
  away_avg_goals_against DOUBLE DEFAULT 0.0,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(home_team, away_team)
);

-- Expected Goals (xG) cache
CREATE TABLE IF NOT EXISTS match_xg (
  match_id BIGINT PRIMARY KEY,
  home_xg DOUBLE,
  away_xg DOUBLE,
  method VARCHAR,
  confidence DOUBLE DEFAULT 0.5,
  computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Team mapping and normalization
CREATE TABLE IF NOT EXISTS team_mappings (
  canonical_id VARCHAR PRIMARY KEY,
  canonical_name VARCHAR NOT NULL,
  provider_names VARCHAR NOT NULL,
  confidence_score DOUBLE DEFAULT 1.0,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  notes VARCHAR
);

CREATE TABLE IF NOT EXISTS team_name_lookups (
  raw_name VARCHAR PRIMARY KEY,
  canonical_id VARCHAR NOT NULL,
  confidence DOUBLE,
  lookup_count INT DEFAULT 1,
  last_looked_up TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (canonical_id) REFERENCES team_mappings(canonical_id)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);
CREATE INDEX IF NOT EXISTS idx_matches_utc ON matches(utc_date);
CREATE INDEX IF NOT EXISTS idx_matches_comp ON matches(competition);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_pscores_model ON prediction_scores(model_version);
"""

# Columns that may need adding to older match_extras tables
_EXTRAS_MIGRATIONS = [
    ("b365ch", "DOUBLE"), ("b365cd", "DOUBLE"), ("b365ca", "DOUBLE"),
    ("psh", "DOUBLE"), ("psd", "DOUBLE"), ("psa", "DOUBLE"),
    ("avgh", "DOUBLE"), ("avgd", "DOUBLE"), ("avga", "DOUBLE"),
    ("maxh", "DOUBLE"), ("maxd", "DOUBLE"), ("maxa", "DOUBLE"),
    ("b365_o25", "DOUBLE"), ("b365_u25", "DOUBLE"),
    ("avg_o25", "DOUBLE"), ("avg_u25", "DOUBLE"),
    ("max_o25", "DOUBLE"), ("max_u25", "DOUBLE"),
    ("b365ahh", "DOUBLE"), ("b365ahha", "DOUBLE"), ("b365ahaw", "DOUBLE"),
    ("hthg", "INT"), ("htag", "INT"),
    # prediction_scores expansions
    ("goals_mae", "DOUBLE"), ("btts_correct", "BOOLEAN"),
    ("ou25_correct", "BOOLEAN"), ("score_correct", "BOOLEAN"),
    ("eg_home", "DOUBLE"), ("eg_away", "DOUBLE"),
    ("p_btts", "DOUBLE"), ("p_o25", "DOUBLE"),
    ("predicted_score_h", "INT"), ("predicted_score_a", "INT"),
]

def _migrate_columns(con):
    """Add new columns to existing tables if they don't exist yet."""
    try:
        existing = {r[0].lower() for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='match_extras'"
        ).fetchall()}
        for col, typ in _EXTRAS_MIGRATIONS[:23]:  # match_extras columns (up to htag)
            if col not in existing:
                try:
                    con.execute(f"ALTER TABLE match_extras ADD COLUMN {col} {typ}")
                except Exception:
                    pass
    except Exception:
        pass

    try:
        existing = {r[0].lower() for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='prediction_scores'"
        ).fetchall()}
        for col, typ in _EXTRAS_MIGRATIONS[21:]:  # prediction_scores columns
            if col not in existing:
                try:
                    con.execute(f"ALTER TABLE prediction_scores ADD COLUMN {col} {typ}")
                except Exception:
                    pass
    except Exception:
        pass


def connect() -> duckdb.DuckDBPyConnection:
    s = settings()
    con = duckdb.connect(s.db_path)
    con.execute(SCHEMA_SQL)
    _migrate_columns(con)
    return con
