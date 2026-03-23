from __future__ import annotations
import logging
import threading
import duckdb
from footy.config import settings

log = logging.getLogger(__name__)

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

-- Glicko-2 rating system state: rating, rating deviation (uncertainty), volatility
CREATE TABLE IF NOT EXISTS glicko2_state (
  team VARCHAR PRIMARY KEY,
  rating DOUBLE NOT NULL DEFAULT 1500.0,
  rd DOUBLE NOT NULL DEFAULT 350.0,        -- rating deviation (φ): uncertainty in rating estimate
  volatility DOUBLE NOT NULL DEFAULT 0.06  -- σ: magnitude of expected rating swings
);

-- Pi-rating system state: separate home/away ratings per team
CREATE TABLE IF NOT EXISTS pi_rating_state (
  team VARCHAR PRIMARY KEY,
  home_rating DOUBLE NOT NULL DEFAULT 1500.0,  -- team's strength when playing at home
  away_rating DOUBLE NOT NULL DEFAULT 1500.0,  -- team's strength when playing away
  matches_played INT NOT NULL DEFAULT 0        -- total matches for adaptive learning rate
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
  raw_json VARCHAR,
  -- The Odds API: Asian Handicap
  odds_ah_line DOUBLE,
  odds_ah_home DOUBLE,
  odds_ah_away DOUBLE,
  -- The Odds API: Both Teams To Score
  odds_btts_yes DOUBLE,
  odds_btts_no DOUBLE,
  -- football-data.org unfolded lineups/formations
  formation_home VARCHAR,
  formation_away VARCHAR,
  lineup_home VARCHAR,
  lineup_away VARCHAR,
  -- API-Football enrichment
  af_xg_home DOUBLE,
  af_xg_away DOUBLE,
  af_possession_home DOUBLE,
  af_possession_away DOUBLE,
  af_stats_json VARCHAR
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

-- Learned ensemble weights from scored historical predictions
CREATE TABLE IF NOT EXISTS ensemble_weights (
  model_name VARCHAR PRIMARY KEY,
  weight DOUBLE NOT NULL,
  n_predictions INT,
  avg_log_loss DOUBLE,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Per-expert accuracy tracking for self-learning and adaptive weighting
CREATE TABLE IF NOT EXISTS expert_performance (
  expert_name VARCHAR PRIMARY KEY,
  accuracy DOUBLE,
  log_loss DOUBLE,
  n_predictions INT,
  avg_confidence DOUBLE,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS expert_performance_by_comp (
  expert_name VARCHAR,
  competition VARCHAR,
  accuracy DOUBLE,
  n_predictions INT,
  updated_at TIMESTAMP,
  PRIMARY KEY(expert_name, competition)
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

-- FPL injury/availability snapshots (Premier League only)
CREATE TABLE IF NOT EXISTS fpl_availability (
  team VARCHAR PRIMARY KEY,
  total_players INT,
  available INT,
  doubtful INT,
  injured INT,
  suspended INT,
  injury_score DOUBLE,
  squad_strength DOUBLE,
  key_absences_json VARCHAR,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FPL fixture difficulty ratings
CREATE TABLE IF NOT EXISTS fpl_fixture_difficulty (
  team VARCHAR PRIMARY KEY,
  fdr_next_3 DOUBLE,
  fdr_next_6 DOUBLE,
  upcoming_json VARCHAR,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

CREATE TABLE IF NOT EXISTS af_context (
  match_id BIGINT PRIMARY KEY,
  fixture_id BIGINT,
  fetched_at TIMESTAMP,
  fixture_json VARCHAR,
  injuries_json VARCHAR,
  home_injuries INTEGER,
  away_injuries INTEGER,
  llm_model VARCHAR,
  llm_summary VARCHAR
);

CREATE TABLE IF NOT EXISTS weather_data (
  match_id BIGINT PRIMARY KEY,
  venue_name VARCHAR,
  latitude DOUBLE,
  longitude DOUBLE,
  kickoff_temperature_c DOUBLE,
  kickoff_apparent_temperature_c DOUBLE,
  kickoff_precipitation_mm DOUBLE,
  kickoff_wind_speed_kmh DOUBLE,
  kickoff_wind_gusts_kmh DOUBLE,
  kickoff_humidity_pct DOUBLE,
  kickoff_cloud_cover_pct DOUBLE,
  kickoff_weather_code INT,
  rainfall_prev_24h_mm DOUBLE,
  rainfall_prev_48h_mm DOUBLE,
  source VARCHAR,
  raw_json VARCHAR,
  fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS referee_assignments (
  match_id BIGINT PRIMARY KEY,
  referee_name VARCHAR,
  competition VARCHAR,
  historical_matches INT DEFAULT 0,
  yellow_cards_per_match DOUBLE,
  red_cards_per_match DOUBLE,
  penalties_per_match DOUBLE,
  home_bias_ratio DOUBLE,
  source VARCHAR,
  raw_json VARCHAR,
  fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_values (
  team VARCHAR PRIMARY KEY,
  competition VARCHAR,
  squad_market_value_eur DOUBLE,
  average_player_value_eur DOUBLE,
  median_player_value_eur DOUBLE,
  squad_size INT,
  average_age DOUBLE,
  foreign_players INT,
  national_team_players INT,
  source VARCHAR,
  raw_json VARCHAR,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS squad_depth (
  team VARCHAR PRIMARY KEY,
  competition VARCHAR,
  players_over_500_min INT,
  players_over_1000_min INT,
  top_3_value_share DOUBLE,
  top_5_value_share DOUBLE,
  source VARCHAR,
  raw_json VARCHAR,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transfermarkt_injuries (
  team VARCHAR,
  player_name VARCHAR,
  injury_type VARCHAR,
  return_date DATE,
  days_absent INT,
  games_missed INT,
  market_value_eur DOUBLE,
  source VARCHAR,
  raw_json VARCHAR,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(team, player_name)
);

CREATE TABLE IF NOT EXISTS odds_history (
  match_id BIGINT,
  bookmaker VARCHAR,
  market VARCHAR,
  outcome VARCHAR,
  price DOUBLE,
  line DOUBLE,
  is_closing BOOLEAN DEFAULT FALSE,
  is_opening BOOLEAN DEFAULT FALSE,
  captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  source VARCHAR,
  raw_json VARCHAR,
  PRIMARY KEY(match_id, bookmaker, market, outcome, captured_at)
);

CREATE TABLE IF NOT EXISTS league_config (
  competition VARCHAR PRIMARY KEY,
  country VARCHAR,
  season_span VARCHAR,
  sofa_tournament_id BIGINT,
  understat_league VARCHAR,
  fbref_slug VARCHAR,
  division_code VARCHAR,
  enabled BOOLEAN DEFAULT TRUE,
  notes VARCHAR,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS provider_status (
  match_id BIGINT,
  provider VARCHAR,
  status VARCHAR,
  detail VARCHAR,
  fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(match_id, provider)
);

CREATE TABLE IF NOT EXISTS stadiums (
  team VARCHAR PRIMARY KEY,
  venue_name VARCHAR,
  city VARCHAR,
  country VARCHAR,
  latitude DOUBLE,
  longitude DOUBLE,
  altitude_m DOUBLE,
  capacity INT,
  surface VARCHAR,
  source VARCHAR,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TheSportsDB response cache (avoids re-downloading team data every run)
CREATE TABLE IF NOT EXISTS thesportsdb_cache (
  team_name VARCHAR PRIMARY KEY,
  response_json VARCHAR NOT NULL,
  fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);
CREATE INDEX IF NOT EXISTS idx_matches_utc ON matches(utc_date);
CREATE INDEX IF NOT EXISTS idx_matches_comp ON matches(competition);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_pscores_model ON prediction_scores(model_version);
CREATE INDEX IF NOT EXISTS idx_ensemble_weights_updated ON ensemble_weights(updated_at);
CREATE INDEX IF NOT EXISTS idx_expert_performance_accuracy ON expert_performance(accuracy DESC);
CREATE INDEX IF NOT EXISTS idx_expert_performance_comp ON expert_performance_by_comp(competition, accuracy DESC);
CREATE INDEX IF NOT EXISTS idx_news_seendate ON news(seendate);
CREATE INDEX IF NOT EXISTS idx_h2h_canonical ON h2h_stats(team_a_canonical, team_b_canonical);
CREATE INDEX IF NOT EXISTS idx_h2h_venue_canonical ON h2h_venue_stats(home_team_canonical, away_team_canonical);
CREATE INDEX IF NOT EXISTS idx_match_extras_comp ON match_extras(competition, season_code);
CREATE INDEX IF NOT EXISTS idx_fpl_avail_team ON fpl_availability(team);
CREATE INDEX IF NOT EXISTS idx_fpl_fdr_team ON fpl_fixture_difficulty(team);
CREATE INDEX IF NOT EXISTS idx_weather_match ON weather_data(match_id);
CREATE INDEX IF NOT EXISTS idx_referee_match ON referee_assignments(match_id);
CREATE INDEX IF NOT EXISTS idx_market_values_comp ON market_values(competition, team);
CREATE INDEX IF NOT EXISTS idx_transfermarkt_team ON transfermarkt_injuries(team);
CREATE INDEX IF NOT EXISTS idx_odds_history_match ON odds_history(match_id, market, bookmaker);
CREATE INDEX IF NOT EXISTS idx_provider_status_provider ON provider_status(provider, status);
CREATE INDEX IF NOT EXISTS idx_stadiums_country ON stadiums(country, team);
CREATE INDEX IF NOT EXISTS idx_thesportsdb_cache_fetched ON thesportsdb_cache(fetched_at);

-- Calibration parameters for probability calibration persistence
CREATE TABLE IF NOT EXISTS calibration_params (
  model_version VARCHAR PRIMARY KEY,
  method VARCHAR NOT NULL DEFAULT 'auto',  -- 'auto', 'isotonic', 'platt', 'temperature'
  scaling_factor DOUBLE,                   -- temperature scaling or sigmoid slope
  shift_factor DOUBLE,                     -- sigmoid intercept or Platt b
  n_samples INT,                           -- number of samples used to fit
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FBref advanced team statistics (xG, shooting, passing, defense)
CREATE TABLE IF NOT EXISTS fbref_team_stats (
  team VARCHAR,
  competition VARCHAR,
  season VARCHAR,
  stat_type VARCHAR,
  games INT,
  goals INT, assists INT,
  xg DOUBLE, npxg DOUBLE, xg_assist DOUBLE,
  xg_per90 DOUBLE, npxg_per90 DOUBLE, goals_per90 DOUBLE,
  shots INT, shots_on_target INT, shots_on_target_pct DOUBLE,
  goals_per_shot DOUBLE, goals_per_shot_on_target DOUBLE,
  average_shot_distance DOUBLE,
  passes_completed INT, passes INT, passes_pct DOUBLE,
  progressive_passes INT, progressive_distance DOUBLE,
  tackles INT, tackles_won INT, tackles_won_pct DOUBLE,
  interceptions INT, blocks INT, clearances INT, errors INT,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(team, competition, season, stat_type)
);

-- Understat per-match stats (enhanced with PPDA and xPts)
CREATE TABLE IF NOT EXISTS understat_match_stats (
  match_id BIGINT PRIMARY KEY,
  home_team VARCHAR, away_team VARCHAR,
  home_xg DOUBLE, away_xg DOUBLE,
  home_npxg DOUBLE, away_npxg DOUBLE,
  home_ppda DOUBLE, away_ppda DOUBLE,
  home_deep DOUBLE, away_deep DOUBLE,
  home_xpts DOUBLE, away_xpts DOUBLE,
  season VARCHAR, competition VARCHAR,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- SoFIFA team ratings (FIFA game ratings)
CREATE TABLE IF NOT EXISTS sofifa_team_ratings (
  team VARCHAR, competition VARCHAR, fifa_version VARCHAR,
  overall INT, attack INT, midfield INT, defence INT,
  transfer_budget DOUBLE, squad_size INT,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(team, competition, fifa_version)
);

-- SoFIFA player ratings aggregated per team
CREATE TABLE IF NOT EXISTS sofifa_player_agg (
  team VARCHAR, competition VARCHAR, fifa_version VARCHAR,
  avg_overall DOUBLE, avg_pace DOUBLE, avg_shooting DOUBLE,
  avg_passing DOUBLE, avg_dribbling DOUBLE, avg_defending DOUBLE,
  avg_physical DOUBLE,
  top3_avg_overall DOUBLE,
  depth_75plus INT, depth_80plus INT,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(team, competition, fifa_version)
);

-- Derived venue statistics (computed from match history)
CREATE TABLE IF NOT EXISTS venue_stats (
  team VARCHAR PRIMARY KEY,
  competition VARCHAR,
  home_matches INT,
  home_win_pct DOUBLE, home_draw_pct DOUBLE, home_loss_pct DOUBLE,
  avg_home_scored DOUBLE, avg_home_conceded DOUBLE,
  home_advantage_strength DOUBLE,
  home_clean_sheet_pct DOUBLE, home_btts_pct DOUBLE,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Derived referee aggregate statistics
CREATE TABLE IF NOT EXISTS referee_stats (
  referee VARCHAR PRIMARY KEY,
  competition VARCHAR,
  total_matches INT,
  avg_home_goals DOUBLE, avg_away_goals DOUBLE,
  avg_total_goals DOUBLE,
  avg_yellow_home DOUBLE, avg_yellow_away DOUBLE,
  avg_red_home DOUBLE, avg_red_away DOUBLE,
  home_win_pct DOUBLE, draw_pct DOUBLE, away_win_pct DOUBLE,
  home_bias DOUBLE,
  card_strictness DOUBLE,
  goals_tendency DOUBLE,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature cache for expert computations
CREATE TABLE IF NOT EXISTS feature_cache (
  match_id BIGINT, expert_name VARCHAR,
  features_json VARCHAR,
  computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(match_id, expert_name)
);

-- Model experiment tracking
CREATE TABLE IF NOT EXISTS model_experiments (
  experiment_id VARCHAR PRIMARY KEY,
  model_version VARCHAR,
  config_json VARCHAR,
  train_logloss DOUBLE, test_logloss DOUBLE,
  train_accuracy DOUBLE, test_accuracy DOUBLE,
  train_brier DOUBLE, test_brier DOUBLE,
  n_features INT, n_experts INT,
  wf_mean_logloss DOUBLE, wf_mean_accuracy DOUBLE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Glicko-2 and Pi-Rating state tables
CREATE INDEX IF NOT EXISTS idx_glicko2_state_team ON glicko2_state(team);
CREATE INDEX IF NOT EXISTS idx_pi_rating_state_team ON pi_rating_state(team);
CREATE INDEX IF NOT EXISTS idx_calibration_params_updated ON calibration_params(last_updated);
CREATE INDEX IF NOT EXISTS idx_fbref_team ON fbref_team_stats(team, competition);
CREATE INDEX IF NOT EXISTS idx_understat_match ON understat_match_stats(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_sofifa_team ON sofifa_team_ratings(team, competition);
CREATE INDEX IF NOT EXISTS idx_venue_stats_team ON venue_stats(team);
CREATE INDEX IF NOT EXISTS idx_referee_stats_ref ON referee_stats(referee);
CREATE INDEX IF NOT EXISTS idx_feature_cache_match ON feature_cache(match_id);

-- v13: Manager tracking (derived from performance shifts)
CREATE TABLE IF NOT EXISTS manager_changes (
  team VARCHAR NOT NULL,
  competition VARCHAR,
  detected_date TIMESTAMP,
  performance_shift DOUBLE,
  direction VARCHAR,  -- 'improvement' or 'decline'
  PRIMARY KEY(team, detected_date)
);

-- v13: Betting line movement tracking
CREATE TABLE IF NOT EXISTS betting_movements (
  match_id BIGINT,
  bookmaker VARCHAR,
  market VARCHAR DEFAULT '1x2',
  open_home DOUBLE, open_draw DOUBLE, open_away DOUBLE,
  close_home DOUBLE, close_draw DOUBLE, close_away DOUBLE,
  movement_home DOUBLE, movement_draw DOUBLE, movement_away DOUBLE,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(match_id, bookmaker, market)
);

-- v13: News sentiment per team
CREATE TABLE IF NOT EXISTS news_sentiment (
  team VARCHAR NOT NULL,
  date DATE NOT NULL,
  tone_score DOUBLE,
  volume INT,
  top_themes VARCHAR,
  source VARCHAR DEFAULT 'gdelt',
  PRIMARY KEY(team, date, source)
);

-- v13: Backtest results history
CREATE TABLE IF NOT EXISTS backtest_results (
  run_id VARCHAR PRIMARY KEY,
  run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  model_version VARCHAR,
  total_matches INT,
  rps DOUBLE,
  log_loss DOUBLE,
  accuracy DOUBLE,
  brier DOUBLE,
  ece DOUBLE,
  upset_recall DOUBLE,
  vs_market_rps DOUBLE,
  config_json VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_betting_movements_match ON betting_movements(match_id);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_team ON news_sentiment(team, date);
"""

# Columns that may need adding to older match_extras tables
_MATCH_EXTRAS_MIGRATIONS = [
    ("b365ch", "DOUBLE"), ("b365cd", "DOUBLE"), ("b365ca", "DOUBLE"),
    ("psh", "DOUBLE"), ("psd", "DOUBLE"), ("psa", "DOUBLE"),
    ("avgh", "DOUBLE"), ("avgd", "DOUBLE"), ("avga", "DOUBLE"),
    ("maxh", "DOUBLE"), ("maxd", "DOUBLE"), ("maxa", "DOUBLE"),
    ("b365_o25", "DOUBLE"), ("b365_u25", "DOUBLE"),
    ("avg_o25", "DOUBLE"), ("avg_u25", "DOUBLE"),
    ("max_o25", "DOUBLE"), ("max_u25", "DOUBLE"),
    ("b365ahh", "DOUBLE"), ("b365ahha", "DOUBLE"), ("b365ahaw", "DOUBLE"),
    ("hthg", "INT"), ("htag", "INT"),
    # The Odds API: Asian Handicap + BTTS
    ("odds_ah_line", "DOUBLE"), ("odds_ah_home", "DOUBLE"), ("odds_ah_away", "DOUBLE"),
    ("odds_btts_yes", "DOUBLE"), ("odds_btts_no", "DOUBLE"),
    # football-data.org unfolded
    ("formation_home", "VARCHAR"), ("formation_away", "VARCHAR"),
    ("lineup_home", "VARCHAR"), ("lineup_away", "VARCHAR"),
    # API-Football stats
    ("af_xg_home", "DOUBLE"), ("af_xg_away", "DOUBLE"),
    ("af_possession_home", "DOUBLE"), ("af_possession_away", "DOUBLE"),
    ("af_stats_json", "VARCHAR"),
    # scraper stack enrichments
    ("understat_xg_home", "DOUBLE"), ("understat_xg_away", "DOUBLE"),
    ("understat_shots_home", "INT"), ("understat_shots_away", "INT"),
    ("sofascore_xg_home", "DOUBLE"), ("sofascore_xg_away", "DOUBLE"),
    ("sofascore_possession_home", "DOUBLE"), ("sofascore_possession_away", "DOUBLE"),
    ("sofascore_rating_home", "DOUBLE"), ("sofascore_rating_away", "DOUBLE"),
    ("expected_lineup_home", "VARCHAR"), ("expected_lineup_away", "VARCHAR"),
    ("venue_name", "VARCHAR"), ("venue_city", "VARCHAR"),
    ("referee_name", "VARCHAR"),
]

# Columns that may need adding to older prediction_scores tables
_PRED_SCORES_MIGRATIONS = [
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
        for col, typ in _MATCH_EXTRAS_MIGRATIONS:
            if col not in existing:
                try:
                    con.execute(f"ALTER TABLE match_extras ADD COLUMN {col} {typ}")
                except Exception as e:
                    log.debug("migrate match_extras.%s: %s", col, e)
    except Exception as e:
        log.debug("migrate match_extras lookup: %s", e)

    try:
        existing = {r[0].lower() for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='prediction_scores'"
        ).fetchall()}
        for col, typ in _PRED_SCORES_MIGRATIONS:
            if col not in existing:
                try:
                    con.execute(f"ALTER TABLE prediction_scores ADD COLUMN {col} {typ}")
                except Exception as e:
                    log.debug("migrate prediction_scores.%s: %s", col, e)
    except Exception as e:
        log.debug("migrate prediction_scores lookup: %s", e)


_connection: duckdb.DuckDBPyConnection | None = None
_schema_initialized: bool = False
_lock = threading.Lock()
_local = threading.local()


def connect() -> duckdb.DuckDBPyConnection:
    """Get a thread-safe DuckDB connection.

    Uses a single primary connection with thread-local cursors.
    DuckDB supports concurrent reads but serializes writes,
    which is fine for our read-heavy workload.
    """
    global _connection, _schema_initialized
    with _lock:
        s = settings()
        if _connection is None:
            _connection = duckdb.connect(s.db_path)
        if not _schema_initialized:
            _connection.execute(SCHEMA_SQL)
            _migrate_columns(_connection)
            _schema_initialized = True
    # Return thread-local cursor for concurrent access
    if not hasattr(_local, 'cursor') or _local.cursor is None:
        _local.cursor = _connection.cursor()
    return _local.cursor


def connect_primary() -> duckdb.DuckDBPyConnection:
    """Get the primary connection (for schema operations and single-threaded use)."""
    global _connection, _schema_initialized
    with _lock:
        s = settings()
        if _connection is None:
            _connection = duckdb.connect(s.db_path)
        if not _schema_initialized:
            _connection.execute(SCHEMA_SQL)
            _migrate_columns(_connection)
            _schema_initialized = True
    return _connection
