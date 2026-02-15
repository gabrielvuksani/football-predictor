"""
Smart caching layer for predictions and metadata.

This module implements a file-based cache for expensive computations:
- Match predictions (with configurable TTL)
- Team metadata (H2H stats, form, xG)
- Odds data
- Model state

Uses SQLite for fast, persistent caching without external dependencies.
Cache expires based on TTL and match status.
"""

from __future__ import annotations
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class MatchCache:
    """Fast cache for match predictions and metadata."""
    
    def __init__(self, db_path: str = "data/cache.db"):
        """
        Initialize cache.
        
        Args:
            db_path: Path to SQLite cache database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Create cache tables if they don't exist."""
        con = sqlite3.connect(str(self.db_path))
        con.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        
        con.execute("""
            CREATE TABLE IF NOT EXISTS prediction_cache (
                match_id INTEGER PRIMARY KEY,
                model_version VARCHAR NOT NULL,
                p_home DOUBLE,
                p_draw DOUBLE,
                p_away DOUBLE,
                p_over_2_5 DOUBLE,
                p_btts DOUBLE,
                confidence DOUBLE DEFAULT 0.5,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                match_status VARCHAR,  -- FINISHED, SCHEDULED, LIVE, etc.
                source VARCHAR DEFAULT 'model'  -- 'model', 'ensemble', 'historical'
            );
        """)
        
        con.execute("""
            CREATE TABLE IF NOT EXISTS metadata_cache (
                key VARCHAR PRIMARY KEY,
                category VARCHAR NOT NULL,  -- 'team', 'h2h', 'odds', 'form'
                value VARCHAR NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                refresh_count INTEGER DEFAULT 0
            );
        """)
        
        con.execute("""
            CREATE TABLE IF NOT EXISTS cache_stats (
                metric VARCHAR PRIMARY KEY,
                value INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_expires ON prediction_cache(expires_at);
        """)
        
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_meta_category ON metadata_cache(category);
        """)
        
        con.commit()
        con.close()
    
    def cache_prediction(
        self,
        match_id: int,
        model_version: str,
        p_home: float,
        p_draw: float,
        p_away: float,
        confidence: float = 0.5,
        ttl_hours: int = 24,
        match_status: str = "SCHEDULED",
    ) -> None:
        """
        Cache a match prediction.
        
        Args:
            match_id: Match ID
            model_version: Model version (e.g., 'v5_ultimate')
            p_home: Home win probability
            p_draw: Draw probability
            p_away: Away win probability
            confidence: Confidence score (0-1)
            ttl_hours: Time-to-live in hours (default 24)
            match_status: Match status (FINISHED, SCHEDULED, LIVE, etc.)
        """
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        con = sqlite3.connect(str(self.db_path))
        con.execute("""
            INSERT OR REPLACE INTO prediction_cache
            (match_id, model_version, p_home, p_draw, p_away, confidence, expires_at, match_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            match_id, model_version,
            p_home, p_draw, p_away,
            confidence, expires_at, match_status
        ])
        con.commit()
        con.close()
        
        logger.debug(f"[cache] prediction cached: match_id={match_id}, model={model_version}, ttl={ttl_hours}h")
    
    def get_prediction(
        self,
        match_id: int,
        model_version: str,
        skip_expired: bool = True,
    ) -> Optional[dict]:
        """
        Retrieve a cached prediction.
        
        Args:
            match_id: Match ID
            model_version: Model version
            skip_expired: Return None if cache expired
        
        Returns:
            dict with p_home, p_draw, p_away, confidence or None
        """
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        
        query = """
            SELECT p_home, p_draw, p_away, confidence, cached_at, expires_at
            FROM prediction_cache
            WHERE match_id = ? AND model_version = ?
        """
        
        row = con.execute(query, [match_id, model_version]).fetchone()
        con.close()
        
        if not row:
            return None
        
        # Check expiration
        if skip_expired and row["expires_at"]:
            expires = datetime.fromisoformat(row["expires_at"])
            if datetime.now() > expires:
                return None
        
        return {
            "p_home": float(row["p_home"]),
            "p_draw": float(row["p_draw"]),
            "p_away": float(row["p_away"]),
            "confidence": float(row["confidence"]),
            "cached_at": row["cached_at"],
        }
    
    def cache_metadata(
        self,
        key: str,
        category: str,
        value: Any,
        ttl_hours: int = 168,  # 1 week for metadata
    ) -> None:
        """
        Cache metadata (team info, H2H, odds, form).
        
        Args:
            key: Cache key (e.g., 'h2h:manchester_city:arsenal')
            category: Category (team, h2h, odds, form, xg)
            value: Value to cache (converted to JSON)
            ttl_hours: Time-to-live in hours
        """
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        con = sqlite3.connect(str(self.db_path))
        con.execute("""
            INSERT OR REPLACE INTO metadata_cache
            (key, category, value, expires_at)
            VALUES (?, ?, ?, ?)
        """, [
            key, category,
            json.dumps(value),
            expires_at
        ])
        con.commit()
        con.close()
        
        logger.debug(f"[cache] metadata cached: key={key}, category={category}, ttl={ttl_hours}h")
    
    def get_metadata(
        self,
        key: str,
        category: str = None,
        skip_expired: bool = True,
    ) -> Optional[Any]:
        """
        Retrieve cached metadata.
        
        Args:
            key: Cache key
            category: Optional category filter
            skip_expired: Return None if cache expired
        
        Returns:
            Cached value or None
        """
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        
        if category:
            query = """
                SELECT value, expires_at
                FROM metadata_cache
                WHERE key = ? AND category = ?
            """
            row = con.execute(query, [key, category]).fetchone()
        else:
            query = """
                SELECT value, expires_at
                FROM metadata_cache
                WHERE key = ?
            """
            row = con.execute(query, [key]).fetchone()
        
        con.close()
        
        if not row:
            return None
        
        # Check expiration
        if skip_expired and row["expires_at"]:
            expires = datetime.fromisoformat(row["expires_at"])
            if datetime.now() > expires:
                return None
        
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return None
    
    def invalidate_predictions(
        self,
        match_id: int = None,
        model_version: str = None,
        match_status: str = None,
    ) -> int:
        """
        Invalidate predictions from cache.
        
        Args:
            match_id: Optional match ID to invalidate
            model_version: Optional model version to invalidate
            match_status: Optional match status filter (e.g., 'FINISHED')
        
        Returns:
            Number of rows deleted
        """
        con = sqlite3.connect(str(self.db_path))
        
        conditions = []
        params = []
        
        if match_id is not None:
            conditions.append("match_id = ?")
            params.append(match_id)
        
        if model_version is not None:
            conditions.append("model_version = ?")
            params.append(model_version)
        
        if match_status is not None:
            conditions.append("match_status = ?")
            params.append(match_status)
        
        if not conditions:
            # Delete all expired predictions
            query = "DELETE FROM prediction_cache WHERE expires_at < datetime('now')"
            cursor = con.execute(query)
        else:
            where_clause = " AND ".join(conditions)
            query = f"DELETE FROM prediction_cache WHERE {where_clause}"
            cursor = con.execute(query, params)
        
        con.commit()
        count = cursor.rowcount
        con.close()
        
        logger.info(f"[cache] invalidated {count} predictions")
        return count
    
    def invalidate_metadata(self, category: str = None, key: str = None) -> int:
        """
        Invalidate metadata from cache.
        
        Args:
            category: Optional category to invalidate
            key: Optional specific key to invalidate
        
        Returns:
            Number of rows deleted
        """
        con = sqlite3.connect(str(self.db_path))
        
        if key:
            query = "DELETE FROM metadata_cache WHERE key = ?"
            cursor = con.execute(query, [key])
        elif category:
            query = "DELETE FROM metadata_cache WHERE category = ?"
            cursor = con.execute(query, [category])
        else:
            query = "DELETE FROM metadata_cache WHERE expires_at < datetime('now')"
            cursor = con.execute(query)
        
        con.commit()
        count = cursor.rowcount
        con.close()
        
        logger.info(f"[cache] invalidated {count} metadata entries")
        return count
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        
        stats = {
            "total_predictions": con.execute(
                "SELECT COUNT(*) as c FROM prediction_cache"
            ).fetchone()["c"],
            "expired_predictions": con.execute(
                "SELECT COUNT(*) as c FROM prediction_cache WHERE expires_at < datetime('now')"
            ).fetchone()["c"],
            "total_metadata": con.execute(
                "SELECT COUNT(*) as c FROM metadata_cache"
            ).fetchone()["c"],
            "expired_metadata": con.execute(
                "SELECT COUNT(*) as c FROM metadata_cache WHERE expires_at < datetime('now')"
            ).fetchone()["c"],
            "metadata_by_category": dict(
                con.execute(
                    "SELECT category, COUNT(*) as c FROM metadata_cache GROUP BY category"
                ).fetchall()
            ),
        }
        
        con.close()
        return stats
    
    def cleanup(self, delete_expired: bool = True) -> dict:
        """
        Clean up cache (remove expired entries).
        
        Args:
            delete_expired: Whether to delete expired entries
        
        Returns:
            dict with cleanup stats
        """
        logger.info("[cache] cleanup started")
        
        stats = {
            "deleted_predictions": 0,
            "deleted_metadata": 0,
        }
        
        if delete_expired:
            stats["deleted_predictions"] = self.invalidate_predictions()
            stats["deleted_metadata"] = self.invalidate_metadata()
        
        logger.info(f"[cache] cleanup complete: {stats}")
        return stats
    
    def clear(self) -> None:
        """Clear entire cache (use with caution)."""
        con = sqlite3.connect(str(self.db_path))
        con.execute("DELETE FROM prediction_cache")
        con.execute("DELETE FROM metadata_cache")
        con.commit()
        con.close()
        logger.warning("[cache] entire cache cleared")


# Global cache instance
_cache: Optional[MatchCache] = None


def get_cache(db_path: str = "data/cache.db") -> MatchCache:
    """Get or create global cache instance."""
    global _cache
    if _cache is None:
        _cache = MatchCache(db_path)
    return _cache


def cache_prediction_batch(
    predictions: list[dict],
    model_version: str = "v1_elo_poisson",
    ttl_hours: int = 6,  # Shorter TTL for upcoming matches
) -> int:
    """
    Cache a batch of predictions efficiently.
    
    Args:
        predictions: List of dicts with keys: match_id, p_home, p_draw, p_away, confidence (optional), match_status (optional)
        model_version: Model version string
        ttl_hours: Time-to-live in hours
    
    Returns:
        Number of predictions cached
    """
    cache = get_cache()
    con = sqlite3.connect(str(cache.db_path))
    
    expires_at = (datetime.now() + timedelta(hours=ttl_hours)).isoformat()
    
    rows = [
        (
            p.get("match_id"),
            model_version,
            p.get("p_home", 0),
            p.get("p_draw", 0),
            p.get("p_away", 0),
            p.get("confidence", 0.5),
            expires_at,
            p.get("match_status", "SCHEDULED"),
        )
        for p in predictions
    ]
    
    con.executemany("""
        INSERT OR REPLACE INTO prediction_cache
        (match_id, model_version, p_home, p_draw, p_away, confidence, expires_at, match_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    
    con.commit()
    count = len(rows)
    con.close()
    
    logger.debug(f"[cache] cached {count} predictions for {model_version}")
    return count


def lookup_cached_predictions(
    match_ids: list[int],
    model_version: str = "v1_elo_poisson",
) -> dict:
    """
    Look up multiple cached predictions.
    
    Args:
        match_ids: List of match IDs to look up
        model_version: Model version
    
    Returns:
        dict mapping match_id -> {p_home, p_draw, p_away, confidence}
        Only includes matches with valid cache hits
    """
    if not match_ids:
        return {}
    
    cache = get_cache()
    con = sqlite3.connect(str(cache.db_path))
    con.row_factory = sqlite3.Row
    
    placeholders = ",".join("?" * len(match_ids))
    query = f"""
        SELECT match_id, p_home, p_draw, p_away, confidence
        FROM prediction_cache
        WHERE match_id IN ({placeholders})
          AND model_version = ?
          AND (expires_at IS NULL OR expires_at > datetime('now'))
    """
    
    rows = con.execute(query, match_ids + [model_version]).fetchall()
    con.close()
    
    result = {
        int(row["match_id"]): {
            "p_home": float(row["p_home"]),
            "p_draw": float(row["p_draw"]),
            "p_away": float(row["p_away"]),
            "confidence": float(row["confidence"]),
        }
        for row in rows
    }
    
    return result


def cache_wrapper(ttl_hours: int = 24):
    """
    Decorator to cache function results.
    
    Usage:
        @cache_wrapper(ttl_hours=24)
        def expensive_function(arg1, arg2):
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}{str(kwargs)}"
            
            cache = get_cache()
            cached = cache.get_metadata(key, category="function_cache")
            
            if cached is not None:
                logger.debug(f"[cache] hit: {func.__name__}")
                return cached
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store result
            cache.cache_metadata(key, "function_cache", result, ttl_hours=ttl_hours)
            logger.debug(f"[cache] miss: {func.__name__}, stored for {ttl_hours}h")
            
            return result
        
        return wrapper
    return decorator
