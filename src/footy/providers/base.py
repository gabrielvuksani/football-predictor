from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import httpx

from footy.config import settings

log = logging.getLogger(__name__)

# Maximum cache entries per provider instance to prevent unbounded memory growth
_DEFAULT_MAX_CACHE_SIZE = 2048


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce usable data."""


@dataclass(slots=True)
class CacheEntry:
    value: Any
    expires_at: float


class BaseProvider:
    """Shared provider plumbing for zero-cost scraper/API integrations.

    Responsibilities:
    - central timeout handling
    - lightweight in-process TTL+LRU cache (bounded size)
    - normalized user-agent headers
    - JSON/text download helpers
    """

    name = "base"

    def __init__(
        self,
        *,
        enabled: bool = True,
        headers: dict[str, str] | None = None,
        max_cache_size: int = _DEFAULT_MAX_CACHE_SIZE,
    ):
        s = settings()
        self.enabled = enabled
        self._ttl = s.scraper_cache_ttl_seconds
        self._timeout = s.request_timeout_seconds
        self._headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FootyPredictor/12.0; +https://github.com/)"
        }
        if headers:
            self._headers.update(headers)
        # OrderedDict for LRU eviction — most recently used at end
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_cache_size = max_cache_size
        self._client: httpx.Client | None = None

    def client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                follow_redirects=True,
                timeout=httpx.Timeout(self._timeout, connect=min(10, self._timeout)),
                headers=self._headers,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            self._client.close()

    def _cache_key(self, url: str, params: dict[str, Any] | None = None) -> str:
        if not params:
            return url
        return f"{url}::{json.dumps(params, sort_keys=True, default=str)}"

    def _get_cached(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if not entry:
            return None
        if entry.expires_at <= time.time():
            self._cache.pop(key, None)
            return None
        # Move to end (most recently used) for LRU
        self._cache.move_to_end(key)
        return entry.value

    def _set_cached(self, key: str, value: Any, ttl: int | None = None) -> Any:
        # If key already exists, remove it first (will be re-added at end)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = CacheEntry(value=value, expires_at=time.time() + float(ttl or self._ttl))
        # Evict oldest entries if over size limit
        while len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)  # Remove oldest (least recently used)
        return value

    def fetch_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        ttl: int | None = None,
        allow_empty: bool = False,
    ) -> str:
        if not self.enabled:
            raise ProviderError(f"Provider {self.name} is disabled")
        key = self._cache_key(url, params)
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        resp = self.client().get(url, params=params)
        resp.raise_for_status()
        text = resp.text
        if not allow_empty and not text.strip():
            raise ProviderError(f"Provider {self.name} returned empty response for {url}")
        return self._set_cached(key, text, ttl=ttl)

    def fetch_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> Any:
        if not self.enabled:
            raise ProviderError(f"Provider {self.name} is disabled")
        key = self._cache_key(url, params)
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        resp = self.client().get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return self._set_cached(key, data, ttl=ttl)

    def fetch_bytes(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> bytes:
        if not self.enabled:
            raise ProviderError(f"Provider {self.name} is disabled")
        key = self._cache_key(url, params)
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        resp = self.client().get(url, params=params)
        resp.raise_for_status()
        return self._set_cached(key, resp.content, ttl=ttl)
