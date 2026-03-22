from __future__ import annotations

import logging
import time

import httpx

from footy.config import settings

log = logging.getLogger("footy.llm.groq")

_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
    return _client


def chat(
    messages: list[dict],
    model: str | None = None,
    *,
    temperature: float = 0.3,
    max_tokens: int = 512,
    max_retries: int = 2,
) -> str:
    s = settings()
    if not s.groq_api_key:
        return ""

    payload = {
        "model": model or s.groq_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {s.groq_api_key}",
        "Content-Type": "application/json",
    }

    client = _get_client()
    url = "https://api.groq.com/openai/v1/chat/completions"
    for attempt in range(1, max_retries + 2):
        try:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                return ""
            return ((choices[0].get("message") or {}).get("content") or "").strip()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            log.debug("Groq HTTP error %s on attempt %d", status, attempt)
            if status in {429, 500, 502, 503, 504} and attempt <= max_retries:
                time.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            return ""
        except Exception as e:
            log.debug("Groq error: %s", e)
            if attempt <= max_retries:
                time.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            return ""
    return ""
