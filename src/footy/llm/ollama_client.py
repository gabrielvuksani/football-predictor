from __future__ import annotations
import logging
import time
import httpx
from footy.config import settings

log = logging.getLogger("footy.llm.ollama")

# Reusable client – avoids per-call TCP + TLS handshake
_client: httpx.Client | None = None

def _get_client() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
        )
    return _client


def chat(
    messages: list[dict],
    model: str | None = None,
    *,
    temperature: float = 0.3,
    num_ctx: int = 2048,
    num_predict: int = 512,
    max_retries: int = 2,
) -> str:
    """Call Ollama chat API with retry, keep-alive, and tuned options.

    Parameters
    ----------
    messages : list[dict]
        Chat messages in OpenAI format.
    model : str | None
        Override model name (default from config).
    temperature : float
        Sampling temperature. Lower = more deterministic.
    num_ctx : int
        Context window size (tokens). 2048 is fast; bump if prompts are large.
    num_predict : int
        Max tokens to generate. 512 is enough for JSON answers.
    max_retries : int
        Number of retry attempts on transient failures.
    """
    s = settings()
    payload = {
        "model": model or s.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    client = _get_client()
    url = f"{s.ollama_host}/api/chat"

    for attempt in range(1, max_retries + 2):  # 1-indexed, total = max_retries + 1
        try:
            r = client.post(url, json=payload)
            r.raise_for_status()

            if not r.text or not r.text.strip():
                return ""

            data = r.json()
            return (data.get("message") or {}).get("content", "").strip()

        except httpx.TimeoutException:
            log.debug("Ollama timeout (attempt %d/%d)", attempt, max_retries + 1)
        except httpx.ConnectError:
            log.debug("Ollama connection failed (attempt %d/%d)", attempt, max_retries + 1)
        except (httpx.HTTPStatusError,) as e:
            if e.response.status_code >= 500 and attempt <= max_retries:
                log.debug("Ollama 5xx (attempt %d): %s", attempt, e)
            else:
                log.debug("Ollama HTTP error: %s", e)
                return ""
        except Exception as e:
            log.debug("Ollama error: %s", e)
            return ""

        # exponential back-off before retry
        if attempt <= max_retries:
            time.sleep(0.5 * (2 ** (attempt - 1)))

    return ""
