from __future__ import annotations
import logging
import httpx
from footy.config import settings

log = logging.getLogger("footy.llm.ollama")

def chat(messages: list[dict], model: str | None = None) -> str:
    """Call Ollama chat API with graceful error handling"""
    s = settings()
    payload = {"model": model or s.ollama_model, "messages": messages, "stream": False}
    
    try:
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{s.ollama_host}/api/chat", json=payload)
            r.raise_for_status()
            
            # Handle empty response
            if not r.text or not r.text.strip():
                return ""
            
            data = r.json()
            response_text = (data.get("message") or {}).get("content", "").strip()
            return response_text
    
    except httpx.TimeoutException:
        log.debug("Ollama timeout - service unavailable")
        return ""
    except httpx.ConnectError:
        log.debug("Ollama connection failed - service unavailable")
        return ""
    except (httpx.HTTPError, httpx.RequestError) as e:
        log.debug(f"Ollama HTTP error: {e}")
        return ""
    except Exception as e:
        log.debug(f"Ollama error: {e}")
        return ""
