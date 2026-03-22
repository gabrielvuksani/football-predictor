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
        try:
            txt2 = chat([{"role":"user","content":"Output ONLY valid JSON. No prose.\n\n"+prompt["content"]}])
            if not txt2 or not txt2.strip():
                return NewsSignal(availability_score=0.0, short_summary="LLM returned empty response")
            obj2 = json.loads(txt2)
            return NewsSignal.model_validate(obj2)
        except (json.JSONDecodeError, ValidationError, Exception):
            return NewsSignal(availability_score=0.0, short_summary="Failed to parse LLM output")
