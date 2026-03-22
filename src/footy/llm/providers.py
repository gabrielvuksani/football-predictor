from __future__ import annotations

from footy.config import settings
from footy.llm import ollama_client
from footy.llm import groq_client


class LLMProviderRouter:
    def __init__(self):
        self.settings = settings()

    def chat(self, messages: list[dict], **kwargs) -> str:
        for provider in self.settings.llm_provider_order:
            provider = provider.lower()
            if provider == "groq":
                text = groq_client.chat(messages, **kwargs)
                if text:
                    return text
            elif provider == "ollama":
                text = ollama_client.chat(messages, **kwargs)
                if text:
                    return text
        return ""


def chat(messages: list[dict], **kwargs) -> str:
    return LLMProviderRouter().chat(messages, **kwargs)
