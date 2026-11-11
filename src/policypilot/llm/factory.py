from __future__ import annotations

from policypilot.config import Settings
from policypilot.llm.base import LLMProvider
from policypilot.llm.fake import FakeProvider


def build_provider(settings: Settings) -> LLMProvider:
    kind = settings.llm_provider.lower()
    if kind == "fake":
        return FakeProvider()
    if kind == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("LLM_PROVIDER=gemini requires GEMINI_API_KEY")
        from policypilot.llm.gemini import GeminiProvider

        return GeminiProvider(
            api_key=settings.gemini_api_key,
            chat_model=settings.llm_model,
            embed_model=settings.embed_model,
        )
    if kind == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        from policypilot.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.openai_api_key,
            chat_model=settings.llm_model,
            embed_model=settings.embed_model,
        )
    raise ValueError(f"unknown LLM_PROVIDER {settings.llm_provider!r}")
