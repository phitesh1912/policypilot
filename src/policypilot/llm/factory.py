from __future__ import annotations

from policypilot.config import Settings
from policypilot.llm.base import LLMProvider
from policypilot.llm.fake import FakeProvider

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Groq retired its Llama 3.x endpoints in 2026; gpt-oss-120b is the strongest free option.
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"


def build_provider(settings: Settings) -> LLMProvider:
    kind = settings.llm_provider.lower()
    if kind == "fake":
        return FakeProvider()

    if kind == "groq":
        # Groq: free, fast, OpenAI-compatible chat. No embeddings endpoint, so vectors come from
        # a local fastembed model — the whole stack stays free.
        if not settings.groq_api_key:
            raise RuntimeError("LLM_PROVIDER=groq requires GROQ_API_KEY")
        from policypilot.llm.composite import CompositeProvider
        from policypilot.llm.local_embed import DEFAULT_MODEL, FastEmbedEmbedder
        from policypilot.llm.openai_provider import OpenAIProvider

        chat = OpenAIProvider(
            api_key=settings.groq_api_key,
            chat_model=settings.llm_model or GROQ_DEFAULT_MODEL,
            embed_model="n/a",
            embed_dim=0,
            base_url=GROQ_BASE_URL,
        )
        chat.name = "groq"
        embedder = FastEmbedEmbedder(
            settings.embed_model or DEFAULT_MODEL, settings.fastembed_cache_dir
        )
        return CompositeProvider(chat, embedder)

    if kind == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("LLM_PROVIDER=gemini requires GEMINI_API_KEY")
        from policypilot.llm.gemini import GeminiProvider

        return GeminiProvider(
            api_key=settings.gemini_api_key,
            chat_model=settings.llm_model or "gemini-2.5-flash",
            embed_model=settings.embed_model or "gemini-embedding-001",
        )

    if kind == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        from policypilot.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.openai_api_key,
            chat_model=settings.llm_model or "gpt-4o-mini",
            embed_model=settings.embed_model or "text-embedding-3-small",
        )

    raise ValueError(f"unknown LLM_PROVIDER {settings.llm_provider!r}")
