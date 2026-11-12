"""Combine a chat-only provider with a separate embedder behind the single LLMProvider interface."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from policypilot.llm.base import Completion, LLMProvider


class Embedder(Protocol):
    name: str
    embed_model: str
    embed_dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class CompositeProvider(LLMProvider):
    """Chat from one provider (e.g. Groq, which has no embeddings API), vectors from another."""

    def __init__(self, chat: LLMProvider, embedder: Embedder, name: str | None = None) -> None:
        self._chat = chat
        self._embedder = embedder
        self.name = name or f"{chat.name}+{embedder.name}"
        self.chat_model = chat.chat_model
        self.embed_model = embedder.embed_model
        self.embed_dim = embedder.embed_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> Completion:
        return self._chat.complete(system, user, temperature=temperature)

    def stream(self, system: str, user: str, *, temperature: float = 0.0) -> Iterator[str]:
        return self._chat.stream(system, user, temperature=temperature)
