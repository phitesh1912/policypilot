"""OpenAI provider (also works with any OpenAI-compatible endpoint via ``base_url``)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from policypilot.llm.base import Completion, LLMProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        chat_model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        embed_dim: int = 1536,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.embed_dim = embed_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            resp = self._client.embeddings.create(
                model=self.embed_model, input=texts[i : i + 128], dimensions=self.embed_dim
            )
            for d in sorted(resp.data, key=lambda d: d.index):
                v = np.asarray(d.embedding, dtype=np.float32)
                n = np.linalg.norm(v)
                out.append((v / n).tolist() if n else v.tolist())
        return out

    @staticmethod
    def _messages(system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> Completion:
        resp = self._client.chat.completions.create(
            model=self.chat_model, messages=self._messages(system, user), temperature=temperature
        )
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    def stream(self, system: str, user: str, *, temperature: float = 0.0) -> Iterator[str]:
        for event in self._client.chat.completions.create(
            model=self.chat_model,
            messages=self._messages(system, user),
            temperature=temperature,
            stream=True,
        ):
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                yield delta
