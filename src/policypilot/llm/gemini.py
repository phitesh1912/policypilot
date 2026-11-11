"""Gemini provider via the official ``google-genai`` SDK (free tier works)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from policypilot.llm.base import Completion, LLMProvider


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        chat_model: str = "gemini-2.5-flash",
        embed_model: str = "gemini-embedding-001",
        embed_dim: int = 768,
    ) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.embed_dim = embed_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        # The embeddings endpoint accepts batches; keep them modest to stay inside request limits.
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            resp = self._client.models.embed_content(
                model=self.embed_model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.embed_dim,
                ),
            )
            for e in resp.embeddings:
                v = np.asarray(e.values, dtype=np.float32)
                n = np.linalg.norm(v)
                out.append((v / n).tolist() if n else v.tolist())
        return out

    def _config(self, system: str, temperature: float):
        from google.genai import types

        return types.GenerateContentConfig(system_instruction=system, temperature=temperature)

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> Completion:
        resp = self._client.models.generate_content(
            model=self.chat_model, contents=user, config=self._config(system, temperature)
        )
        usage = resp.usage_metadata
        return Completion(
            text=resp.text or "",
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )

    def stream(self, system: str, user: str, *, temperature: float = 0.0) -> Iterator[str]:
        for event in self._client.models.generate_content_stream(
            model=self.chat_model, contents=user, config=self._config(system, temperature)
        ):
            if event.text:
                yield event.text
