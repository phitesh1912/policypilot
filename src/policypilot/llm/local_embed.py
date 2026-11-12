"""Local CPU embeddings via fastembed (ONNX). No API key, no per-token cost, ~130 MB model
download on first use. Used with chat-only providers such as Groq."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dims, strong for short English passages


class FastEmbedEmbedder:
    name = "fastembed"

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding

        self.embed_model = model
        self._model = TextEmbedding(
            model_name=model, cache_dir=str(cache_dir) if cache_dir else None
        )
        self.embed_dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for v in self._model.embed(texts, batch_size=64):
            a = np.asarray(v, dtype=np.float32)
            n = np.linalg.norm(a)
            out.append((a / n).tolist() if n else a.tolist())
        return out
