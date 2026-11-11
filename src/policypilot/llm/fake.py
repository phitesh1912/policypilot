"""Deterministic provider for tests, CI and the offline demo. No network, no randomness.

Embeddings: each token is mapped to a fixed pseudo-random unit vector seeded by a stable hash of
the token; a text's embedding is the normalised sum (with stemmed BM25 tokens, so lexical overlap
≈ cosine similarity). Crude, but deterministic and good enough to exercise vector search, RRF and
the semantic cache end-to-end.

Completions: parses the context blocks out of the prompt, picks the block with the most token
overlap with the question, and returns its first sentences with the block's citation. With no
context, or no overlap at all, it returns the refusal text — so the refusal path is testable
offline too.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator

import numpy as np

from policypilot.llm.base import (
    CONTEXT_HEADER,
    QUESTION_HEADER,
    REFUSAL_TEXT,
    Completion,
    LLMProvider,
)
from policypilot.retrieval.bm25 import tokenize

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _token_vector(token: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha1(token.encode("utf-8")).digest()[:4], "big")
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


class FakeProvider(LLMProvider):
    name = "fake"
    chat_model = "fake-chat"
    embed_model = "fake-embed"
    embed_dim = 64

    def __init__(self, dim: int = 64) -> None:
        self.embed_dim = dim
        self._cache: dict[str, np.ndarray] = {}
        self.calls: list[tuple[str, str]] = []  # (kind, payload) for assertions in tests

    def _tok_vec(self, token: str) -> np.ndarray:
        v = self._cache.get(token)
        if v is None:
            v = self._cache[token] = _token_vector(token, self.embed_dim)
        return v

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            self.calls.append(("embed", text))
            toks = tokenize(text)
            if not toks:
                out.append([0.0] * self.embed_dim)
                continue
            v = np.sum([self._tok_vec(t) for t in toks], axis=0)
            n = np.linalg.norm(v)
            out.append((v / n).tolist() if n else v.tolist())
        return out

    # --- completion --------------------------------------------------------------------------

    @staticmethod
    def _parse(user: str) -> tuple[list[tuple[str, str]], str]:
        """Return ([(citation, body), ...], question)."""
        question = ""
        if QUESTION_HEADER in user:
            ctx_part, question = user.split(QUESTION_HEADER, 1)
        else:
            ctx_part = user
        blocks: list[tuple[str, str]] = []
        for raw in ctx_part.split(CONTEXT_HEADER)[1:]:
            head, _, body = raw.partition("\n")
            blocks.append((head.strip(), body.strip()))
        return blocks, question.strip()

    def _answer(self, user: str) -> str:
        blocks, question = self._parse(user)
        if not blocks:
            return REFUSAL_TEXT
        q = set(tokenize(question))
        best, best_overlap = None, 0
        for cite, body in blocks:
            overlap = len(q & set(tokenize(body)))
            if overlap > best_overlap:
                best, best_overlap = (cite, body), overlap
        if best is None:
            return REFUSAL_TEXT
        cite, body = best
        # Drop the "Title > Section" prefix line if present, then take the first two sentences.
        text = body.split("\n\n", 1)[1] if "\n\n" in body else body
        sentences = _SENTENCE_RE.split(text.replace("\n", " ").strip())
        return " ".join(sentences[:2]).strip() + f" {cite}"

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> Completion:
        self.calls.append(("complete", user))
        text = self._answer(user)
        return Completion(
            text=text,
            prompt_tokens=len(tokenize(system)) + len(tokenize(user)),
            completion_tokens=len(tokenize(text)),
        )

    def stream(self, system: str, user: str, *, temperature: float = 0.0) -> Iterator[str]:
        text = self.complete(system, user, temperature=temperature).text
        for word in text.split(" "):
            yield word + " "
