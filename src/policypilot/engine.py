"""The orchestrator: ingest → retrieve → gate → pack → generate → ground, with caching."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from policypilot.answer import SYSTEM_PROMPT, build_prompt, ground
from policypilot.budget import Pricing, pack
from policypilot.cache import LRUSemanticCache
from policypilot.config import Settings
from policypilot.ingest.chunker import Chunker
from policypilot.ingest.loader import load_directory
from policypilot.llm.base import REFUSAL_TEXT, LLMProvider
from policypilot.llm.factory import build_provider
from policypilot.models import AskResult, RetrievalTrace, RetrievedChunk, TokenUsage, User
from policypilot.retrieval.base import Retriever
from policypilot.retrieval.bm25 import BM25Retriever
from policypilot.retrieval.hybrid import HybridRetriever
from policypilot.retrieval.vector import VectorRetriever, make_client


@dataclass
class IngestStats:
    documents: int
    chunks: int
    seconds: float


class Engine:
    def __init__(
        self,
        settings: Settings,
        *,
        provider: LLMProvider | None = None,
        retriever: Retriever | None = None,
        chunker: Chunker | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider or build_provider(settings)
        self.chunker = chunker or Chunker(settings.chunk_tokens, settings.chunk_overlap)
        if retriever is None:
            client = make_client(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                path=settings.qdrant_path,
            )
            retriever = HybridRetriever(
                BM25Retriever(),
                VectorRetriever(self.provider, client, settings.qdrant_collection),
                rrf_k=settings.rrf_k,
            )
        self.retriever = retriever
        self.cache: LRUSemanticCache[AskResult] = LRUSemanticCache(
            settings.cache_capacity, settings.cache_similarity_threshold
        )
        self.pricing = Pricing(
            settings.price_input_per_1m, settings.price_output_per_1m, settings.price_embed_per_1m
        )
        self.indexed_chunks = 0

    # --- ingestion ---------------------------------------------------------------------------

    def ingest(self, directory: Path | None = None) -> IngestStats:
        t0 = time.perf_counter()
        docs = load_directory(directory or self.settings.policies_dir)
        chunks = self.chunker.chunk_documents(docs)
        self.retriever.index(chunks)
        self.cache.clear()  # anything answered from the old index is now stale
        self.indexed_chunks = len(chunks)
        return IngestStats(len(docs), len(chunks), round(time.perf_counter() - t0, 3))

    # --- shared pipeline pieces --------------------------------------------------------------

    def _scope(self, user: User) -> frozenset[str]:
        return frozenset(r.value for r in user.visible_regions())

    def _retrieve(self, question: str, user: User) -> tuple[list[float], list[RetrievedChunk]]:
        q_vec = self.provider.embed_one(question)
        kwargs: dict[str, Any] = {}
        if isinstance(self.retriever, HybridRetriever | VectorRetriever):
            kwargs["query_vector"] = q_vec
        hits = self.retriever.search(question, user, self.settings.retrieval_k, **kwargs)
        return q_vec, hits

    def retrieve(self, question: str, user: User) -> list[RetrievedChunk]:
        """Retrieval only (no cache, no model). Used by the eval to score retrievers cheaply."""
        return self._retrieve(question, user)[1]

    def _gate(self, hits: list[RetrievedChunk]) -> bool:
        """True if retrieval is confident enough to involve the model."""
        return bool(hits) and hits[0].score >= self.settings.refusal_threshold

    @staticmethod
    def _trace(hits: list[RetrievedChunk], selected: list[RetrievedChunk]) -> list[RetrievalTrace]:
        chosen = {h.chunk.chunk_id for h in selected}
        return [
            RetrievalTrace(
                chunk_id=h.chunk.chunk_id,
                doc_id=h.chunk.doc_id,
                section=h.chunk.section,
                score=round(h.score, 6),
                sources=h.sources or {h.source: h.rank},
                in_context=h.chunk.chunk_id in chosen,
            )
            for h in hits
        ]

    def _refusal(self, hits: list[RetrievedChunk], usage: TokenUsage, t0: float) -> AskResult:
        return AskResult(
            answer=REFUSAL_TEXT,
            citations=[],
            refused=True,
            tokens=usage,
            cost_usd=self.pricing.cost(usage),
            retrieved=self._trace(hits, []),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    # --- ask ---------------------------------------------------------------------------------

    def ask(self, question: str, user: User) -> AskResult:
        t0 = time.perf_counter()
        usage = TokenUsage(embedding=self.chunker.count_tokens(question))
        q_vec, hits = self._retrieve(question, user)

        cached = self.cache.get(q_vec, self._scope(user))
        if cached is not None:
            return cached.model_copy(
                update={
                    "cached": True,
                    "cost_usd": 0.0,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                }
            )

        if not self._gate(hits):
            result = self._refusal(hits, usage, t0)
            self.cache.put(q_vec, self._scope(user), result)
            return result

        selected, ctx_tokens = pack(hits, self.settings.context_budget_tokens)
        prompt = build_prompt(question, selected)
        completion = self.provider.complete(SYSTEM_PROMPT, prompt)
        answer, citations, refused = ground(completion.text, selected)
        usage = TokenUsage(
            prompt=completion.prompt_tokens,
            completion=completion.completion_tokens,
            embedding=usage.embedding,
            context=ctx_tokens,
        )
        result = AskResult(
            answer=answer,
            citations=citations,
            refused=refused,
            tokens=usage,
            cost_usd=self.pricing.cost(usage),
            retrieved=self._trace(hits, selected),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        self.cache.put(q_vec, self._scope(user), result)
        return result

    def ask_stream(self, question: str, user: User) -> Iterator[dict[str, Any]]:
        """Yield SSE-ready events: {"type": "meta"}, then {"type": "token"}*, then {"type": "done"}.

        The retrieval gate runs *before* any token is streamed, so a refusal never starts as a
        half-answer. Citation grounding runs on the full text at the end; the client should
        replace what it has displayed with the final ``answer`` in the ``done`` event.
        """
        t0 = time.perf_counter()
        usage = TokenUsage(embedding=self.chunker.count_tokens(question))
        q_vec, hits = self._retrieve(question, user)

        cached = self.cache.get(q_vec, self._scope(user))
        if cached is not None:
            result = cached.model_copy(update={"cached": True, "cost_usd": 0.0})
            yield {"type": "meta", "retrieved": [t.model_dump() for t in result.retrieved]}
            yield {"type": "token", "text": result.answer}
            yield {"type": "done", "result": result.model_dump(mode="json")}
            return

        if not self._gate(hits):
            result = self._refusal(hits, usage, t0)
            self.cache.put(q_vec, self._scope(user), result)
            yield {"type": "meta", "retrieved": [t.model_dump() for t in result.retrieved]}
            yield {"type": "token", "text": result.answer}
            yield {"type": "done", "result": result.model_dump(mode="json")}
            return

        selected, ctx_tokens = pack(hits, self.settings.context_budget_tokens)
        yield {"type": "meta", "retrieved": [t.model_dump() for t in self._trace(hits, selected)]}
        prompt = build_prompt(question, selected)
        pieces: list[str] = []
        for piece in self.provider.stream(SYSTEM_PROMPT, prompt):
            pieces.append(piece)
            yield {"type": "token", "text": piece}
        full = "".join(pieces)
        answer, citations, refused = ground(full, selected)
        usage = TokenUsage(
            prompt=self.chunker.count_tokens(SYSTEM_PROMPT) + self.chunker.count_tokens(prompt),
            completion=self.chunker.count_tokens(full),
            embedding=usage.embedding,
            context=ctx_tokens,
        )
        result = AskResult(
            answer=answer,
            citations=citations,
            refused=refused,
            tokens=usage,
            cost_usd=self.pricing.cost(usage),
            retrieved=self._trace(hits, selected),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        self.cache.put(q_vec, self._scope(user), result)
        yield {"type": "done", "result": result.model_dump(mode="json")}
