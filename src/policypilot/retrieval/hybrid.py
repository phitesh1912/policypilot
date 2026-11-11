"""Hybrid retrieval: BM25 + dense vectors merged with Reciprocal Rank Fusion.

RRF (Cormack et al., 2009): fused(d) = Σ_r 1 / (k + rank_r(d)) over every ranking r the document
appears in. It needs no score normalisation — BM25 scores and cosine similarities live on
incomparable scales — and it rewards documents that both retrievers agree on. k=60 is the value
from the paper; it damps the advantage of being rank 1 vs rank 2.

Useful reference points with k=60: rank 1 in one list only = 0.0164; rank 1 in both = 0.0328;
rank 1 + rank 10 = 0.0307. So a fused-score threshold of ~0.02 means "the top chunk must appear
in *both* top-k lists", which is the retrieval-level refusal gate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from policypilot.models import Chunk, RetrievedChunk, User
from policypilot.retrieval.base import Retriever
from policypilot.retrieval.bm25 import BM25Retriever
from policypilot.retrieval.vector import VectorRetriever


def rrf_fuse(rankings: Sequence[Sequence[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    """Fuse several rankings into one. Deterministic: ties broken by chunk_id."""
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    per_source: dict[str, dict[str, int]] = defaultdict(dict)
    for ranking in rankings:
        for rc in ranking:
            cid = rc.chunk.chunk_id
            scores[cid] += 1.0 / (k + rc.rank)
            chunks[cid] = rc.chunk
            per_source[cid][rc.source] = rc.rank
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        RetrievedChunk(
            chunk=chunks[cid], score=s, rank=r, source="hybrid", sources=dict(per_source[cid])
        )
        for r, (cid, s) in enumerate(ordered, start=1)
    ]


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        rrf_k: int = 60,
        candidates_per_retriever: int | None = None,
    ) -> None:
        self.bm25 = bm25
        self.vector = vector
        self.rrf_k = rrf_k
        self.candidates_per_retriever = candidates_per_retriever

    def index(self, chunks: list[Chunk]) -> None:
        self.bm25.index(chunks)
        self.vector.index(chunks)

    def search(
        self,
        query: str,
        user: User,
        k: int = 10,
        *,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        n = self.candidates_per_retriever or k
        lexical = self.bm25.search(query, user, n)
        dense = self.vector.search(query, user, n, query_vector=query_vector)
        return rrf_fuse([lexical, dense], k=self.rrf_k)[:k]
