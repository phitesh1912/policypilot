"""Dense retrieval over Qdrant (embedded on disk, in-memory for tests, or Qdrant Cloud)."""

from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from policypilot.llm.base import LLMProvider
from policypilot.models import Chunk, RetrievedChunk, User
from policypilot.retrieval.base import Retriever

_NS = uuid.UUID("8f1b6e2a-5c3d-4e7f-9a1b-2c3d4e5f6a7b")


def point_id(chunk_id: str) -> str:
    """Qdrant wants UUID/int ids; derive a stable UUID from the human-readable chunk id."""
    return str(uuid.uuid5(_NS, chunk_id))


def make_client(*, url: str | None = None, api_key: str | None = None, path: Path | None = None):
    if url:
        return QdrantClient(url=url, api_key=api_key)
    if path is None:
        return QdrantClient(":memory:")
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


class VectorRetriever(Retriever):
    name = "vector"

    def __init__(
        self,
        provider: LLMProvider,
        client: QdrantClient | None = None,
        collection: str = "policies",
    ) -> None:
        self.provider = provider
        self.client = client or QdrantClient(":memory:")
        self.collection = collection

    def index(self, chunks: list[Chunk]) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=self.provider.embed_dim, distance=Distance.COSINE),
        )
        if not chunks:
            return
        vectors = self.provider.embed([c.text for c in chunks])
        points = [
            PointStruct(id=point_id(c.chunk_id), vector=v, payload=c.model_dump(mode="json"))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[i : i + 256])

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def search(self, query: str, user: User, k: int = 10) -> list[RetrievedChunk]:
        if not self.client.collection_exists(self.collection):
            return []
        # Access control is a pre-filter inside the ANN search: forbidden regions never compete.
        region_filter = Filter(
            must=[
                FieldCondition(
                    key="region", match=MatchAny(any=[r.value for r in user.visible_regions()])
                )
            ]
        )
        result = self.client.query_points(
            self.collection,
            query=self.provider.embed_one(query),
            query_filter=region_filter,
            limit=k,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                chunk=Chunk.model_validate(p.payload),
                score=float(p.score),
                rank=r,
                source=self.name,
            )
            for r, p in enumerate(result.points, start=1)
        ]
