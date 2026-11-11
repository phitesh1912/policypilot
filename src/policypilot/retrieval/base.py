"""Retriever interface. BM25, vector and hybrid retrievers all implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod

from policypilot.models import Chunk, RetrievedChunk, User


class Retriever(ABC):
    name: str

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """(Re)build the index from scratch."""

    @abstractmethod
    def search(self, query: str, user: User, k: int = 10) -> list[RetrievedChunk]:
        """Return the top-``k`` chunks the ``user`` is allowed to see, best first.

        Access control is applied *inside* retrieval, not as a post-filter, so that a user in
        region IN can never have a US-only chunk occupy one of their ``k`` slots.
        """


def allowed(chunk: Chunk, user: User) -> bool:
    return chunk.region in user.visible_regions()
