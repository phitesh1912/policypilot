"""Semantic LRU cache, implemented by hand: a dict for O(1) node lookup plus a doubly linked list
for O(1) recency updates and eviction.

Lookup is *semantic*: a query hits if some cached query embedding has cosine similarity ≥
``threshold``. That makes lookup O(n·d) (a scan over n cached vectors of dimension d) rather than
O(1); with n ≤ a few hundred this is microseconds and far cheaper than an LLM call. For larger
caches the scan would move into the vector DB.

Entries are partitioned by ``scope`` — the set of regions the asking user may see — so an answer
built from US-only documents can never be served to an IN employee who asks the same question.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable
from typing import Generic, TypeVar

import numpy as np

V = TypeVar("V")


class _Node(Generic[V]):
    __slots__ = ("key", "embedding", "scope", "value", "prev", "next")

    def __init__(self, key: int, embedding: np.ndarray, scope: Hashable, value: V) -> None:
        self.key = key
        self.embedding = embedding
        self.scope = scope
        self.value = value
        self.prev: _Node[V] | None = None
        self.next: _Node[V] | None = None


class LRUSemanticCache(Generic[V]):
    def __init__(self, capacity: int = 256, threshold: float = 0.95) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.capacity = capacity
        self.threshold = threshold
        self._map: dict[int, _Node[V]] = {}
        self._ids = itertools.count()
        # Sentinels: head.next is most-recent, tail.prev is least-recent.
        self._head: _Node[V] = _Node(-1, np.empty(0), None, None)  # type: ignore[arg-type]
        self._tail: _Node[V] = _Node(-2, np.empty(0), None, None)  # type: ignore[arg-type]
        self._head.next = self._tail
        self._tail.prev = self._head
        self.hits = 0
        self.misses = 0

    # --- linked-list primitives --------------------------------------------------------------

    def _unlink(self, node: _Node[V]) -> None:
        assert node.prev is not None and node.next is not None
        node.prev.next = node.next
        node.next.prev = node.prev

    def _push_front(self, node: _Node[V]) -> None:
        first = self._head.next
        assert first is not None
        node.prev = self._head
        node.next = first
        self._head.next = node
        first.prev = node

    def _evict_lru(self) -> None:
        lru = self._tail.prev
        assert lru is not None and lru is not self._head
        self._unlink(lru)
        del self._map[lru.key]

    # --- public API --------------------------------------------------------------------------

    @staticmethod
    def _as_unit(embedding: list[float] | np.ndarray) -> np.ndarray:
        v = np.asarray(embedding, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    def get(self, embedding: list[float] | np.ndarray, scope: Hashable) -> V | None:
        q = self._as_unit(embedding)
        best: _Node[V] | None = None
        best_sim = self.threshold
        for node in self._map.values():
            if node.scope != scope:
                continue
            sim = float(np.dot(q, node.embedding))
            if sim >= best_sim:
                best, best_sim = node, sim
        if best is None:
            self.misses += 1
            return None
        self.hits += 1
        self._unlink(best)
        self._push_front(best)
        return best.value

    def put(self, embedding: list[float] | np.ndarray, scope: Hashable, value: V) -> None:
        node = _Node(next(self._ids), self._as_unit(embedding), scope, value)
        self._map[node.key] = node
        self._push_front(node)
        if len(self._map) > self.capacity:
            self._evict_lru()

    def clear(self) -> None:
        self._map.clear()
        self._head.next = self._tail
        self._tail.prev = self._head

    def __len__(self) -> int:
        return len(self._map)

    def keys_mru_first(self) -> list[int]:
        """Insertion keys in recency order; used by tests to assert LRU behaviour."""
        out: list[int] = []
        node = self._head.next
        while node is not None and node is not self._tail:
            out.append(node.key)
            node = node.next
        return out
