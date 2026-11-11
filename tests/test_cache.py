import numpy as np
import pytest

from policypilot.cache import LRUSemanticCache


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim)
    v[seed % dim] = 1.0
    return v


def test_exact_hit_and_miss():
    c = LRUSemanticCache(capacity=4, threshold=0.95)
    c.put(_vec(0), "s", "a")
    assert c.get(_vec(0), "s") == "a"
    assert c.get(_vec(1), "s") is None
    assert (c.hits, c.misses) == (1, 1)


def test_near_duplicate_hits_within_threshold():
    c = LRUSemanticCache(capacity=4, threshold=0.9)
    base = np.array([1.0, 0.0, 0.0])
    c.put(base, "s", "v")
    near = np.array([0.98, 0.1, 0.0])  # cos ≈ 0.995
    far = np.array([0.5, 0.5, 0.0])  # cos ≈ 0.707
    assert c.get(near, "s") == "v"
    assert c.get(far, "s") is None


def test_scope_isolation_prevents_cross_region_leak():
    c = LRUSemanticCache(capacity=4, threshold=0.9)
    c.put(_vec(0), frozenset({"US", "GLOBAL"}), "us-answer")
    assert c.get(_vec(0), frozenset({"IN", "GLOBAL"})) is None
    assert c.get(_vec(0), frozenset({"US", "GLOBAL"})) == "us-answer"


def test_lru_eviction_order():
    c = LRUSemanticCache(capacity=3, threshold=0.99)
    c.put(_vec(0), "s", 0)
    c.put(_vec(1), "s", 1)
    c.put(_vec(2), "s", 2)
    assert c.keys_mru_first() == [2, 1, 0]
    c.get(_vec(0), "s")  # touch 0 -> most recent
    assert c.keys_mru_first() == [0, 2, 1]
    c.put(_vec(3), "s", 3)  # evicts 1 (least recent)
    assert len(c) == 3
    assert c.get(_vec(1), "s") is None
    assert c.get(_vec(2), "s") == 2
    assert c.keys_mru_first() == [2, 3, 0]


def test_capacity_one():
    c = LRUSemanticCache(capacity=1, threshold=0.99)
    c.put(_vec(0), "s", "a")
    c.put(_vec(1), "s", "b")
    assert len(c) == 1
    assert c.get(_vec(0), "s") is None
    assert c.get(_vec(1), "s") == "b"


def test_clear():
    c = LRUSemanticCache(capacity=3, threshold=0.99)
    c.put(_vec(0), "s", 0)
    c.clear()
    assert len(c) == 0
    assert c.get(_vec(0), "s") is None
    c.put(_vec(1), "s", 1)
    assert c.keys_mru_first() == [1]


def test_best_match_wins_not_first_match():
    c = LRUSemanticCache(capacity=4, threshold=0.5)
    c.put(np.array([1.0, 0.0]), "s", "x-axis")
    c.put(np.array([0.0, 1.0]), "s", "y-axis")
    assert c.get(np.array([0.2, 1.0]), "s") == "y-axis"


def test_invalid_params():
    with pytest.raises(ValueError):
        LRUSemanticCache(capacity=0)
    with pytest.raises(ValueError):
        LRUSemanticCache(threshold=0.0)
    with pytest.raises(ValueError):
        LRUSemanticCache(threshold=1.5)
